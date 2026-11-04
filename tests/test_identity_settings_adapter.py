"""Space-specific policy selection and structured source-quality conversion."""

from pathlib import Path

import pytest
from test_sample_test import make_input, read_json, settings_for, write_json

from app.adapters.sample_test import load_sample_test
from app.contracts.evidence import CleanReviewStatus, ScoreSemantics
from app.contracts.identity import MatchingPolicy
from app.settings import SettingsError, load_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_TEXT = (PROJECT_ROOT / "config.example.toml").read_text()


def configured(tmp_path, extension="", *, enabled=False):
    text = CONFIG_TEXT.replace(
        "auto_assign_enabled = false", f"auto_assign_enabled = {str(enabled).lower()}", 1
    )
    path = tmp_path / "settings.toml"
    path.write_text(text + extension)
    return load_settings(path)


def test_unconfigured_space_uses_uncalibrated_shared_defaults(tmp_path):
    settings = configured(tmp_path)
    policy = settings.matching.policy_for("model-space")
    assert policy.feature_space_id == "model-space"
    assert policy.min_common_parts == settings.matching.min_common_parts
    assert policy.max_templates_per_player == settings.matching.max_templates_per_player
    assert policy.auto_assign_enabled is False
    assert policy.tau is None and policy.delta is None
    assert policy.calibration is None


def test_explicit_space_policy_inherits_shared_fields_and_is_selected_exactly(tmp_path):
    settings = configured(
        tmp_path,
        '\n[[matching.policies]]\nfeature_space_id="space-a"\nsample_bucket_s=0.75\n',
    )
    policy = settings.matching.policy_for("space-a")
    assert policy.sample_bucket_s == 0.75
    assert policy.min_common_parts == 2
    assert settings.matching.policy_for("space-b").sample_bucket_s == 0.5


@pytest.mark.parametrize(
    "extension",
    [
        '\n[[matching.policies]]\nfeature_space_id="a"\nmin_common_parts=3\n',
        '\n[[matching.policies]]\nfeature_space_id="a"\n'
        '[[matching.policies]]\nfeature_space_id="a"\n',
        '\n[[matching.policies]]\nfeature_space_id="a"\nunknown_threshold=1.0\n',
        '\n[[matching.policies]]\nfeature_space_id="a"\nmin_support_rate=1.1\n',
        '\n[[matching.policies]]\nfeature_space_id="a"\nsample_bucket_s="0.5"\n',
    ],
)
def test_invalid_or_conflicting_space_configuration_fails(tmp_path, extension):
    with pytest.raises(SettingsError):
        configured(tmp_path, extension)


@pytest.mark.parametrize("extension", ["", '\n[[matching.policies]]\nfeature_space_id="a"\n'])
def test_enabling_auto_assignment_without_calibration_fails(tmp_path, extension):
    with pytest.raises(SettingsError, match="calibrat"):
        configured(tmp_path, extension, enabled=True)


def test_calibrated_space_is_explicit_and_cannot_be_reused_for_another_space(tmp_path):
    # Synthetic calibration assertion tests configuration binding, not real accuracy.
    policy = MatchingPolicy(feature_space_id="a", tau=0.1, delta=0.05)
    extension = (
        '\n[[matching.policies]]\nfeature_space_id="a"\ntau=0.1\ndelta=0.05\n'
        '[matching.policies.calibration]\nstatus="validated"\n'
        'dataset_version="synthetic-policy-test"\n'
        f'policy_fingerprint="{policy.fingerprint}"\n'
    )
    settings = configured(tmp_path, extension, enabled=True)
    assert settings.matching.policy_for("a").calibrated
    assert settings.matching.policy_for("a").auto_assign_enabled
    with pytest.raises(SettingsError, match="feature_space_id 'b'"):
        settings.matching.policy_for("b")
    changed = extension.replace('feature_space_id="a"', 'feature_space_id="b"')
    with pytest.raises(SettingsError, match="fingerprint"):
        configured(tmp_path, changed, enabled=True)


def test_adapter_preserves_score_semantics_and_does_not_imply_manual_approval(tmp_path):
    root = make_input(tmp_path / "input")
    batch = load_sample_test(settings_for(root, tmp_path), batch_id="quality")
    evidence = batch.identity_evidence
    for sample in evidence.samples:
        quality = sample.tracking_quality
        assert quality.score == 0.9
        assert quality.score_semantics == ScoreSemantics.DETECTION
        assert quality.clean_review == CleanReviewStatus.UNREVIEWED
        if sample.clip_id == "beta":
            assert [risk.kind for risk in quality.risks] == ["quality_review"]
            assert quality.risks[0].segment_id == sample.segment_id
            assert quality.risks[0] in evidence.risks
        else:
            assert quality.risks == []
        assert quality.mixed_identity is None


def test_adapter_translates_quality_tags_with_segment_scope_and_keeps_provenance(tmp_path):
    root = make_input(tmp_path / "input")
    path = root / "target" / "alpha" / "selection.json"
    selection = read_json(path)
    selection["track_segments"][0]["quality_review_reasons"] = (
        "mixed_identity;duplicate_box_candidate;unrecognized_source_reason"
    )
    write_json(path, selection)
    batch = load_sample_test(settings_for(root, tmp_path), batch_id="quality")
    evidence = batch.identity_evidence
    alpha = [sample for sample in evidence.samples if sample.clip_id == "alpha"]
    risk_kinds = {risk.kind for risk in alpha[0].tracking_quality.risks}
    assert risk_kinds == {"mixed_identity", "duplicate_box", "quality_review"}
    assert all(sample.tracking_quality.mixed_identity for sample in alpha)
    assert all(sample.tracking_quality.clean_review == "unreviewed" for sample in alpha)
    for risk in alpha[0].tracking_quality.risks:
        assert risk.start_s == 0.0 and risk.end_s == 2.5
        assert risk.source.locations[0].record_key == "track_segments[0]"
    before = path.read_bytes()
    again = load_sample_test(settings_for(root, tmp_path), batch_id="another-batch")
    assert again.identity_evidence.risks == evidence.risks
    assert path.read_bytes() == before
