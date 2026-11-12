"""Sample input assembly yields actual, safely validated feature rows."""

import numpy as np
import pytest
from test_sample_test import arrays, make_input, replace_arrays, settings_for

from app.adapters.sample_test import SampleTestError, load_sample_test, load_sample_test_input
from app.contracts.identity import MatchingPolicy
from app.identity.prepare import prepare_evidence, resolve_feature
from app.storage.database import connect, transaction
from app.storage.features import save_feature_value
from app.storage.repositories import Repository
from app.storage.snapshot import read_identity_snapshot


def test_assembled_features_preserve_npz_rows_and_variable_shape(tmp_path):
    root = make_input(tmp_path / "input", parts=4, dimension=7)
    data = arrays(root)
    data["embeddings"][1, :, :] = 0.0
    data["embeddings"][1, :, 1] = 1.0
    data["visibility_scores"][0, 1] = 0.25
    replace_arrays(root, data)
    settings = settings_for(root, tmp_path)
    batch, values = load_sample_test_input(settings, batch_id="ready", clip_ids=["alpha"])
    assert batch == load_sample_test(settings, batch_id="ready", clip_ids=["alpha"])
    evidence = batch.identity_evidence
    space = evidence.feature_spaces[0]
    assert space.shape == [4, 7]
    assert len(values) == len(evidence.samples)
    for sample in evidence.samples:
        ref = sample.features[0]
        feature = values[(ref.feature_file_id, ref.row_index)]
        np.testing.assert_array_equal(
            np.asarray(feature.values).reshape(space.shape), data["embeddings"][ref.row_index]
        )
        assert feature.visibility == data["visibility_scores"][ref.row_index].tolist()
        assert resolve_feature(sample, space, values) == feature
        assert sample.tracking_quality.clean_review == "unreviewed"
    prepared = prepare_evidence(
        batch, MatchingPolicy(feature_space_id=space.feature_space_id), feature_values=values
    )
    assert sum(len(segment.samples) for segment in prepared.segments) == len(evidence.samples)
    assert not settings.paths.runtime_root.exists()


def test_assembled_features_can_be_saved_and_reopened(tmp_path):
    root = make_input(tmp_path / "input")
    batch, values = load_sample_test_input(settings_for(root, tmp_path), batch_id="ready")
    path = tmp_path / "library.sqlite3"
    connection = connect(path)
    try:
        repo = Repository(connection)
        with transaction(connection):
            repo.ensure_match(batch.match_id)
            repo.store_batch(batch)
            evidence = batch.identity_evidence
            spaces = {space.feature_space_id: space for space in evidence.feature_spaces}
            files = {file.feature_file_id: file for file in evidence.feature_files}
            for (file_id, row), value in values.items():
                file = files[file_id]
                assert save_feature_value(
                    connection, batch.match_id, file, spaces[file.feature_space_id], row, value
                )
            repo.advance_revision(
                batch.match_id, 0, actor="test", reason="assembled input", changes={"input": True}
            )
    finally:
        connection.close()
    connection = connect(path)
    try:
        snapshot = read_identity_snapshot(connection, batch.match_id)
        assert snapshot.feature_values == values
        assert {sample.sample_id: sample for sample in snapshot.gallery.evidence.samples} == {
            sample.sample_id: sample for sample in batch.identity_evidence.samples
        }
        assert not snapshot.gallery.players
        assert not snapshot.gallery.templates
    finally:
        connection.close()


@pytest.mark.parametrize("failure", ["hash", "object", "symlink"])
def test_feature_assembly_uses_safe_adapter_validation(tmp_path, failure):
    root = make_input(tmp_path / "input")
    npz = root / "target" / "alpha" / "features.npz"
    if failure == "hash":
        with npz.open("ab") as file:
            file.write(b"tampered")
    elif failure == "object":
        data = arrays(root)
        data["sample_ids"] = data["sample_ids"].astype(object)
        replace_arrays(root, data)
    else:
        outside = tmp_path / "outside.npz"
        npz.rename(outside)
        npz.symlink_to(outside)
    settings = settings_for(root, tmp_path)
    with pytest.raises(SampleTestError):
        load_sample_test_input(settings, batch_id="unsafe")
    assert not settings.paths.runtime_root.exists()
