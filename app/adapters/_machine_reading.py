"""A positive file allowlist for the machine handoff, independent of answer readers."""

from pathlib import Path

from app.adapters._reading import InputReadError, ReadOnlyInput


def clip_component(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in {".", "..", "target", "repository-review"}
        or any(char in value for char in ("/", "\\", "\x00"))
    ):
        raise InputReadError(f"invalid machine clip/path component: {value!r}")
    return value


class MachineInputReader(ReadOnlyInput):
    """Only known machine files can be opened, including through file()/data().

    No symlinks inside the input tree are accepted. The shared reader additionally
    opens canonical components using O_NOFOLLOW to defend against swaps at open.
    Directory discovery lists names only; unlisted files are never hashed/read.
    """

    def __init__(self, root: Path):
        if root.is_symlink():
            raise InputReadError("machine input root must not be a symbolic link")
        super().__init__(root)
        if {"target", "repository-review"}.intersection(self.root.parts):
            raise InputReadError("forbidden machine input root")

    @staticmethod
    def _allowed(parts: tuple[str, ...], directory: bool) -> bool:
        if not parts:
            return False
        kind = parts[0]
        if kind not in {"track", "identity", "video"}:
            return False
        if directory and len(parts) == 1:
            return True
        if len(parts) >= 2:
            clip_component(parts[1] if kind != "video" else Path(parts[1]).stem)
        if directory:
            return (kind in {"track", "identity"} and len(parts) == 2) or (
                kind == "track" and len(parts) == 3 and parts[2] == "samples"
            )
        if kind == "video":
            return len(parts) == 2 and Path(parts[1]).suffix == ".mp4"
        fixed = (
            {"tracks.csv", "tracks.jsonl", "video_meta.json"}
            if kind == "track"
            else {"samples.jsonl", "feature_contract.json", "features.npz"}
        )
        return (len(parts) == 3 and parts[2] in fixed) or (
            kind == "track"
            and len(parts) == 4
            and parts[2] == "samples"
            and Path(parts[3]).suffix == ".png"
        )

    def path(self, reference: str | Path, *, directory: bool = False) -> Path:
        raw = Path(reference)
        if raw.is_absolute() or ".." in raw.parts or not self._allowed(raw.parts, directory):
            raise InputReadError(f"{reference}: not in machine input file allowlist")
        current = self.root
        for part in raw.parts:
            current = current / part
            if current.is_symlink():
                raise InputReadError(f"{reference}: symbolic links are forbidden in machine inputs")
        path = super().path(reference, directory=directory)
        if path.relative_to(self.root) != raw:
            raise InputReadError(f"{reference}: machine input canonical path changed")
        return path

    def discover_clips(self) -> list[str]:
        groups = {}
        for kind in ("track", "identity"):
            directory = self.path(kind, directory=True)
            names = set()
            for path in directory.iterdir():
                if path.is_symlink():
                    raise InputReadError(f"{kind}/{path.name}: symbolic links are forbidden")
                if path.is_dir():
                    names.add(clip_component(path.name))
            groups[kind] = names
        video = self.path("video", directory=True)
        names = set()
        for path in video.iterdir():
            if path.is_symlink():
                raise InputReadError(f"video/{path.name}: symbolic links are forbidden")
            if path.is_file() and path.suffix == ".mp4":
                names.add(clip_component(path.stem))
        groups["video"] = names
        if not groups["track"] or any(value != groups["track"] for value in groups.values()):
            raise InputReadError(f"machine clip sets must be nonempty and agree: {groups}")
        return sorted(groups["track"])
