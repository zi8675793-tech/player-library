"""Read-only, root-confined readers shared by input adapters."""

import hashlib
import json
import math
import os
import stat
from contextlib import ExitStack, contextmanager
from pathlib import Path

from app.contracts.common import SourceLocation
from app.contracts.evidence import SourceFile


class InputReadError(ValueError):
    """An external input is missing, unsafe, or invalid."""


def fingerprint(value: object) -> str:
    """Hash a record, independently of its file's layout or other records."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def stable_id(kind: str, *key: object) -> str:
    return f"{kind}_{fingerprint(key)}"


class ReadOnlyInput:
    def __init__(self, root: Path):
        try:
            self.root = root.resolve(strict=True)
            if not self.root.is_dir():
                raise InputReadError(f"{root}: input root must be a directory")
        except (OSError, RuntimeError, ValueError) as exc:
            raise InputReadError(f"{root}: cannot resolve input root: {exc}") from exc
        self.sources: dict[str, SourceFile] = {}
        self._data: dict[str, bytes] = {}

    def path(self, reference: str | Path, *, directory: bool = False) -> Path:
        """External references are root-relative; resolve symlinks before any read."""
        raw = Path(reference)
        if not str(reference).strip() or raw.is_absolute() or ".." in raw.parts:
            raise InputReadError(f"{reference}: expected a root-relative path without traversal")
        try:
            path = (self.root / raw).resolve(strict=True)
            if not path.is_relative_to(self.root):
                raise InputReadError(f"{reference}: resolved path escapes allowed input root")
            if not (path.is_dir() if directory else path.is_file()):
                raise InputReadError(
                    f"{reference}: expected a {'directory' if directory else 'file'}"
                )
        except (OSError, RuntimeError, ValueError) as exc:
            raise InputReadError(f"{reference}: cannot read input: {exc}") from exc
        return path

    @contextmanager
    def _open(self, reference: str | Path):
        """Anchor every open to the root, refusing symlink swaps after resolution.

        Internal symlinks are allowed by path(), then their canonical components
        are opened with dir_fd/O_NOFOLLOW. A swapped directory cannot redirect a
        subsequent open outside the root. No file is opened with write flags.
        """
        path = self.path(reference)
        try:
            with ExitStack() as stack:
                directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                parent = os.open(self.root, directory_flags)
                stack.callback(os.close, parent)
                parts = path.relative_to(self.root).parts
                for part in parts[:-1]:
                    parent = os.open(part, directory_flags, dir_fd=parent)
                    stack.callback(os.close, parent)
                descriptor = os.open(
                    parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
                )
                stream = stack.enter_context(os.fdopen(descriptor, "rb"))
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise InputReadError(f"{reference}: expected a regular input file")
                yield stream
        except (OSError, ValueError) as exc:
            raise InputReadError(f"{reference}: cannot read input: {exc}") from exc

    def file(self, reference: str | Path, expected_hash: str | None = None) -> SourceFile:
        key = str(reference)
        if key not in self.sources:
            try:
                digest = hashlib.sha256()
                size = 0
                with self._open(reference) as stream:
                    while chunk := stream.read(1024 * 1024):
                        size += len(chunk)
                        digest.update(chunk)
            except OSError as exc:
                raise InputReadError(f"{reference}: cannot read input: {exc}") from exc
            self.sources[key] = SourceFile(
                file_id=stable_id("file", key), path=key, sha256=digest.hexdigest(), size_bytes=size
            )
        source = self.sources[key]
        if expected_hash is not None and source.sha256 != expected_hash:
            raise InputReadError(
                f"{reference}: sha256 mismatch: expected {expected_hash}, got {source.sha256}"
            )
        return source

    def data(self, reference: str | Path) -> bytes:
        key = str(reference)
        if key not in self._data:
            try:
                with self._open(reference) as stream:
                    data = stream.read()
            except OSError as exc:
                raise InputReadError(f"{reference}: cannot read input: {exc}") from exc
            self._data[key] = data
            source = SourceFile(
                file_id=stable_id("file", key),
                path=key,
                sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
            )
            previous = self.sources.get(key)
            if previous is not None and previous != source:
                raise InputReadError(f"{reference}: input changed while being read")
            self.sources[key] = source
        return self._data[key]

    def text(self, reference: str | Path) -> str:
        try:
            return self.data(reference).decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise InputReadError(f"{reference}: invalid UTF-8: {exc}") from exc

    def location(
        self, reference: str | Path, *, line: int | None = None, record: str | None = None
    ) -> SourceLocation:
        return SourceLocation(
            file_id=self.file(reference).file_id, line_number=line, record_key=record
        )

    def json(self, reference: str | Path) -> dict:
        return json_object(self.text(reference), str(reference))

    def jsonl(self, reference: str | Path) -> list[tuple[int, dict]]:
        rows = []
        for line, content in enumerate(self.text(reference).splitlines(), 1):
            rows.append((line, json_object(content, f"{reference}:{line}")))
        return rows


def json_object(content: str, context: str) -> dict:
    def reject_constant(value: str):
        raise ValueError(f"non-finite JSON number {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def finite_float(value: str):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite JSON number {value}")
        return number

    try:
        value = json.loads(
            content,
            parse_constant=reject_constant,
            parse_float=finite_float,
            object_pairs_hook=unique_keys,
        )
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (ValueError, RecursionError) as exc:
        raise InputReadError(f"{context}: invalid JSON: {exc}") from exc
