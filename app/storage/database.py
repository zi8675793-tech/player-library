"""SQLite connections and atomic, ordered schema migrations.

Only an explicit connection creates the runtime database. Repository functions do
not commit; callers own transaction boundaries.
"""

import hashlib
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.settings import Settings

DATABASE_FILENAME = "player-library.sqlite3"
MIGRATIONS_DIR = Path(__file__).with_name("migrations")
_MIGRATION_NAME = re.compile(r"^(\d+)_.+\.sql$")


class MigrationError(RuntimeError):
    """Migration sequence or previously applied migration is inconsistent."""


def database_path(settings: Settings) -> Path:
    """Determine the runtime path without creating any files or directories."""
    return settings.paths.runtime_root / DATABASE_FILENAME


def connect(path: str | Path, *, timeout: float = 5.0) -> sqlite3.Connection:
    """Open a connection with foreign keys and migrate it; close it on failure."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=timeout, isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {max(0, int(timeout * 1000))}")
        migrate(connection)
    except BaseException:
        connection.close()
        raise
    return connection


def _statements(script: str) -> Iterator[str]:
    statement = ""
    for line in script.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            yield statement
            statement = ""
    if statement.strip():
        raise MigrationError("incomplete SQL statement in migration")


def migrate(connection: sqlite3.Connection, migrations_dir: str | Path = MIGRATIONS_DIR) -> None:
    """Apply all unapplied migrations in one transaction; never rebuild data.

    SQLite's executescript commits implicitly, so individual complete statements
    are executed instead. A failure rolls back the entire upgrade.
    """
    if connection.in_transaction:
        raise RuntimeError("migrations require a connection outside a transaction")
    files = []
    for path in Path(migrations_dir).iterdir():
        match = _MIGRATION_NAME.fullmatch(path.name)
        if match:
            files.append((int(match.group(1)), path))
    files.sort(key=lambda entry: entry[0])
    if len({version for version, _ in files}) != len(files):
        raise MigrationError("duplicate migration version")
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version INTEGER PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL)"
        )
        applied = {
            row[0]: (row[1], row[2])
            for row in connection.execute("SELECT version, name, checksum FROM schema_migrations")
        }
        available = {version for version, _ in files}
        if set(applied) - available:
            raise MigrationError("database contains migrations absent from migration directory")
        if applied and any(version < max(applied) for version in available - set(applied)):
            raise MigrationError("cannot insert a migration before an applied version")
        for version, path in files:
            script = path.read_text(encoding="utf-8")
            checksum = hashlib.sha256(script.encode("utf-8")).hexdigest()
            if version in applied:
                if applied[version] != (path.name, checksum):
                    raise MigrationError(f"applied migration {version} was changed")
                continue
            for statement in _statements(script):
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations(version, name, checksum) VALUES (?, ?, ?)",
                (version, path.name, checksum),
            )
        if 4 in available and 4 not in applied:
            from app.storage.revision_states import capture_state

            for row in connection.execute("SELECT match_id,revision FROM matches").fetchall():
                capture_state(connection, row[0], row[1], legacy_checkpoint=True)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


@contextmanager
def transaction(
    connection: sqlite3.Connection, *, immediate: bool = True
) -> Iterator[sqlite3.Connection]:
    """Commit a caller-owned transaction, or roll it all back on exception.

    Use immediate=False for a consistent read snapshot; the snapshot is fixed
    when the first query executes. Nested transactions must use a savepoint.
    """
    if connection.in_transaction:
        raise RuntimeError("connection already has an active transaction")
    connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
    try:
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
