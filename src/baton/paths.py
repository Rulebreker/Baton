"""Filesystem locations and safe-write helpers.

Everything Baton persists (vault, usage state, logs, checkpoints) lives under
one private directory *outside* the repository, so a careless `git add -A`
cannot pick it up.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

HOME_ENV = "BATON_HOME"


def baton_home() -> Path:
    """Return Baton's data directory (not created)."""
    override = os.environ.get(HOME_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".baton"


def ensure_private_dir(path: Path) -> Path:
    """Create `path` readable by the owner only.

    On POSIX this is a real 0700. On Windows `chmod` only toggles the read-only
    bit, so protection comes from the user-profile ACL; the README says so
    instead of pretending otherwise.
    """
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        try:
            path.chmod(stat.S_IRWXU)
        except OSError:
            pass
    return path


def secure_write(path: Path, data: bytes) -> None:
    """Atomically write `data` to `path` with owner-only permissions.

    Write to a temp file in the same directory, fsync, then `os.replace`. A
    crash or power cut leaves either the old file or the new one, never a
    truncated vault or half-written usage state.
    """
    # Only a directory we create ourselves is made private. An existing parent
    # is left alone: with `--config /some/project/baton.yaml` the parent is
    # the user's own directory, and silently chmod-ing it to 0700 would be a
    # nasty side effect (it only ever happened on POSIX, where chmod is real).
    if not path.parent.is_dir():
        ensure_private_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:  # mkstemp creates the file 0600
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
