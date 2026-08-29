from __future__ import annotations

import atexit
import os
from pathlib import Path
from threading import Lock
from typing import BinaryIO


class WriterLeaseError(RuntimeError):
    pass


_guard = Lock()
_handle: BinaryIO | None = None
_leased_database: Path | None = None
_leased_pid: int | None = None


def require_writer_lease(database: Path) -> None:
    global _handle, _leased_database, _leased_pid
    resolved = database.resolve()
    pid = os.getpid()
    with _guard:
        if _handle is not None and _leased_pid == pid:
            if _leased_database != resolved:
                raise WriterLeaseError(f"process already owns a different database: {_leased_database}")
            return
        lease_path = resolved.with_suffix(resolved.suffix + ".writer.lock")
        lease_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = lease_path.open("r+b")
        except FileNotFoundError:
            try:
                handle = lease_path.open("x+b")
            except FileExistsError:
                handle = lease_path.open("r+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            _lock_first_byte(handle)
        except OSError as error:
            handle.close()
            raise WriterLeaseError(
                f"database writer already owned by another process: {resolved}"
            ) from error
        handle.seek(0)
        handle.write(str(pid).encode("ascii").ljust(32, b" "))
        handle.truncate(32)
        handle.flush()
        _handle = handle
        _leased_database = resolved
        _leased_pid = pid


def release_writer_lease() -> None:
    global _handle, _leased_database, _leased_pid
    with _guard:
        if _handle is None:
            return
        try:
            _handle.seek(0)
            _unlock_first_byte(_handle)
        finally:
            _handle.close()
            _handle = None
            _leased_database = None
            _leased_pid = None


def writer_lease_status() -> dict[str, object]:
    with _guard:
        return {
            "owned_by_process": bool(_handle is not None and _leased_pid == os.getpid()),
            "pid": _leased_pid if _handle is not None else None,
            "database": str(_leased_database) if _leased_database is not None else None,
        }


def _lock_first_byte(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_first_byte(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


atexit.register(release_writer_lease)
