"""Hash seal + chain verify for Deep-Only runs (spec §FILE-seal, §A.5).

Normative formula (Deep-Only)::

    hash_i=SHA256(prev||SHA256(deep_index)||SHA256(deep)||canonical(replay))

``SHA256(deep)`` is the SHA-256 of ``deep.bin`` bytes, or the empty-string hash
when ``deep.bin`` is absent (``triggered=false``). ``canonical(replay)`` is
canonical JSON (sorted keys, no whitespace) of ``deep_index.json:replay``.
Genesis ``prev`` is 64x``0``.

Concurrency & scale:
- ``_last_chain_hash`` reads from the tail of ``chain.jsonl`` in O(1) time.
- ``seal_run`` acquires an exclusive lock on ``chain.jsonl.lock`` so concurrent
  sealing workers never fork the linear hash chain.

Verify recomputes 5 steps: index sha, deep sha, canonical replay, ``hash_i``,
then compare + chain continuity. Any 1-byte edit to ``deep.bin`` or
``deep_index.json`` -> FAIL.

Exit mapping for CLI: E06 ``RunNotFoundError`` (unknown run), E07 FAIL
(``verify()`` returns ``False``; ``SealBrokenError`` is programmatic twin).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

from glassbox.exceptions import RunNotFoundError
from glassbox.storage.layout import get_store_dir, run_dir

GENESIS_PREV = "0" * 64
ALGORITHM = "hash_i=SHA256(prev||SHA256(deep_index)||SHA256(deep)||canonical(replay))"
EMPTY_SHA = hashlib.sha256(b"").hexdigest()


def _sha_file(path: str) -> str:
    """SHA256 hex of file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_replay(replay: dict[str, Any]) -> str:
    """Canonical JSON: sorted keys, no whitespace."""
    return json.dumps(replay, sort_keys=True, separators=(",", ":"))


def compute_hash(prev: str, sha_index: str, sha_deep: str, replay: dict[str, Any]) -> str:
    """Compute ``hash_i`` from parts per the normative Deep-Only formula."""
    return hashlib.sha256(
        (prev + sha_index + sha_deep + canonical_replay(replay)).encode("utf-8")
    ).hexdigest()


def _chain_path_for_run(run_path: str) -> str:
    """``chain.jsonl`` for a run dir (``$BLACKBOX_DIR/chain.jsonl``)."""
    abs_run = os.path.abspath(run_path)
    store = os.path.dirname(os.path.dirname(abs_run))
    return os.path.join(store, "chain.jsonl")


@contextlib.contextmanager
def _chain_lock(chain_path: str) -> Iterator[None]:
    """Cross-platform exclusive lock on ``chain.jsonl.lock``."""
    os.makedirs(os.path.dirname(chain_path) or ".", exist_ok=True)
    lock_path = chain_path + ".lock"
    try:
        from filelock import FileLock

        with FileLock(lock_path, timeout=10):
            yield
        return
    except ImportError:
        pass

    deadline = time.monotonic() + 10.0
    fd: int | None = None
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except OSError:
            if time.monotonic() >= deadline:
                break
            time.sleep(0.005)
    try:
        yield
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
            with contextlib.suppress(OSError):
                os.unlink(lock_path)


def _last_chain_hash(chain_path: str) -> str:
    """Last ``hash`` in chain in O(1) tail seek, or genesis when absent/empty."""
    if not os.path.isfile(chain_path):
        return GENESIS_PREV
    try:
        with open(chain_path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            pos = fh.tell()
            buf = b""
            while pos > 0:
                step = min(4096, pos)
                pos -= step
                fh.seek(pos)
                buf = fh.read(step) + buf
                lines = [ln.strip() for ln in buf.splitlines() if ln.strip()]
                for raw_line in reversed(lines[1:] if pos > 0 and len(lines) > 1 else lines):
                    try:
                        return str(json.loads(raw_line.decode("utf-8"))["hash"])
                    except (ValueError, KeyError, TypeError, UnicodeDecodeError):
                        continue
    except OSError:
        return GENESIS_PREV
    return GENESIS_PREV


def seal_run(run_path: str, sealed_by: str = "glassbox-deep") -> str:
    """Seal a Deep run dir: write ``seal.json`` + append ``chain.jsonl``.

    ``SHA256(deep_index)`` covers ``deep_index.json`` bytes and ``SHA256(deep)``
    covers ``deep.bin`` bytes (or empty-string hash when ``triggered=false``);
    ``verify_run`` walks the same chain, so any 1-byte edit fails.

    Returns:
        The ``hash_i`` hex string.

    Raises:
        FileNotFoundError: E02/E06 missing ``deep_index.json``.
        ValueError: E02 missing ``replay`` object in ``deep_index.json``.
    """
    index_path = os.path.join(run_path, "deep_index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"E02: no deep_index.json in {run_path}")
    with open(index_path, encoding="utf-8") as fh:
        index = json.load(fh)
    if not isinstance(index.get("replay"), dict):
        raise ValueError(f"E02: deep_index.json missing replay in {run_path}")
    run_id = str(index.get("run_id", os.path.basename(os.path.normpath(run_path))))
    sha_index = _sha_file(index_path)
    deep_path = os.path.join(run_path, "deep.bin")
    sha_deep = _sha_file(deep_path) if os.path.isfile(deep_path) else EMPTY_SHA

    chain_path = _chain_path_for_run(run_path)
    with _chain_lock(chain_path):
        prev = _last_chain_hash(chain_path)
        digest = compute_hash(prev, sha_index, sha_deep, index["replay"])
        seal: dict[str, Any] = {
            "run_id": run_id,
            "hash": digest,
            "prev_hash": prev,
            "sha_index": sha_index,
            "sha_deep": sha_deep,
            "algorithm": ALGORITHM,
            "sealed_by": sealed_by,
            "sealed_at_iso": datetime.now(timezone.utc).isoformat(),
        }
        with open(os.path.join(run_path, "seal.json"), "w", encoding="utf-8") as fh:
            json.dump(seal, fh, indent=2, sort_keys=True)
            fh.write("\n")
        with open(chain_path, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"run_id": run_id, "hash": digest, "prev_hash": prev}, sort_keys=True)
                + "\n"
            )
    return digest


def verify_run(run_id: str, store: str | None = None) -> bool:
    """Recompute the 5 seal steps; True=OK, False=FAIL.

    Raises:
        RunNotFoundError: E06 unknown run (missing run dir).
    """
    root = get_store_dir(store)
    path = run_dir(root, run_id)
    if not os.path.isdir(path):
        raise RunNotFoundError(f"E06: {run_id} not found in {root}/runs/.")
    index_path = os.path.join(path, "deep_index.json")
    seal_path = os.path.join(path, "seal.json")
    if not os.path.isfile(index_path) or not os.path.isfile(seal_path):
        return False
    try:
        with open(seal_path, encoding="utf-8") as fh:
            seal = json.load(fh)
        with open(index_path, encoding="utf-8") as fh:
            index = json.load(fh)
    except (OSError, ValueError):
        return False
    if not isinstance(index.get("replay"), dict):
        return False
    sha_index = _sha_file(index_path)
    deep_path = os.path.join(path, "deep.bin")
    if bool(index.get("triggered")) and not os.path.isfile(deep_path):
        return False
    sha_deep = _sha_file(deep_path) if os.path.isfile(deep_path) else EMPTY_SHA
    if sha_index != seal.get("sha_index") or sha_deep != seal.get("sha_deep"):
        return False
    recomputed = compute_hash(
        str(seal.get("prev_hash", "")),
        sha_index,
        sha_deep,
        index["replay"],
    )
    if recomputed != seal.get("hash"):
        return False

    chain_path = os.path.join(root, "chain.jsonl")
    if not os.path.isfile(chain_path):
        return False
    expected_prev = GENESIS_PREV
    with open(chain_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                return False
            if entry.get("prev_hash") != expected_prev:
                return False
            if entry.get("run_id") == run_id:
                return bool(
                    entry.get("hash") == recomputed and seal.get("prev_hash") == expected_prev
                )
            expected_prev = str(entry.get("hash", ""))
    return False
