"""GlassBox public API — Zero-overhead flight recorder & interpretability engine.

:meth:`GlassBox.watch` declares a Deep watchlist with precedence:
call kwargs > watchlist file > preset expansion > built-ins
(``trigger=always``, ``window=all tokens every 1``, ``precision=fp16``);
``__enter__`` validates addresses vs manifest, checks checkpoint drift, and
runs the estimator (RED + no force -> ``SizeFlagError``) BEFORE any
streaming starts (fail fast, nothing sealed). When ``model=`` is passed (or
``ctx.attach(model)`` is called), :class:`~glassbox.tap.hooks.DeepTap` hooks
are attached directly to the model's watched parent modules. ``__exit__``
detaches hooks, writes ``deep.bin`` + unified ``deep_index.json`` (including
prompt/output/timing/replay context), seals ``seal.json`` + ``chain.jsonl``,
and updates the Hot search index.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from types import TracebackType
from typing import Any

from glassbox.controller import (
    check_checkpoint,
    estimate_size,
    parse_trigger,
    precision_bytes,
)
from glassbox.exceptions import (
    BlackBoxError,
    CheckpointMismatchError,
    GlassBoxError,
    ManifestExistsError,
    RunNotFoundError,
    SealBrokenError,
    SizeFlagError,
    UnknownAddressError,
)
from glassbox.scan import scan as _scan_impl
from glassbox.search.index import append_hot
from glassbox.storage.layout import (
    allocate_run,
    default_run_context,
    get_store_dir,
    write_absent_deep_index,
    write_deep_bin,
    write_deep_index,
)
from glassbox.storage.seal import seal_run, verify_run
from glassbox.tap.hooks import DeepTap
from glassbox.to_dataframe import SealBrokenWarning, to_dataframe
from glassbox.watchlist import (
    PRESETS,
    expand_addresses,
    is_single_address,
    parse_watchlist,
    validate_addresses,
)

__version__ = "0.1.0"

__all__ = [
    "BlackBox",
    "BlackBoxError",
    "CheckpointMismatchError",
    "DeepTap",
    "GlassBox",
    "GlassBoxError",
    "ManifestExistsError",
    "RunNotFoundError",
    "SealBrokenError",
    "SealBrokenWarning",
    "SizeFlagError",
    "UnknownAddressError",
    "__version__",
    "estimate_bytes",
    "flag_for_bytes",
    "lens",
    "profile",
    "scan",
    "to_dataframe",
    "verify_run",
    "watch",
]

_TAG_RE = re.compile(r"^[a-z0-9_]+=.+$")
_VALID_ADDRESS_HINT = "Valid: N-01-0001..N-32-14336, A-01-01..A-32-32, LOGITS, EMBED."
_DEFAULT_ARCH = {"layers": 32, "hidden": 4096, "heads": 32, "mlp_per_layer": 14336, "vocab": 32000}


def estimate_bytes(w: int, t: int, r: int, b: int) -> float:
    """Estimator ``bytes = W*T*R*B*1.15`` (spec section CTRL-estimator)."""
    total, _ = estimate_size(w, t, r, b)
    return total


def flag_for_bytes(nbytes: float) -> str:
    """GREEN (<50GB) / YELLOW (50-1000GB) / RED (>1000GB)."""
    if nbytes < 50e9:
        return "GREEN"
    if nbytes <= 1000e9:
        return "YELLOW"
    return "RED"


def validate_tag_dict(tag: dict[str, str] | None) -> list[str]:
    """Validate ``tag`` dict keys/values; returns sorted ``["k=v"]`` list."""
    if not tag:
        return []
    out: list[str] = []
    for key, value in tag.items():
        item = f"{key}={value}"
        if not _TAG_RE.match(item):
            raise ValueError(f"E02: bad tag {item!r}; expected [a-z0-9_]+=value")
        out.append(item)
    return sorted(out)


def _find_on_disk_manifest(
    model_id: str | None = None,
    checkpoint_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Find a matching manifest in ``manifests/*/manifest.json`` if present."""
    import glob

    candidates: list[dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join("manifests", "*", "manifest.json"))):
        try:
            with open(path, encoding="utf-8") as fh:
                man = json.load(fh)
            if not isinstance(man, dict) or "architecture" not in man:
                continue
            if model_id and man.get("model_id") == model_id:
                if not checkpoint_sha256 or man.get("checkpoint_sha256") == checkpoint_sha256:
                    return man
            candidates.append(man)
        except (OSError, ValueError):
            continue
    return candidates[0] if len(candidates) == 1 and not checkpoint_sha256 else None


def _tokenize_utf8(text: str) -> list[int]:
    if not text:
        return []
    return list(text.encode("utf-8"))


class _WatchContext:
    """Deep-Only context manager returned by :meth:`BlackBox.watch` (spec section A.9)."""

    def __init__(
        self,
        watchlist: str,
        add: list[str] | None = None,
        window: dict[str, Any] | None = None,
        trigger: str | dict[str, Any] | None = None,
        precision: str | None = None,
        force: bool = False,
        tag: dict[str, str] | None = None,
        *,
        model: Any = None,
        manifest: dict[str, Any] | str | None = None,
        tokens: int | None = None,
        runs: int = 1,
        seed: int = 42,
        prompt: str = "",
        capture_mode: str = "all",
        head_reduction: str = "l2_norm",
        batch_index: int = 0,
    ) -> None:
        self.watchlist = watchlist
        self.add = list(add) if add else []
        self.window = dict(window) if window else {}
        self.trigger = trigger
        self.precision = precision
        self.force = force
        self.tag = dict(tag) if tag else {}
        self.model = model
        self.manifest_src = manifest
        self.tokens_override = tokens
        self.runs = int(runs)
        self.seed = int(seed)
        self.prompt_text = str(prompt)
        self.output_text = ""
        self.capture_mode = capture_mode
        self.head_reduction = head_reduction
        self.batch_index = int(batch_index)

        self.run_id: str | None = None
        self.run_dir: str | None = None
        self.seal: dict[str, Any] | None = None
        self.tap: DeepTap | None = None
        self._wl: dict[str, Any] = {}
        self._manifest: dict[str, Any] | None = None
        self._t0_iso: str = ""
        self._t0_mono: float = 0.0
        self._orig_generate: Any = None
        self._orig_forward: Any = None
        self._attached_model: Any = None

    def _resolve_manifest(self, raw_yaml: dict[str, Any]) -> dict[str, Any]:
        if isinstance(self.manifest_src, dict):
            return self.manifest_src
        if isinstance(self.manifest_src, str) and os.path.isfile(self.manifest_src):
            with open(self.manifest_src, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                return loaded
        file_model_id = str(raw_yaml.get("model_id", "M-001"))
        file_sha = str(raw_yaml.get("checkpoint_sha256", ""))
        found = _find_on_disk_manifest(file_model_id, file_sha)
        if found is not None:
            return found
        any_found = _find_on_disk_manifest(file_model_id, None)
        if any_found is not None and file_sha and any_found.get("checkpoint_sha256") != file_sha:
            raise CheckpointMismatchError(
                f"E04: checkpoint mismatch expected {any_found.get('checkpoint_sha256')} "
                f"got {file_sha}. Re-scan; old watchlists invalid."
            )
        return {
            "model_id": file_model_id,
            "checkpoint_sha256": file_sha or ("0" * 64),
            "architecture": dict(_DEFAULT_ARCH),
        }

    def __enter__(self) -> _WatchContext:
        import yaml

        validate_tag_dict(self.tag)
        if not os.path.isfile(self.watchlist):
            raise FileNotFoundError(f"E02: watchlist file not found: {self.watchlist}")
        with open(self.watchlist, encoding="utf-8") as fh:
            raw_yaml = yaml.safe_load(fh)
        if not isinstance(raw_yaml, dict):
            raise ValueError(f"E02: bad watchlist {self.watchlist}")

        # Quick check for unknown address tokens before preset expansion
        for item in self.add:
            try:
                for exp in expand_addresses([str(item)]):
                    if not is_single_address(exp):
                        raise UnknownAddressError(
                            f"E04: unknown address {item}; {_VALID_ADDRESS_HINT}"
                        )
            except ValueError as exc:
                raise UnknownAddressError(
                    f"E04: unknown address {item}; {_VALID_ADDRESS_HINT}"
                ) from exc
        for entry in raw_yaml.get("entries", []) or []:
            if isinstance(entry, dict) and "addresses" in entry:
                for raw_addr in entry.get("addresses", []) or []:
                    try:
                        expand_addresses([str(raw_addr)])
                    except ValueError as exc:
                        raise UnknownAddressError(
                            f"E04: unknown address {raw_addr}; {_VALID_ADDRESS_HINT}"
                        ) from exc
            if isinstance(entry, dict) and "preset" in entry:
                if str(entry["preset"]) not in PRESETS:
                    raise ValueError(f"E02: unknown preset {entry['preset']!r}")

        self._manifest = self._resolve_manifest(raw_yaml)
        self._wl = parse_watchlist(
            self.watchlist,
            manifest=self._manifest,
            add=self.add or None,
            window=self.window or None,
            trigger=self.trigger,
            precision=self.precision,
        )
        check_checkpoint(self._manifest, self._wl)
        validate_addresses(list(self._wl["addresses"]), self._manifest)
        eff_trigger = parse_trigger(self._wl.get("effective_trigger"))
        eff_precision = str(self._wl.get("effective_precision", "fp16"))
        b = precision_bytes(eff_precision)

        merged_win = self._wl["merged_window"]
        if self.tokens_override is not None:
            t_est = int(self.tokens_override)
        else:
            t_est = max(1, int(merged_win["end_token"]) - int(merged_win["start_token"]))
        w_count = len(self._wl["addresses"])
        total_bytes, flag = estimate_size(w_count, t_est, max(1, self.runs), b)
        if flag == "RED" and not self.force:
            raise SizeFlagError(
                f"E05: RED-flagged {total_bytes / 1e9:.1f}GB estimate "
                f"(W={w_count} T={t_est} R={self.runs} B={b}); require --force. Nothing ran."
            )

        arch = self._manifest.get("architecture", _DEFAULT_ARCH)
        num_heads = int(arch.get("heads", 32))
        self.tap = DeepTap(
            list(self._wl["addresses"]),
            window=merged_win,
            trigger=eff_trigger,
            dtype=eff_precision,
            num_heads=num_heads,
            head_reduction=self.head_reduction,
            capture_mode=self.capture_mode,
            batch_index=self.batch_index,
        )
        self._t0_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        self._t0_mono = time.monotonic()

        if self.model is not None:
            self.attach(self.model)
        return self

    def attach(self, model: Any) -> DeepTap:
        """Attach DeepTap hooks to ``model`` and wrap its ``generate``/``forward``."""
        if self.tap is None:
            raise RuntimeError("BlackBox.watch context must be entered before attach()")
        self.tap.attach(model)
        self._attached_model = model
        if hasattr(model, "generate") and callable(model.generate):
            orig_gen = model.generate
            self._orig_generate = orig_gen

            def _wrapped_generate(*args: Any, **kwargs: Any) -> Any:
                if args and isinstance(args[0], str) and not self.prompt_text:
                    self.prompt_text = args[0]
                elif isinstance(kwargs.get("prompt"), str) and not self.prompt_text:
                    self.prompt_text = str(kwargs["prompt"])
                res = orig_gen(*args, **kwargs)
                if self.tap is not None and self.tap._staging_seq:
                    self.tap.flush_sequence()
                if isinstance(res, str):
                    self.output_text = res
                elif hasattr(res, "text") and isinstance(res.text, str):
                    self.output_text = res.text
                return res

            try:
                model.generate = _wrapped_generate
            except Exception:
                pass
        if hasattr(model, "forward") and callable(model.forward):
            orig_fwd = model.forward
            self._orig_forward = orig_fwd

            def _wrapped_forward(*args: Any, **kwargs: Any) -> Any:
                res = orig_fwd(*args, **kwargs)
                if self.tap is not None and self.tap._staging_seq:
                    self.tap.flush_sequence()
                return res

            try:
                model.forward = _wrapped_forward
            except Exception:
                pass
        return self.tap

    def record_synthetic(self, n_tokens: int = 16, seed: int | None = None) -> int:
        """Populate deterministic architecture Deep activations when no live model ran."""
        if self.tap is None:
            return 0
        rng = random.Random(self.seed if seed is None else seed)
        committed = 0
        for pos in range(n_tokens):
            vals = {addr: round(rng.uniform(-1.0, 1.0), 4) for addr in self.tap.watched}
            if self.tap.end_token(pos, values=vals):
                committed += 1
        return committed

    def _restore_model(self) -> None:
        if self.tap is not None:
            self.tap.detach()
        if self._attached_model is not None:
            if self._orig_generate is not None:
                try:
                    self._attached_model.generate = self._orig_generate
                except Exception:
                    pass
            if self._orig_forward is not None:
                try:
                    self._attached_model.forward = self._orig_forward
                except Exception:
                    pass
        self._attached_model = None
        self._orig_generate = None
        self._orig_forward = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        _ = (exc_type, exc_tb)
        if self.tap is not None and self.tap._staging_seq:
            self.tap.flush_sequence()
        self._restore_model()

        store = get_store_dir()
        run_id, run_dir = allocate_run(store)
        t_end_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        elapsed_ms = max(0.0, (time.monotonic() - self._t0_mono) * 1000.0)
        tags = validate_tag_dict(self.tag)

        model_id = str(self._wl.get("model_id", "M-001"))
        ckpt_sha = str(self._wl.get("checkpoint_sha256", "0" * 64))
        watchlist_id = str(self._wl.get("watchlist_id", "deep"))
        watchlist_sha = str(
            self._wl.get(
                "watchlist_sha256",
                hashlib.sha256(self.watchlist.encode("utf-8")).hexdigest(),
            )
        )
        patch = self._wl.get("patch")

        error: dict[str, Any] | None = None
        if exc_val is not None:
            error = {"code": type(exc_val).__name__, "message": str(exc_val)}

        tap = self.tap
        n_captured = len(tap.ring) if tap is not None else 0
        prompt_tokens = _tokenize_utf8(self.prompt_text)
        output_tokens = (
            _tokenize_utf8(self.output_text)
            if self.output_text
            else list(tap.token_positions if tap is not None else range(n_captured))
        )

        ctx = default_run_context(
            run_id,
            model_id=model_id,
            checkpoint_sha256=ckpt_sha,
            watchlist_ref={
                "watchlist_id": watchlist_id,
                "watchlist_sha256": watchlist_sha,
                "patch": patch,
            },
            prompt={
                "text": self.prompt_text,
                "tokens": prompt_tokens,
                "tokenizer": "utf8-bytes",
            },
            output={
                "text": self.output_text,
                "tokens": output_tokens,
                "finish_reason": "error" if error else "stop",
            },
            logits_summary=list(tap.logits_summary) if tap and tap.logits_summary else None,
            timing={
                "t_start_iso": self._t0_iso,
                "t_end_iso": t_end_iso,
                "ms_per_token": round(elapsed_ms / max(1, n_captured), 4),
                "cuda_event_ms": round(elapsed_ms, 4),
            },
            sampling={
                "seed": self.seed,
                "temp": 0.0,
                "top_p": 1.0,
                "sampler": "greedy",
            },
            flags={
                "bad_output": error is not None,
                "ring_overflow": bool(tap.flags.get("ring_overflow", False)) if tap else False,
                "hook_detached": bool(tap.flags.get("hook_detached", False)) if tap else False,
                "truncated": False,
            },
            tags=tags,
            replay={
                "seed": self.seed,
                "temp": 0.0,
                "top_p": 1.0,
                "sampler": "greedy",
                "cudnn_deterministic": True,
                "checkpoint_sha256": ckpt_sha,
                "versions": {"torch": "unknown", "cuda": "cpu"},
            },
            error=error,
        )

        if tap is not None and n_captured > 0:
            mat = tap.to_matrix()
            write_deep_bin(run_dir, mat, tap.dtype)
            write_deep_index(
                run_dir,
                run_id,
                triggered=True,
                dtype=tap.dtype,
                tokens=n_captured,
                every_n=int(tap.window.get("every_n", 1)),
                watched=tap.watched,
                window=tap.window,
                token_positions=tap.token_positions,
                capture_mode=tap.capture_mode,
                dtype_meta=tap.dtype_meta,
                context=ctx,
            )
        else:
            eff_dtype = tap.dtype if tap is not None else "fp16"
            eff_watched = tap.watched if tap is not None else list(self._wl.get("addresses", []))
            eff_every_n = int(tap.window.get("every_n", 1)) if tap is not None else 1
            write_absent_deep_index(
                run_dir,
                run_id,
                dtype=eff_dtype,
                every_n=eff_every_n,
                watched=eff_watched,
                context=ctx,
            )

        digest = seal_run(run_dir)
        seal_path = os.path.join(run_dir, "seal.json")
        with open(seal_path, encoding="utf-8") as fh:
            self.seal = json.load(fh)
        self.seal.setdefault("hash", digest)
        try:
            append_hot(run_id, store)
        except Exception:
            pass
        self.run_id = run_id
        self.run_dir = run_dir


class GlassBox:
    """Public Deep-Only API surface (spec sections A.9-A.12)."""

    @staticmethod
    def watch(
        watchlist: str,
        add: list[str] | None = None,
        window: dict[str, Any] | None = None,
        trigger: str | dict[str, Any] | None = None,
        precision: str | None = None,
        force: bool = False,
        tag: dict[str, str] | None = None,
        *,
        model: Any = None,
        manifest: dict[str, Any] | str | None = None,
        tokens: int | None = None,
        runs: int = 1,
        seed: int = 42,
        prompt: str = "",
        capture_mode: str = "all",
        head_reduction: str = "l2_norm",
        batch_index: int = 0,
    ) -> _WatchContext:
        """Declare a Deep watchlist; validates on ``__enter__``, seals on ``__exit__``."""
        return _WatchContext(
            watchlist,
            add=add,
            window=window,
            trigger=trigger,
            precision=precision,
            force=force,
            tag=tag,
            model=model,
            manifest=manifest,
            tokens=tokens,
            runs=runs,
            seed=seed,
            prompt=prompt,
            capture_mode=capture_mode,
            head_reduction=head_reduction,
            batch_index=batch_index,
        )

    @staticmethod
    def scan(
        path: str,
        output: str | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Scan local weights on CPU into a Deep manifest."""
        return _scan_impl(path, output=output, force=force)

    @staticmethod
    def verify(run_id: str, store: str | None = None) -> bool:
        """Verify the hash chain for ``run_id`` (True = chain OK)."""
        return verify_run(run_id, store)

    @staticmethod
    def to_dataframe(
        run_id: str,
        what: str = "deep",
        tokens: tuple[int, int] | None = None,
        addresses: list[str] | None = None,
        store: str | None = None,
    ) -> Any:
        """Dataframe view over a sealed Deep run (``what='deep'|'full'``)."""
        return to_dataframe(run_id, what=what, tokens=tokens, addresses=addresses, store=store)

    @staticmethod
    def lens(
        run_id: str,
        store: str | None = None,
        token_pos: int | None = None,
    ) -> dict[str, Any]:
        """Compute mechanistic logit lens and residual stream progression."""
        from glassbox.lens import compute_logit_lens

        return compute_logit_lens(run_id, store=store, token_pos=token_pos)

    @staticmethod
    def profile(
        model: Any | None,
        manifest: dict[str, Any],
        sample_inputs: list[str] | None = None,
        top_k: int = 1000,
        output_yaml: str | None = None,
    ) -> list[str]:
        """Calibrate and select the top-k most active components into a watchlist."""
        from glassbox.profile import export_profile_watchlist, profile_model

        addrs = profile_model(model, manifest, sample_inputs, top_k=top_k)
        if output_yaml:
            export_profile_watchlist(
                addrs,
                str(manifest.get("model_id", "M-001")),
                str(manifest.get("checkpoint_sha256", "0" * 64)),
                output_yaml,
            )
        return addrs


BlackBox = GlassBox  # Backward-compatible alias


def profile(
    model: Any | None,
    manifest: dict[str, Any],
    sample_inputs: list[str] | None = None,
    top_k: int = 1000,
    output_yaml: str | None = None,
) -> list[str]:
    """Module-level alias for :meth:`GlassBox.profile`."""
    return GlassBox.profile(
        model, manifest, sample_inputs=sample_inputs, top_k=top_k, output_yaml=output_yaml
    )


def lens(
    run_id: str,
    store: str | None = None,
    token_pos: int | None = None,
) -> dict[str, Any]:
    """Module-level alias for :meth:`GlassBox.lens`."""
    return GlassBox.lens(run_id, store=store, token_pos=token_pos)


def watch(
    watchlist: str,
    add: list[str] | None = None,
    window: dict[str, Any] | None = None,
    trigger: str | dict[str, Any] | None = None,
    precision: str | None = None,
    force: bool = False,
    tag: dict[str, str] | None = None,
    *,
    model: Any = None,
    manifest: dict[str, Any] | str | None = None,
    tokens: int | None = None,
    runs: int = 1,
    seed: int = 42,
    prompt: str = "",
    capture_mode: str = "all",
    head_reduction: str = "l2_norm",
    batch_index: int = 0,
) -> _WatchContext:
    """Module-level alias for :meth:`GlassBox.watch`."""
    return GlassBox.watch(
        watchlist,
        add=add,
        window=window,
        trigger=trigger,
        precision=precision,
        force=force,
        tag=tag,
        model=model,
        manifest=manifest,
        tokens=tokens,
        runs=runs,
        seed=seed,
        prompt=prompt,
        capture_mode=capture_mode,
        head_reduction=head_reduction,
        batch_index=batch_index,
    )


def scan(
    path: str,
    output: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Module-level alias for :meth:`GlassBox.scan`."""
    return GlassBox.scan(path, output=output, force=force)
