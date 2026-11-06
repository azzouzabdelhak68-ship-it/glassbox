# THREAT MODEL — seal review (Phase 4, plan §3.4 sign-off)

> Scope: v1 local hash chain (no HSM/Sigstore — documented, not pretended).
> Method: enumerate trust boundaries, attacker capabilities, mitigations,
> residual risks. Re-verify with the commands at the bottom after any
> change to `storage/seal.py`, `tap/sidecar.py`, or `tap/wrapper.py`.

## Trust boundaries

1. **Model process** (untrusted for integrity: user code, hooks, adapters run here).
2. **Sidecar / WAL** (witness: separate process/container, append-only, fsync per token).
3. **Sealed store** (`glassbox_store/`: `deep.bin`, `deep_index.json`, `seal.json`, `chain.jsonl`).
4. **Analyst** (reads via `verify` / `find` / `view`; trusts nothing without `chain OK`).

## Attacker table

| Attacker action | Capability assumed | Mitigation (in code) | Residual risk (v1) |
|---|---|---|---|
| Edit `outer.json` after seal | write access to store | `verify` recomputes `SHA256(outer)` → FAIL (exit 7); chain walk breaks at link | None if analyst runs `verify` (UI shows red badge otherwise — never blank, never silent) |
| Edit `deep.bin` after seal | write access to store | `sha_deep` covers raw bytes; 1-byte tamper tested (`test_deep_integration`) | Same as above |
| Reorder/drop a run from `chain.jsonl` | write access to store | `prev_hash` linkage; `verify --all` reports `SUMMARY ... FAIL` | Genesis/first-link trust is local (no external anchor in v1) |
| Kill main (`SIGKILL`/OOM) mid-run | crash the model process | WAL rows fsync'd per token; `Sidecar.attach + seal_prefix` seals the prefix; `verify` passes for it with E1001 resume hint | Unflushed ring slots (≤3 tokens) lost — bounded, disclosed in the hint |
| Kill the sidecar | crash the witness | Ring is file-backed mmap; new sidecar re-attaches; sealed prefix intact | Tokens after last poll unsealed until re-poll |
| Fill the disk (TB-scale watchlist) | legit but greedy run | Estimator RED>1000GB requires `--force`; `retention_state.json` flags; never auto-deletes | `--force` is explicit user responsibility (flag, never block) |
| Bypass `redactor=` to exfiltrate prompts | malicious caller code | Hook identity recorded in `outer.json:versions.redactor`; sealed bytes are post-hook (auditable what ran) | Box provides the hook point, not a mandated filter — caller owns the filter (privacy lock) |
| Hook throws / ring overflows | faulty model or burst | Auto-`detach()`, `hook_detached`/`ring_overflow` flags; Outer still seals; job continues | Degraded Deep (documented in flags, visible in viewer) |
| Replay attack (old sealed run as new) | copy sealed dirs | `run_id` + chain position bind order; duplicate `run_id` fails alloc | Cross-store copies detectable only by checkpoint+chain inspection |

## Explicit non-goals (v1 honesty locks)

- No HSM/Sigstore/external timestamping: a root attacker on the store host can
  rewrite history wholesale. v1 detects edits, not host compromise.
- No access control on `$GLASSBOX_DIR`: filesystem permissions are the
  operator's job (`chmod`, mount options).
- No redaction, no encryption at rest, no per-user schemas (see
  `docs/FORMAT_GUIDE.md`: we document, you adapt).
- Never run on private user traffic without legal basis (README + viewer
  footer carry the warning on every render).

## Sign-off checklist (re-run on seal-path changes)

```bash
GLASSBOX_GPU=cpu pytest tests/test_seal_verify.py tests/test_deep_integration.py -q
python -m glassbox verify --all                      # expect SUMMARY: N OK, 0 FAIL
grep -rn "synchronize" glassbox/tap/hooks.py         # must be empty
```
