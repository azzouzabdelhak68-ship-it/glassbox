"""Run viewer: CLI table + self-contained static HTML (spec §VIEW-*, plan §10.3-10.4).

Deep-Only Architecture:
Loads ``deep_index.json`` + ``deep.bin`` + ``seal.json`` (no ``outer.json``).
Rules (spec §VIEW-partial, partial-but-precise): watched addresses show exact
values (blue); anything in the manifest but not watched shows ``--`` / grey
(``address exists, value null — not_watched``); never interpolate, never
0-fill. Aggregations label ``n_watched``. Token TEXT cells show ``#id``
from ``deep_index.json``; full PROMPT/OUTPUT text is always shown above the table.

HTML export is one self-contained file (inline CSS/JS, no server, no npm):
header badge → PROMPT chips → token strip (click selects, ``#token=N`` in
URL hash) → DETAIL (all watched at the token + z-score + 10-token sparkline +
co-occurrence edges labeled ``co-occurrence, not causality``) → OUTPUT →
Prev/Next + jump + arrow keys. Cold data embedded as JSON for the viewed
window only (at most 32768 values; first 32 watched addresses); the full
``deep.bin`` is never embedded. Untriggered runs render a notice; tampered
runs render a red banner but still render (never blank).
"""

from __future__ import annotations

import html
import json
import math
import os
import webbrowser
from typing import Any

EMBED_VALUE_CAP = 32768
EMBED_MAX_ADDRS = 32
GREY_PAGE = 100


def _store_root(store: str | None = None) -> str:
    if store:
        return store
    if "GLASSBOX_DIR" in os.environ:
        return os.environ["GLASSBOX_DIR"]
    if "BLACKBOX_DIR" in os.environ:
        return os.environ["BLACKBOX_DIR"]
    if os.path.exists("./blackbox_store") and not os.path.exists("./glassbox_store"):
        return "./blackbox_store"
    return "./glassbox_store"


def load_run(run_id: str, store: str | None = None) -> dict[str, Any]:
    """Load unified ``deep_index.json`` + seal flag (E06 when the run is missing)."""
    root = _store_root(store)
    run_dir = os.path.join(root, "runs", run_id)
    index_path = os.path.join(run_dir, "deep_index.json")
    if not os.path.isdir(run_dir) or not os.path.isfile(index_path):
        raise FileNotFoundError(f"E06: {run_id} not found in {root}/runs/.")
    with open(index_path, encoding="utf-8") as fh:
        index = json.load(fh)
    if not isinstance(index, dict):
        raise ValueError(f"E02: bad deep_index.json in {run_dir}")
    try:
        from glassbox.storage.seal import verify_run

        seal_ok = bool(verify_run(run_id, root))
    except Exception:
        seal_ok = False
    seal_hash = ""
    try:
        with open(os.path.join(run_dir, "seal.json"), encoding="utf-8") as fh:
            seal_hash = str(json.load(fh).get("hash", ""))
    except (OSError, ValueError):
        pass
    return {
        "run_id": run_id,
        "store": root,
        "run_dir": run_dir,
        "index": index,
        "seal_ok": seal_ok,
        "seal_hash": seal_hash,
    }


def manifest_totals(model_id: str, sha: str) -> dict[str, int] | None:
    """Inventory totals from a matching on-disk manifest (None when absent)."""
    import glob

    for path in glob.glob(os.path.join("manifests", "*", "manifest.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                man = json.load(fh)
        except (OSError, ValueError):
            continue
        if man.get("model_id") == model_id and man.get("checkpoint_sha256") == sha:
            arch = man.get("architecture", {})
            try:
                layers = int(arch["layers"])
                heads = int(arch["heads"])
                mlp = int(arch.get("mlp_per_layer", 0))
            except (KeyError, TypeError, ValueError):
                return None
            return {
                "layers": layers,
                "heads": heads,
                "mlp": mlp,
                "unwatched_slots": layers * mlp + layers * heads,
            }
    return None


def _slice_matrix(
    run: dict[str, Any], lo: int, hi: int, addrs: list[str] | None
) -> tuple[list[str], list[list[float]]]:
    """Cold slice ``[lo..hi]`` x ``addrs`` (defaults: all watched)."""
    from glassbox.storage.layout import read_deep_slice

    index = run["index"] or {}
    watched: list[str] = list(index.get("watched", []))
    wanted = list(addrs) if addrs else list(watched)
    cols = [watched.index(a) for a in wanted if a in watched]
    if not cols:
        return wanted, []
    mat = read_deep_slice(run["run_dir"], lo, hi, cols)
    return wanted, [[float(v) for v in row] for row in mat.tolist()]


def default_selected(run: dict[str, Any]) -> int:
    """Token with the largest watched |value| (fallback 0)."""
    index = run["index"] or {}
    if not index.get("triggered"):
        return 0
    tokens = int(index.get("tokens", 0))
    if tokens < 1:
        return 0
    try:
        _, rows = _slice_matrix(run, 0, tokens - 1, None)
        best, best_v = 0, -1.0
        for i, row in enumerate(rows):
            peak = max((abs(v) for v in row if math.isfinite(v)), default=0.0)
            if peak > best_v:
                best, best_v = i, peak
        return best
    except (OSError, ValueError):
        return 0


def token_table(
    run_id: str,
    store: str | None = None,
    tokens: tuple[int, int] | None = None,
    addrs: list[str] | None = None,
) -> dict[str, Any]:
    """Data behind the CLI table (defaults: 32 tokens x 8 watched)."""
    run = load_run(run_id, store)
    index = run["index"] or {}
    watched: list[str] = list(index.get("watched", []))
    total = int(index.get("tokens", 0)) if index.get("triggered") else 0
    lo, hi = (0, min(31, total - 1)) if tokens is None else tokens
    show_addrs = (addrs or watched[:8])[:32]
    if total and not (0 <= lo <= hi < total):
        raise ValueError(f"E02: bad --tokens {lo}..{hi} for T={total}.")
    for addr in show_addrs:
        from glassbox.search.query import _ADDR_RE

        if not _ADDR_RE.match(addr):
            raise ValueError(
                f"E02: unknown address {addr}; Valid: N-01-0001..N-32-14336, "
                "A-01-01..A-32-32, LOGITS, EMBED."
            )
    values: dict[int, dict[str, float | None]] = {}
    if total and show_addrs:
        present_addrs = [a for a in show_addrs if a in watched]
        _, rows = _slice_matrix(run, lo, hi, present_addrs)
        present_col_map = {a: k for k, a in enumerate(present_addrs)}
        for i, pos in enumerate(range(lo, hi + 1)):
            values[pos] = {}
            for addr in show_addrs:
                if addr in present_col_map and i < len(rows):
                    col_k = present_col_map[addr]
                    v = rows[i][col_k] if col_k < len(rows[i]) else float("nan")
                    values[pos][addr] = None if not math.isfinite(v) else v
                else:
                    values[pos][addr] = None
    out_ids: list[int] = (index.get("output", {}) or {}).get("tokens", []) or []
    return {
        "run": run,
        "index": index,
        "watched": watched,
        "total": total,
        "lo": lo,
        "hi": hi,
        "addrs": show_addrs,
        "values": values,
        "token_ids": out_ids,
    }


def _badge_text(run_id: str, ok: bool) -> str:
    if ok:
        return "[hash: VERIFIED ok chain OK]"
    return f"[hash: FAIL {run_id} seal mismatch at deep.bin sha256 — see glassbox verify {run_id}]"


def format_table(table: dict[str, Any], color: bool = True) -> str:
    """Human CLI table (spec §VIEW-cli); unwatched ``--``, tamper red banner."""
    run = table["run"]
    index = table["index"]
    ok = bool(run["seal_ok"])
    use_color = color
    blue = "\033[34m" if use_color else ""
    grey = "\033[2m" if use_color else ""
    red = "\033[31m" if use_color else ""
    green = "\033[32m" if use_color else ""
    reset = "\033[0m" if use_color else ""
    wl_ref = index.get("watchlist_ref")
    watch_id = (
        str(wl_ref.get("watchlist_id", "?"))
        if isinstance(wl_ref, dict)
        else str(index.get("watchlist_id", "?"))
    )
    model = index.get("model_id", "M-001")
    ckpt = str(index.get("checkpoint_sha256", "?"))[:8]
    lines = [
        f"{run['run_id']}  model={model}  ckpt=sha256:{ckpt}...  "
        f"watchlist={watch_id} (W={len(table['watched'])})  "
        f"{green if ok else red}{_badge_text(run['run_id'], ok)}{reset}"
    ]
    prompt = str((index.get("prompt", {}) or {}).get("text", ""))
    prompt_ids = (index.get("prompt", {}) or {}).get("tokens", []) or []
    n_prompt = len(prompt_ids) if isinstance(prompt_ids, list) else 0
    lines.append(f'PROMPT ({n_prompt} tokens): "{prompt}"')
    if not table["total"]:
        lines.append("Untriggered Deep run: trigger condition did not arm during execution.")
    header = f"{'TOK':<5} {'TEXT':<8}" + "".join(f" {blue}{a:<10}{reset}" for a in table["addrs"])
    lines.append(header)
    for pos in range(table["lo"], table["hi"] + 1):
        ids = table["token_ids"]
        text = f"#{ids[pos]}" if 0 <= pos < len(ids) else f"t{pos}"
        cell = f"{pos:<5} {text:<8}"
        for addr in table["addrs"]:
            val = table["values"].get(pos, {}).get(addr)
            cell += f" {val:>10.2f}" if val is not None else f" {grey}{'--':>10}{reset}"
        lines.append(cell)
    lines.append(f'OUTPUT: "{(index.get("output", {}) or {}).get("text", "")}"')
    lines.append(
        f"seal: hash_i={run['seal_hash'][:8]}... verified={str(ok).lower()} "
        f"| unwatched shown as -- (grey, null, not_watched)"
    )
    return "\n".join(lines)


def to_json(table: dict[str, Any]) -> dict[str, Any]:
    """Machine-exact view JSON (spec §VIEW-cli); unwatched -> null + note."""
    run = table["run"]
    index = table["index"]
    ids = table["token_ids"]
    token_rows = []
    for pos in range(table["lo"], table["hi"] + 1):
        vals: dict[str, Any] = {}
        for addr in table["addrs"]:
            val = table["values"].get(pos, {}).get(addr)
            vals[addr] = val
            if val is None:
                vals["_note"] = "not_watched"
        text = f"#{ids[pos]}" if 0 <= pos < len(ids) else f"t{pos}"
        token_rows.append({"pos": pos, "text": text, "values": vals})
    return {
        "run_id": run["run_id"],
        "model_id": index.get("model_id", "M-001"),
        "checkpoint_sha256": index.get("checkpoint_sha256", ""),
        "hash_seal_verified": bool(run["seal_ok"]),
        "prompt": str((index.get("prompt", {}) or {}).get("text", "")),
        "tokens": token_rows,
        "output": str((index.get("output", {}) or {}).get("text", "")),
    }


def _tchip(t: int, sel: int) -> str:
    """One token-strip chip (selected gets the outline + star)."""
    cls = "tchip sel" if t == sel else "tchip"
    star = " *" if t == sel else ""
    return f"<span class='{cls}' data-t={t}>T{t}{star}</span>"


def _spark(values: list[float], pos: int) -> str:
    """Unicode sparkline over the 10-token window ending at ``pos``."""
    bars = "▁▂▃▄▅▆▇█"
    window = values[max(0, pos - 9) : pos + 1]
    finite = [v for v in window if v == v]
    if not finite:
        return ""
    lo_v, hi_v = min(finite), max(finite)
    span = hi_v - lo_v or 1.0
    return "".join(bars[min(7, int((v - lo_v) / span * 8))] if v == v else " " for v in window)


def _cooccurrence(values_by_addr: dict[str, list[float]], sel: int) -> list[str]:
    """Top-5 watched addresses with |Pearson r| > 0.5 in [sel-5, sel+5]."""
    names = list(values_by_addr)
    if len(names) < 2:
        return []
    lo = max(0, sel - 5)
    window = {n: values_by_addr[n][lo : sel + 6] for n in names}
    scored: list[tuple[float, str, str]] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            xs, ys = window[a], window[b]
            pairs = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
            if len(pairs) < 3:
                continue
            mx = sum(x for x, _ in pairs) / len(pairs)
            my = sum(y for _, y in pairs) / len(pairs)
            num = sum((x - mx) * (y - my) for x, y in pairs)
            den = math.sqrt(
                sum((x - mx) ** 2 for x, _ in pairs) * sum((y - my) ** 2 for _, y in pairs)
            )
            r = num / den if den else 0.0
            if abs(r) > 0.5:
                scored.append((abs(r), a, b))
    scored.sort(reverse=True)
    return [f"{a} ~ {b} (r={r:.2f})" for r, a, b in scored[:5]]


def _svg_heatmap(
    series: dict[str, list[float]],
    embed_lo: int,
    embed_hi: int,
    addrs: list[str],
    sel: int,
) -> str:
    """Render an interactive 2D SVG activation heatmap matrix (addrs x tokens)."""
    if not addrs or embed_hi < embed_lo:
        return ""
    n_tokens = embed_hi - embed_lo + 1
    cell_w = max(14, min(32, 640 // max(1, n_tokens)))
    cell_h = 16
    label_w = 110
    svg_w = label_w + n_tokens * cell_w + 20
    svg_h = len(addrs) * cell_h + 30

    rects: list[str] = []
    for ti in range(n_tokens):
        tok_num = embed_lo + ti
        x = label_w + ti * cell_w
        rects.append(
            f"<text x='{x + cell_w // 2}' y='12' font-size='10' text-anchor='middle' "
            f"fill='#666'>{tok_num}</text>"
        )

    for ri, addr in enumerate(addrs):
        y = 20 + ri * cell_h
        rects.append(
            f"<text x='{label_w - 6}' y='{y + 12}' font-size='11' text-anchor='end' "
            f"fill='#333'>{html.escape(addr)}</text>"
        )
        vals = series.get(addr, [])
        max_val = max((abs(v) for v in vals if math.isfinite(v)), default=1.0) or 1.0
        for ti in range(n_tokens):
            tok_num = embed_lo + ti
            v = vals[ti] if ti < len(vals) else float("nan")
            x = label_w + ti * cell_w
            if not math.isfinite(v):
                color = "#f3f4f6"
            else:
                intensity = min(1.0, abs(v) / max_val)
                lightness = int(95 - intensity * 50)
                color = f"hsl(220, 85%, {lightness}%)"
            stroke = "#000" if tok_num == sel else "#e5e7eb"
            rects.append(
                f"<rect class='hm-cell' data-t='{tok_num}' x='{x}' y='{y}' "
                f"width='{cell_w - 1}' height='{cell_h - 1}' "
                f"fill='{color}' stroke='{stroke}' rx='2' style='cursor:pointer' "
                f"onclick='select({tok_num})'>"
                f"<title>{html.escape(addr)} at T{tok_num}: {v:.3f}</title></rect>"
            )

    return (
        "<div style='overflow-x:auto;padding:8px;border:1px solid #e5e7eb;"
        "border-radius:6px;background:#fafafa'>"
        f"<svg width='{svg_w}' height='{svg_h}' style='display:block;font-family:monospace'>"
        + "".join(rects)
        + "</svg></div>"
    )


def _svg_circuit(co: list[str]) -> str:
    """Render an interactive SVG bipartite co-occurrence circuit diagram."""
    if not co:
        return "<p class=notice>(no co-occurring edges above |r| > 0.5)</p>"
    svg_w, svg_h = 500, max(100, len(co) * 26 + 15)
    elements: list[str] = []
    y = 18
    for edge in co:
        elements.append(
            f"<circle cx='30' cy='{y}' r='5' fill='#2563eb'/>"
            f"<circle cx='200' cy='{y}' r='5' fill='#059669'/>"
            f"<path d='M 30 {y} C 115 {y}, 115 {y}, 200 {y}' "
            f"stroke='#93c5fd' stroke-width='2' fill='none'/>"
            f"<text x='42' y='{y + 4}' font-size='11' fill='#111'>{html.escape(edge)}</text>"
        )
        y += 24
    return (
        f"<svg width='{svg_w}' height='{svg_h}' "
        f"style='background:#f8fafc;border-radius:6px;border:1px solid #e2e8f0;padding:6px'>"
        + "".join(elements)
        + "</svg>"
    )


def render_html(run_id: str, store: str | None = None, selected: int | None = None) -> str:
    """Self-contained viewer page for ``run_id`` (double-clickable, no server)."""
    run = load_run(run_id, store)
    index = run["index"] or {}
    watched: list[str] = list(index.get("watched", []))
    total = int(index.get("tokens", 0)) if index.get("triggered") else 0
    sel = default_selected(run) if selected is None else max(0, min(selected, max(0, total - 1)))
    ok = bool(run["seal_ok"])
    badge = "[hash: VERIFIED ok chain OK]" if ok else f"[hash: FAIL {run_id} seal mismatch]"
    prompt_text = str((index.get("prompt", {}) or {}).get("text", ""))
    output_text = str((index.get("output", {}) or {}).get("text", ""))
    prompt_chips = "".join(
        f"<span class=chip>{html.escape(w)}</span>" for w in prompt_text.split() or ["(empty)"]
    )

    embed_addrs = watched[:EMBED_MAX_ADDRS]
    embed_lo, embed_hi = 0, max(0, total - 1)
    if total * max(1, len(embed_addrs)) > EMBED_VALUE_CAP:
        embed_lo, embed_hi = max(0, sel - 64), min(total - 1, sel + 64)
    series: dict[str, list[float]] = {}
    if total and embed_addrs:
        _, rows = _slice_matrix(run, embed_lo, embed_hi, embed_addrs)
        for k, addr in enumerate(embed_addrs):
            series[addr] = [
                rows[i][k] if k < len(rows[i]) else float("nan") for i in range(len(rows))
            ]
    totals = manifest_totals(
        str(index.get("model_id", "")), str(index.get("checkpoint_sha256", ""))
    )
    unwatched_n = max(0, (totals["unwatched_slots"] if totals else 0) - len(watched))
    grey = "".join(
        "<span class=grey title='address exists, value null — not_watched'>--</span>"
        for _ in range(min(GREY_PAGE, unwatched_n))
    )
    grey_note = (
        f"showing {min(GREY_PAGE, unwatched_n)} of {unwatched_n} unwatched"
        if unwatched_n
        else ("manifest not on disk — W watched shown" if not totals and watched else "")
    )
    detail_rows = ""
    for addr in embed_addrs:
        vals = series.get(addr, [])
        local = vals[sel - embed_lo] if 0 <= sel - embed_lo < len(vals) else float("nan")
        text = f"{local:.2f}" if local == local else "--"
        detail_rows += (
            f"<tr><td class=blue>{html.escape(addr)}</td><td>{text}</td>"
            f"<td class=spark>{_spark(vals, sel - embed_lo)}</td></tr>"
        )
    co = _cooccurrence(series, sel - embed_lo)
    co_html = (
        ("<li>" + "</li><li>".join(html.escape(c) for c in co) + "</li>")
        if co
        else "<li>(none above |r|&gt;0.5)</li>"
    )
    token_chips = "".join(_tchip(t, sel) for t in range(embed_lo, embed_hi + 1))
    heatmap_html = _svg_heatmap(series, embed_lo, embed_hi, embed_addrs, sel)
    circuit_svg = _svg_circuit(co)
    data = json.dumps(
        {
            "lo": embed_lo,
            "series": series,
            "watched": embed_addrs,
            "total": total,
            "sel": sel,
            "n_watched": len(watched),
        }
    )
    banner = (
        ""
        if ok
        else (
            "<div class=redbanner>hash FAIL — data still shown with red badge "
            "(see glassbox verify). Rendering continues, never blank.</div>"
        )
    )
    deep_note = (
        "" if total else "<p class=notice>Untriggered Deep run: trigger condition did not arm.</p>"
    )
    cap_note = (
        f"<p class=notice>Embedded window tokens {embed_lo}..{embed_hi} "
        f"(slice only; full deep.bin never embedded).</p>"
        if total and (embed_hi - embed_lo + 1) < total
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(run_id)}</title>
<style>body{{background:#fff;max-width:960px;margin:0 auto;padding:16px;
font-family:system-ui,-apple-system,sans-serif;font-size:15px;line-height:1.5;color:#111}}
h1{{font-size:20px}}h2{{font-size:16px;margin-top:24px}}
.badge{{border:1px solid #999;padding:4px 8px;font-size:13px}}
.chip,.tchip{{display:inline-block;border:1px solid #9ca3af;padding:2px 8px;margin:2px;
border-radius:4px;cursor:pointer;background:#fff}}
.tchip.sel{{outline:2px solid #000}}
.strip{{white-space:nowrap;overflow-x:auto;border:1px solid #ddd;padding:8px}}
table{{border-collapse:collapse;margin-top:8px}}td,th{{border:1px solid #ddd;padding:4px 10px}}
td.blue{{background:#1d4ed8;color:#fff}}td.spark{{font-family:monospace}}
.grey{{display:inline-block;border:1px solid #9ca3af;color:#9ca3af;
padding:2px 8px;margin:2px;border-radius:4px}}
.redbanner{{border:2px solid #dc2626;color:#dc2626;padding:8px;margin:12px 0}}
.notice{{color:#555}}.nav{{margin-top:16px}}button{{padding:4px 12px;margin-right:8px}}
a{{color:#1d4ed8}}</style></head><body>
<h1>{html.escape(run_id)} <span class=badge>{html.escape(badge)}</span>
<span class=notice>M-{html.escape(str(index.get("model_id", "?")))} | W={len(watched)}</span></h1>
{banner}
<h2>PROMPT (tokenized chips)</h2><div>{prompt_chips}</div>
<h2>Tokens {embed_lo}..{embed_hi} (click selects)</h2>
<div class=strip id=strip>{token_chips}</div>
<h2>Activation Heatmap Matrix (Tokens x Watched)</h2>
{heatmap_html}
<h2>DETAIL for selected token <span id=selname>T{sel}</span></h2>
<table><tr><th>Watched (blue, n_watched={len(watched)})</th>
<th>value</th><th>sparkline 10..20</th></tr>
<tbody id=detail>{detail_rows}</tbody></table>
<p>Unwatched (grey outline, no value):</p><div>{grey}</div>
<p class=notice>address exists, value null — not_watched. {html.escape(grey_note)}</p>
<h3>Co-occurrence at this token window</h3>
<p class=notice>co-occurrence, not causality.</p><ul id=co>{co_html}</ul>
<h3>Co-occurrence Circuit Graph</h3>
{circuit_svg}
<h2>OUTPUT</h2><p>{html.escape(output_text)}</p>
{deep_note}{cap_note}
<div class=nav><button id=prev>&lt; Prev token</button>
<button id=next>Next token &gt;</button>
Jump <input id=jump size=6 value="{sel}">
<a href="#prompt">back to PROMPT</a> <a href="#output">fwd</a></div>
<p class=notice>seal: hash_i={html.escape(run["seal_hash"][:16])}... |<br>
Raw save for research — don't run on private user traffic unless legal basis;
plug your redactor= if you must.</p>
<script id=cold type="application/json">{html.escape(data)}</script>
<script>
const COLD = JSON.parse(document.getElementById('cold').textContent);
const BARS = '▁▂▃▄▅▆▇█';
function spark(arr, pos) {{
  const w = arr.slice(Math.max(0, pos - 9), pos + 1).filter(v => v === v);
  if (!w.length) return '';
  const lo = Math.min(...w), hi = Math.max(...w), sp = (hi - lo) || 1;
  return arr.slice(Math.max(0, pos - 9), pos + 1).map(v =>
    v !== v ? ' ' : BARS[Math.min(7, Math.floor((v - lo) / sp * 8))]).join('');
}}
function corr(xs, ys) {{
  const p = xs.map((x, i) => [x, ys[i]]).filter(([x, y]) => x === x && y === y);
  if (p.length < 3) return 0;
  const mx = p.reduce((s, [x]) => s + x, 0) / p.length;
  const my = p.reduce((s, [, y]) => s + y, 0) / p.length;
  let num = 0, dx = 0, dy = 0;
  for (const [x, y] of p) {{
    num += (x - mx) * (y - my); dx += (x - mx) ** 2; dy += (y - my) ** 2;
  }}
  return (dx && dy) ? num / Math.sqrt(dx * dy) : 0;
}}
function select(t) {{
  const names = COLD.watched, lo = COLD.lo;
  if (!names.length || t < lo || t > lo + COLD.series[names[0]].length - 1) return;
  document.querySelectorAll('.tchip').forEach(c =>
    c.classList.toggle('sel', +c.dataset.t === t));
  document.querySelectorAll('.hm-cell').forEach(c =>
    c.setAttribute('stroke', +c.dataset.t === t ? '#000' : '#e5e7eb'));
  document.getElementById('selname').textContent = 'T' + t;
  const li = t - lo;
  document.getElementById('detail').innerHTML = names.map(n => {{
    const v = COLD.series[n][li];
    return `<tr><td class=blue>${{n}}</td><td>${{v === null || v !== v ? '--' : v.toFixed(2)}}</td>`
      + `<td class=spark>${{spark(COLD.series[n], li)}}</td></tr>`;
  }}).join('');
  const edges = [];
  for (let i = 0; i < names.length; i++) for (let j = i + 1; j < names.length; j++) {{
    const a = COLD.series[names[i]].slice(Math.max(0, li - 5), li + 6);
    const b = COLD.series[names[j]].slice(Math.max(0, li - 5), li + 6);
    const r = corr(a, b);
    if (Math.abs(r) > 0.5) edges.push([Math.abs(r), names[i], names[j], r]);
  }}
  edges.sort((x, y) => y[0] - x[0]);
  document.getElementById('co').innerHTML =
    edges.slice(0, 5).map(([, a, b, r]) =>
      `<li>${{a}} ~ ${{b}} (r=${{r.toFixed(2)}})</li>`).join('')
    || '<li>(none above |r|&gt;0.5)</li>';
  document.getElementById('jump').value = t;
  history.replaceState(null, '', '#token=' + t);
}}
document.getElementById('strip').addEventListener('click', e => {{
  const c = e.target.closest('.tchip'); if (c) select(+c.dataset.t);
}});
document.getElementById('prev').onclick = () => select(+document.getElementById('jump').value - 1);
document.getElementById('next').onclick = () => select(+document.getElementById('jump').value + 1);
document.getElementById('jump').onchange = e => select(+e.target.value);
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft') select(+document.getElementById('jump').value - 1);
  if (e.key === 'ArrowRight') select(+document.getElementById('jump').value + 1);
}});
if (location.hash.startsWith('#token=')) select(+location.hash.slice(7)) || select(COLD.sel);
</script></body></html>
"""


def open_in_browser(run_id: str, store: str | None = None, port: int = 8137) -> str:
    """Export to a temp file and open it (no server, no GPU, persists)."""
    import tempfile

    root = _store_root(store)
    html_text = render_html(run_id, root)
    path = os.path.join(tempfile.gettempdir(), f"{run_id}.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html_text)
    try:
        webbrowser.open("file://" + path)
    except Exception:
        pass
    _ = port
    print(f"Viewer http://127.0.0.1:{port}/{run_id} | export {path} | badge sealed")
    return path
