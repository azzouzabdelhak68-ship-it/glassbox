# FORMAT_GUIDE — we document, you edit

> **Rule: we document, you edit — `glassbox export --zip` gives plain files. We won't build your schema.**

## What export gives you

```bash
glassbox export --zip --run RUN-000184 --with-manifest --out ./RUN-000184.zip
# flat zip, no proprietary DB:
#   deep.bin          (COLD, optional: float LE row-major [T, W], token-major, chunked sparse supported)
#   deep_index.json   (HOT unified manifest: addresses, context, prompt, output, replay)
#   seal.json         (HOT: hash_i chain link)
#   manifest.json     (only with --with-manifest: inventory of addresses)
```

## How to adapt to your format

1. Unzip: `unzip RUN-000184.zip -d ./run184/`.
2. Map per-model layout via `adapters/format/*.json`:
   - `adapters/format/llama3.json` (Llama-3 family: row_major_float16).
   - `adapters/format/mistral.json` (Mistral family).
3. Convert: `deep.bin` decoder reads `[T, W]` rows; frame via `glassbox.to_dataframe(run)` for float32 logical values.
4. Validate: every example JSON must parse (`python -m json.tool`); re-seal is yours after edit (edited runs fail `glassbox verify` by design).

We provide the hook point (`FORMAT_GUIDE.md` + `adapters/format/<family>.json`), not per-user builds.
