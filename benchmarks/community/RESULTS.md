# Community benchmarks — the report

> `benchmark.sh` + estimator are the method; this table IS the report. One dir per run: `benchmarks/community/<date>-<gpu>-<model>-W<w>/` with `result.json` + `env.txt`.

Spec keys: `mode, W, T, R, precision, overhead_pct, ms_per_token_baseline, ms_per_token_tapped, deep_bin_bytes_per_run, total_GB, flag, torch, cuda, gpu, os`.

| mode | W | T | R | precision | overhead_pct | ms_baseline | ms_tapped | GB | flag | torch | cuda | gpu | os |
|------|---|---|---|-----------|--------------|-------------|-----------|----|------|-------|------|-----|----|
<!-- ms_baseline = ms_per_token_baseline; ms_tapped = ms_per_token_tapped; GB = total_GB (W*T*R*B*1.15); flag = GREEN (<50GB) / YELLOW (50-1000GB) / RED (>1000GB) -->
| deep | 1002 | 500 | 2 | fp16 | n/a | n/a | 0.4335 | 0.002305 | GREEN | 2.12.1 | cpu | none | Windows 10 |
