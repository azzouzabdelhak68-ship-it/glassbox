"""The Eureka Experiment (v3.1) — Mechanistic Discovery & Causal Control of Self-Doubt.

Modules:
- `analyzer_auditor`: Cell 3A Independent LLM semantic auditor & JSON parser.
- `cdla_svd_engine`: Cell 3B Contrastive DLA tensor math & SVD extraction.
- `phase1_runner`: Phase 1 dense flight recording across 25 prompt pairs.
- `phase2_knockout`: Phase 2 causal necessity validation via Mean-Ablation.
- `phase3_trigger`: Phase 3 causal sufficiency validation via Pulsed Steering.
"""

from .analyzer_auditor import (
    ANALYZER_SYSTEM_PROMPT,
    audit_cohort,
    audit_run,
    build_analyzer_prompt,
    format_token_trace,
    heuristic_audit,
    parse_analyzer_output,
)
from .cdla_svd_engine import (
    CDLASvdEngine,
    benjamini_hochberg,
    compute_paired_effect_sizes,
    extract_svd_pc1,
)
from .phase1_runner import (
    format_chat_prompt,
    load_prompt_battery,
    run_phase1_cohort,
    save_phase1_records,
)
from .phase2_knockout import (
    build_ablation_targets,
    grade_completion,
    run_phase2_eval,
)
from .phase3_trigger import (
    detect_hesitation,
    generate_orthogonal_vector,
    is_coherent,
    run_phase3_eval,
)

__all__ = [
    "ANALYZER_SYSTEM_PROMPT",
    "CDLASvdEngine",
    "audit_cohort",
    "audit_run",
    "benjamini_hochberg",
    "build_ablation_targets",
    "build_analyzer_prompt",
    "compute_paired_effect_sizes",
    "detect_hesitation",
    "extract_svd_pc1",
    "format_chat_prompt",
    "format_token_trace",
    "generate_orthogonal_vector",
    "grade_completion",
    "heuristic_audit",
    "is_coherent",
    "load_prompt_battery",
    "parse_analyzer_output",
    "run_phase1_cohort",
    "run_phase2_eval",
    "run_phase3_eval",
    "save_phase1_records",
]
