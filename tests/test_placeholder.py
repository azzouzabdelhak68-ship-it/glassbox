"""Placeholder tests — Phase 0 Loop 1 DoD (plan.md §7 Loop 3, §13)."""

from __future__ import annotations

from glassbox.cli import COMMANDS, build_parser


def test_cli_lists_eight_commands() -> None:
    """CLI must expose exactly the 8 spec §A.1-A.8 subcommands."""
    assert sorted(COMMANDS) == [
        "compare",
        "export",
        "find",
        "run",
        "scan",
        "size",
        "verify",
        "view",
    ]


def test_placeholder() -> None:
    """Minimal passing test so GLASSBOX_GPU=cpu pytest is green."""
    parser = build_parser()
    assert parser.prog == "glassbox"
    assert True


def test_backward_compatibility_alias() -> None:
    """Verify that legacy 'import blackbox' continues to work seamlessly."""
    from blackbox.tap.hooks import DeepTap

    import blackbox
    from blackbox import BlackBox, BlackBoxError

    assert BlackBox is not None
    assert issubclass(BlackBoxError, Exception)
    assert DeepTap is not None
    assert hasattr(blackbox, "watch")
