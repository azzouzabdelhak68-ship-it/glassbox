"""GlassBox GitHub Wiki Documentation Suite — Automated E2E Verification Harness.

Four-Tier Verification Architecture:
- Tier 1: Feature Coverage (Core pages, navigation sidebar, cookbooks, sync tooling)
- Tier 2: Boundary & Corner Cases (Visual diagrams, length, header hierarchies, TOC)
- Tier 3: Cross-Feature & Syntax Validation (AST syntax parsing, GlassBox symbol matching,
          CLI command & option verification, internal markdown link resolution)
- Tier 4: Real-World & Integrity Scenarios (Git repository non-intrusiveness audit,
          sync script CLI execution)
"""

from __future__ import annotations

import ast
import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path Constants & Target Inventory
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = REPO_ROOT / "wiki"
SCRIPTS_DIR = REPO_ROOT / "scripts"
SYNC_SCRIPT = SCRIPTS_DIR / "sync_wiki.py"

REQUIRED_PRIMARY_PAGES = [
    "Home.md",
    "User-Guide.md",
    "Research-Playbook.md",
    "Discovery-and-Hypothesis-Catalog.md",
    "Activation-Steering-and-Ablation-Guide.md",
    "Architecture-and-Internals.md",
]

REQUIRED_COOKBOOKS = [
    "Cookbook-Colab-and-Solo-Testing.md",
    "Cookbook-Troubleshooting-and-FAQ.md",
]

NAVIGATION_FILES = [
    "_Sidebar.md",
]

ALL_WIKI_PAGES = REQUIRED_PRIMARY_PAGES + REQUIRED_COOKBOOKS

VALID_CLI_SUBCOMMANDS = {
    "scan",
    "run",
    "find",
    "compare",
    "verify",
    "view",
    "size",
    "export",
}

# ---------------------------------------------------------------------------
# Helper Extraction & Resolution Utilities
# ---------------------------------------------------------------------------


def extract_python_blocks(content: str) -> list[str]:
    """Extract all python code blocks from markdown content."""
    pattern = r"```(?:python|py)\s*\n(.*?)\n```"
    raw_blocks = re.findall(pattern, content, flags=re.DOTALL)
    cleaned: list[str] = []
    for block in raw_blocks:
        # Handle REPL-style blocks if present
        lines = block.splitlines()
        if any(line.startswith(">>> ") for line in lines):
            repl_lines = []
            for line in lines:
                if line.startswith(">>> "):
                    repl_lines.append(line[4:])
                elif line.startswith("... "):
                    repl_lines.append(line[4:])
            cleaned.append("\n".join(repl_lines))
        else:
            cleaned.append(block)
    return cleaned


def extract_glassbox_imports(code: str) -> list[tuple[str, str]]:
    """Extract (module, symbol_name) from python code block."""
    imports: list[tuple[str, str]] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "glassbox" or mod.startswith("glassbox."):
                for alias in node.names:
                    imports.append((mod, alias.name))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "glassbox" or alias.name.startswith("glassbox."):
                    imports.append((alias.name, alias.name))
    return imports


def get_registered_cli_flags() -> dict[str, set[str]]:
    """Dynamically extract registered options and flags from glassbox.cli."""
    from glassbox.cli import build_parser

    parser = build_parser()
    mapping: dict[str, set[str]] = {}
    for action in parser._actions:
        choices = getattr(action, "choices", None) or {}
        if isinstance(choices, dict):
            for cmd_name, subparser in choices.items():
                flags: set[str] = set()
                for sub_action in getattr(subparser, "_actions", []):
                    for opt in getattr(sub_action, "option_strings", []):
                        flags.add(opt)
                flags.add("-h")
                flags.add("--help")
                mapping[cmd_name] = flags
    return mapping


def extract_cli_invocations(content: str) -> list[tuple[str, list[str]]]:
    """Extract CLI invocations of glassbox and their flags from markdown content."""
    invocations: list[tuple[str, list[str]]] = []
    # Match lines with glassbox <cmd> or python -m glassbox <cmd> (avoiding import statements)
    pattern = (
        r"(?<!from\s)(?<!import\s)(?:\bglassbox|python[^\S\r\n]+-m[^\S\r\n]+glassbox(?:\.cli)?)"
        r"[^\S\r\n]+([a-z0-9_-]+)(.*?)(?:\r?\n|$|`)"
    )
    for match in re.finditer(pattern, content):
        cmd = match.group(1).strip()
        rest = match.group(2)
        # Extract flags starting with -- or -
        flags = re.findall(r"(?:^|\s)(--[a-z0-9_-]+|-[a-zA-Z])", rest)
        invocations.append((cmd, flags))
    return invocations


def has_visual_diagram(content: str) -> bool:
    """Check if content includes Mermaid diagrams or ASCII/Unicode box-drawing flowcharts."""
    if "```mermaid" in content:
        return True

    # High-clarity Unicode box drawing characters
    unicode_box_chars = set("┌┐└┘├┤┬┴┼─│═║╔╗╚╝▲▼►◄◆◇◈")
    unicode_count = sum(content.count(char) for char in unicode_box_chars)
    if unicode_count >= 6:
        return True

    # ASCII flowchart patterns (+---+ or +===+ or |   | or -->)
    ascii_box_patterns = [
        r"\+[-=]{3,}\+",
        r"\|[\sA-Za-z0-9_-]{3,}\|",
        r"-->|==>|<--|<==",
        r"\[[-A-Za-z0-9_\s]+\]\s*(?:-->|->)\s*\[",
    ]
    matches = sum(len(re.findall(p, content)) for p in ascii_box_patterns)
    return matches >= 3


def extract_internal_links(content: str) -> list[str]:
    """Extract internal markdown link targets (excluding external URLs)."""
    # Regex for standard markdown links [text](target)
    pattern = r"\[(?:[^\]]+)\]\(([^\"'\)]+)(?:\s+\"[^\"]*\")?\)"
    raw_targets = re.findall(pattern, content)
    internal: list[str] = []
    for target in raw_targets:
        target = target.strip()
        if target.startswith(("http://", "https://", "mailto:", "ftp://")):
            continue
        # Skip pure inline anchors or code-like patterns
        if target:
            internal.append(target)
    return internal


def normalize_anchor(text: str) -> str:
    """Convert header text to GitHub-compatible anchor slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text


def extract_headers_as_anchors(content: str) -> set[str]:
    """Extract markdown headers and generate normalized anchor slugs."""
    anchors: set[str] = set()
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("#"):
            header_text = line.lstrip("#").strip()
            anchors.add(normalize_anchor(header_text))
            # Also allow direct raw alphanumeric slug
            anchors.add(re.sub(r"\W+", "-", header_text.lower()).strip("-"))
    return anchors


def resolve_wiki_link(
    wiki_dir: Path, source_file: Path, target: str
) -> tuple[bool, str]:
    """Resolve an internal wiki link against the filesystem and headers."""
    if target.startswith("#"):
        # Anchor within the same file
        anchor = target[1:].lower()
        anchors = extract_headers_as_anchors(source_file.read_text(encoding="utf-8"))
        if anchor in anchors or any(anchor in a or a in anchor for a in anchors):
            return True, "anchor resolved"
        return False, f"Anchor '{target}' not found in {source_file.name}"

    file_part, _, anchor_part = target.partition("#")
    # GitHub Wiki supports slugs without .md or with .md
    candidate_names = [file_part, f"{file_part}.md"]
    found_file: Path | None = None
    for cand in candidate_names:
        cand_path = wiki_dir / cand
        if cand_path.is_file():
            found_file = cand_path
            break

    if not found_file:
        return False, f"Target page '{file_part}' does not exist in {wiki_dir}"

    if anchor_part:
        anchors = extract_headers_as_anchors(found_file.read_text(encoding="utf-8"))
        norm_anchor = anchor_part.lower()
        if norm_anchor in anchors or any(norm_anchor in a or a in norm_anchor for a in anchors):
            return True, "anchor resolved in target"
        return False, f"Anchor '#{anchor_part}' not found in {found_file.name}"

    return True, "page resolved"


# ---------------------------------------------------------------------------
# Tier 1: Feature Coverage & Asset Inventory
# ---------------------------------------------------------------------------


class TestTier1FeatureCoverage:
    """Validate existence, non-emptiness, and accessibility of all documentation assets."""

    @pytest.mark.parametrize("page_name", REQUIRED_PRIMARY_PAGES)
    def test_primary_wiki_pages_exist(self, page_name: str) -> None:
        """Verify each of the 6 primary wiki pages exists and is non-empty."""
        page_path = WIKI_DIR / page_name
        assert page_path.is_file(), f"Primary wiki page missing: {page_path}"
        assert page_path.stat().st_size > 100, f"Primary wiki page is empty: {page_path}"

    @pytest.mark.parametrize("cookbook_name", REQUIRED_COOKBOOKS)
    def test_supplemental_cookbooks_exist(self, cookbook_name: str) -> None:
        """Verify supplemental cookbooks for solo experimenters exist and are non-empty."""
        cookbook_path = WIKI_DIR / cookbook_name
        assert cookbook_path.is_file(), f"Cookbook page missing: {cookbook_path}"
        assert cookbook_path.stat().st_size > 100, f"Cookbook page is empty: {cookbook_path}"

    def test_sidebar_navigation_exists(self) -> None:
        """Verify GitHub Wiki persistent sidebar navigation file exists."""
        sidebar_path = WIKI_DIR / "_Sidebar.md"
        assert sidebar_path.is_file(), f"Sidebar navigation missing: {sidebar_path}"
        assert sidebar_path.stat().st_size > 50, f"Sidebar file is empty: {sidebar_path}"

    def test_sync_tooling_script_exists(self) -> None:
        """Verify wiki synchronization tooling script exists in scripts/."""
        assert SYNC_SCRIPT.is_file(), f"Sync script missing: {SYNC_SCRIPT}"
        assert SYNC_SCRIPT.stat().st_size > 100, f"Sync script is empty: {SYNC_SCRIPT}"


# ---------------------------------------------------------------------------
# Tier 2: Boundary & Corner Cases
# ---------------------------------------------------------------------------


class TestTier2BoundaryAndCornerCases:
    """Validate structural boundaries, visual diagram mandates, lengths, and hierarchies."""

    @pytest.mark.parametrize("page_name", ALL_WIKI_PAGES)
    def test_visual_diagrams_present_on_all_pages(self, page_name: str) -> None:
        """Requirement R3: Verify EVERY page has visual diagrams (Mermaid or ASCII/Unicode)."""
        page_path = WIKI_DIR / page_name
        assert page_path.is_file(), f"Wiki page missing: {page_name}"
        content = page_path.read_text(encoding="utf-8")
        assert has_visual_diagram(content), (
            f"Page '{page_name}' lacks required visual diagram "
            f"(must include Mermaid or high-clarity Unicode/ASCII box-drawing scheme)."
        )

    @pytest.mark.parametrize("page_name", REQUIRED_PRIMARY_PAGES)
    def test_primary_pages_minimum_content_length(self, page_name: str) -> None:
        """Boundary analysis: Primary research guides must contain substantive technical content."""
        page_path = WIKI_DIR / page_name
        assert page_path.is_file(), f"Wiki page missing: {page_name}"
        content = page_path.read_text(encoding="utf-8")
        # Primary guides must have at least 2,000 characters
        assert len(content) >= 2000, (
            f"Page '{page_name}' is too brief ({len(content)} chars). "
            f"Expected substantive technical depth (>= 2000 chars)."
        )

    @pytest.mark.parametrize("page_name", ALL_WIKI_PAGES)
    def test_proper_header_hierarchy(self, page_name: str) -> None:
        """Boundary analysis: Heading hierarchy must be well-formed without invalid skips."""
        page_path = WIKI_DIR / page_name
        assert page_path.is_file(), f"Wiki page missing: {page_name}"
        lines = page_path.read_text(encoding="utf-8").splitlines()

        h1_count = 0
        prev_level = 0
        hierarchy_errors: list[str] = []

        for lineno, line in enumerate(lines, 1):
            sline = line.strip()
            if sline.startswith("#") and not sline.startswith("#!"):
                m = re.match(r"^(#+)\s", sline)
                if m:
                    level = len(m.group(1))
                    if level == 1:
                        h1_count += 1
                    # Heading jump boundary: level cannot jump more than +1 (e.g. # to ###)
                    if prev_level > 0 and level > prev_level + 1:
                        hierarchy_errors.append(
                            f"Line {lineno}: header jumped from H{prev_level} to H{level}"
                        )
                    prev_level = level

        assert h1_count >= 1, f"Page '{page_name}' must have at least one H1 header (# Title)."
        assert not hierarchy_errors, (
            f"Page '{page_name}' contains header hierarchy violations:\n"
            + "\n".join(hierarchy_errors)
        )

    def test_home_page_table_of_contents_links(self) -> None:
        """Verify Home.md contains a complete Table of Contents linking to all wiki pages."""
        home_path = WIKI_DIR / "Home.md"
        assert home_path.is_file(), "Home.md missing"
        content = home_path.read_text(encoding="utf-8")

        for required_page in REQUIRED_PRIMARY_PAGES[1:] + REQUIRED_COOKBOOKS:
            slug = required_page[:-3]
            pattern = (
                rf"\[[^\]]+\]\((?:{re.escape(slug)}|{re.escape(required_page)})(?:#[^\)]*)?\)"
            )
            assert re.search(pattern, content), (
                f"Home.md Table of Contents missing navigation link to '{required_page}'."
            )

    def test_sidebar_navigation_covers_all_pages(self) -> None:
        """Verify _Sidebar.md contains navigation links to all primary pages and cookbooks."""
        sidebar_path = WIKI_DIR / "_Sidebar.md"
        assert sidebar_path.is_file(), "_Sidebar.md missing"
        content = sidebar_path.read_text(encoding="utf-8")

        for page in ALL_WIKI_PAGES:
            slug = page[:-3]
            pattern = rf"\[[^\]]+\]\((?:{re.escape(slug)}|{re.escape(page)})(?:#[^\)]*)?\)"
            assert re.search(pattern, content), (
                f"_Sidebar.md missing persistent navigation link to '{page}'."
            )


# ---------------------------------------------------------------------------
# Tier 3: Cross-Feature & Syntax Validation
# ---------------------------------------------------------------------------


class TestTier3CrossFeatureAndSyntaxValidation:
    """Validate AST syntax, GlassBox symbols, CLI commands & options, and internal links."""

    def test_python_code_blocks_ast_syntax(self) -> None:
        """Extract all Python code blocks across all wiki files and verify ast.parse compilation."""
        assert WIKI_DIR.is_dir(), f"Wiki directory {WIKI_DIR} does not exist"
        md_files = sorted(WIKI_DIR.glob("*.md"))
        assert md_files, f"No markdown files found in {WIKI_DIR}"

        syntax_errors: list[str] = []
        total_blocks = 0

        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8")
            blocks = extract_python_blocks(content)
            for idx, block in enumerate(blocks, 1):
                total_blocks += 1
                try:
                    ast.parse(block)
                except SyntaxError as exc:
                    syntax_errors.append(
                        f"[{md_file.name}] Code block #{idx} (line {exc.lineno}): {exc.msg}\n"
                        f"Code:\n{block}"
                    )

        assert not syntax_errors, (
            f"Encountered {len(syntax_errors)} syntax error(s) across {total_blocks} code blocks:\n"
            + "\n"
            + "=" * 60
            + "\n"
            + "\n---\n".join(syntax_errors)
        )
        assert total_blocks > 0, "No Python code blocks discovered across wiki documentation."

    def test_imported_glassbox_symbols_validity(self) -> None:
        """Verify all imported GlassBox symbols exist in the repository implementation."""
        assert WIKI_DIR.is_dir(), f"Wiki directory {WIKI_DIR} does not exist"
        md_files = sorted(WIKI_DIR.glob("*.md"))

        invalid_symbols: list[str] = []
        verified_count = 0

        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8")
            blocks = extract_python_blocks(content)
            for block in blocks:
                imports = extract_glassbox_imports(block)
                for mod_name, sym_name in imports:
                    verified_count += 1
                    try:
                        mod = importlib.import_module(mod_name)
                    except ImportError:
                        invalid_symbols.append(
                            f"[{md_file.name}] Cannot import module '{mod_name}'"
                        )
                        continue

                    # If importing a specific symbol (e.g. from glassbox.steering import steer)
                    if sym_name != mod_name:
                        if not hasattr(mod, sym_name):
                            invalid_symbols.append(
                                f"[{md_file.name}] Symbol '{sym_name}' not in module '{mod_name}'"
                            )

        assert not invalid_symbols, (
            "Discovered invalid or nonexistent GlassBox symbols:\n"
            + "\n".join(invalid_symbols)
        )
        assert verified_count > 0, "Expected at least one GlassBox symbol import in documentation."

    def test_cli_commands_and_options_validity(self) -> None:
        """Verify all CLI invocations match registered subcommands and options in glassbox.cli."""
        assert WIKI_DIR.is_dir(), f"Wiki directory {WIKI_DIR} does not exist"
        valid_options_by_cmd = get_registered_cli_flags()
        md_files = sorted(WIKI_DIR.glob("*.md"))

        cli_errors: list[str] = []
        total_invocations = 0

        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8")
            invocations = extract_cli_invocations(content)
            for cmd, flags in invocations:
                total_invocations += 1
                if cmd not in VALID_CLI_SUBCOMMANDS:
                    cli_errors.append(
                        f"[{md_file.name}] Unknown CLI subcommand 'glassbox {cmd}' "
                        f"(valid: {sorted(VALID_CLI_SUBCOMMANDS)})"
                    )
                    continue

                allowed_flags = valid_options_by_cmd.get(cmd, set())
                for flag in flags:
                    if flag not in allowed_flags:
                        cli_errors.append(
                            f"[{md_file.name}] Invalid flag '{flag}' for 'glassbox {cmd}' "
                            f"(allowed: {sorted(allowed_flags)})"
                        )

        assert not cli_errors, (
            f"CLI verification detected {len(cli_errors)} invalid invocations:\n"
            + "\n".join(cli_errors)
        )
        assert total_invocations > 0, "Expected CLI examples in documentation suite."

    def test_internal_markdown_links_resolution(self) -> None:
        """Verify internal markdown links and cross-references resolve to existing targets."""
        assert WIKI_DIR.is_dir(), f"Wiki directory {WIKI_DIR} does not exist"
        md_files = sorted(WIKI_DIR.glob("*.md"))

        link_errors: list[str] = []
        total_links = 0

        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8")
            targets = extract_internal_links(content)
            for target in targets:
                total_links += 1
                ok, reason = resolve_wiki_link(WIKI_DIR, md_file, target)
                if not ok:
                    link_errors.append(f"[{md_file.name}] Link target '{target}' broken: {reason}")

        assert not link_errors, (
            f"Discovered {len(link_errors)} broken internal link(s) across {total_links} links:\n"
            + "\n".join(link_errors)
        )
        assert total_links > 0, "Expected internal markdown cross-references across wiki pages."


# ---------------------------------------------------------------------------
# Tier 4: Real-World & Integrity Scenarios
# ---------------------------------------------------------------------------


class TestTier4RealWorldAndIntegrityScenarios:
    """Validate repository non-intrusiveness and sync tooling execution."""

    def test_git_status_non_intrusiveness(self) -> None:
        """Enforce non-intrusiveness: zero repository source code modified outside documentation."""
        cmd = ["git", "status", "--porcelain"]
        res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        changed_lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]

        # Allowed paths for modification / creation
        allowed_prefixes = (
            "wiki/",
            "scripts/sync_wiki.py",
            "tests/test_wiki_suite.py",
            "TEST_INFRA.md",
            "TEST_READY.md",
            "PROJECT.md",
            ".agents/",
            ".pytest_cache/",
            ".gitignore",
        )

        intrusive_modifications: list[str] = []
        for line in changed_lines:
            # Porcelain format: XY PATH (or XY PATH1 -> PATH2)
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                path = parts[1].replace("\\", "/")
                if not any(path.startswith(prefix) for prefix in allowed_prefixes):
                    intrusive_modifications.append(f"Intrusive file touched: {line}")

        assert not intrusive_modifications, (
            "Repository non-intrusiveness violated! Source files outside docs were modified:\n"
            + "\n".join(intrusive_modifications)
        )

    def test_sync_wiki_script_configuration_and_help(self) -> None:
        """Verify scripts/sync_wiki.py remote configuration, branch, AST, and --help execution."""
        assert SYNC_SCRIPT.is_file(), f"Sync script missing: {SYNC_SCRIPT}"

        code = SYNC_SCRIPT.read_text(encoding="utf-8")
        # Validate AST compilation
        ast.parse(code)

        # Verify target repository remote URL and branch
        assert "https://github.com/azzouzabdelhak68-ship-it/glassbox.wiki.git" in code, (
            "scripts/sync_wiki.py must configure the remote URL: "
            "'https://github.com/azzouzabdelhak68-ship-it/glassbox.wiki.git'"
        )
        assert "master" in code, "scripts/sync_wiki.py must target the 'master' branch."

        # Execute --help CLI invocation to verify CLI ergonomics
        res = subprocess.run(
            [sys.executable, str(SYNC_SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, (
            f"scripts/sync_wiki.py --help failed with exit code {res.returncode}:\n{res.stderr}"
        )
        assert "wiki" in res.stdout.lower(), "Help output must reference wiki synchronization."
