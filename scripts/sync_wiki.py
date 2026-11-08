#!/usr/bin/env python3
"""GlassBox GitHub Wiki Synchronization Tool.

Synchronizes the local `wiki/` documentation directory with the GitHub Wiki
remote repository (`https://github.com/azzouzabdelhak68-ship-it/glassbox.wiki.git`)
on branch `master`.

Supported workflows:
  python scripts/sync_wiki.py --check     # Verify local files & remote connectivity
  python scripts/sync_wiki.py --dry-run   # Simulate clone, stage, and diff without push
  python scripts/sync_wiki.py --push      # Stage, commit, and push updates to remote

Part of Milestone M1 (Wiki Infrastructure & Tooling).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_REMOTE = "https://github.com/azzouzabdelhak68-ship-it/glassbox.wiki.git"
DEFAULT_BRANCH = "master"
DEFAULT_MESSAGE = "Update GlassBox GitHub Wiki documentation suite"
DEFAULT_WIKI_DIR = "wiki"


def run_cmd(
    cmd: list[str],
    cwd: Path | str | None = None,
    capture_output: bool = True,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    """Execute a subprocess command returning completed process or raising CalledProcessError."""
    cwd_str = str(cwd) if cwd else None
    return subprocess.run(
        cmd,
        cwd=cwd_str,
        text=True,
        capture_output=capture_output,
        check=True,
        timeout=timeout,
    )


def check_git_installed() -> tuple[bool, str]:
    """Check if git is installed and accessible on PATH."""
    try:
        res = run_cmd(["git", "--version"], timeout=10)
        return True, res.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as err:
        return False, str(err)


def check_wiki_directory(wiki_dir: Path) -> tuple[bool, list[Path], str]:
    """Verify that local wiki/ directory exists and list markdown files."""
    if not wiki_dir.exists():
        return False, [], f"Directory '{wiki_dir}' does not exist."
    if not wiki_dir.is_dir():
        return False, [], f"Path '{wiki_dir}' is not a directory."

    md_files = sorted(wiki_dir.glob("*.md"))
    if not md_files:
        return False, [], f"No markdown (*.md) files found in '{wiki_dir}'."

    home_file = wiki_dir / "Home.md"
    if not home_file.exists():
        found_names = [f.name for f in md_files]
        return (
            False,
            md_files,
            f"Required entrypoint 'Home.md' missing from '{wiki_dir}'. Found: {found_names}",
        )

    return True, md_files, f"Found {len(md_files)} markdown files."


def check_remote_connectivity(remote_url: str, branch: str) -> tuple[bool, str]:
    """Verify connectivity and branch presence on the remote git repository."""
    try:
        res = run_cmd(["git", "ls-remote", remote_url, f"refs/heads/{branch}"], timeout=20)
        output = res.stdout.strip()
        if not output:
            # Check HEAD as fallback if branch is default
            res_head = run_cmd(["git", "ls-remote", remote_url, "HEAD"], timeout=20)
            output_head = res_head.stdout.strip()
            if output_head:
                commit_hash = output_head.split()[0]
                return True, f"Connected to remote (HEAD commit: {commit_hash[:8]})"
            return False, f"Remote reached, but branch 'refs/heads/{branch}' not found."
        commit_hash = output.split()[0]
        return True, f"Connected to remote (branch '{branch}' at {commit_hash[:8]})"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as err:
        stderr_msg = getattr(err, "stderr", str(err)).strip()
        return False, f"Failed to connect to remote: {stderr_msg or str(err)}"


def print_fallback_instructions(
    wiki_dir: Path,
    remote_url: str = DEFAULT_REMOTE,
    branch: str = DEFAULT_BRANCH,
    message: str = DEFAULT_MESSAGE,
    error_detail: str | None = None,
) -> None:
    """Print actionable manual Git CLI fallback instructions."""
    print("\n" + "=" * 80)
    print("MANUAL FALLBACK GIT SYNCHRONIZATION COMMANDS")
    print("=" * 80)
    if error_detail:
        print(f"Diagnostic details: {error_detail}\n")
    print("If automated sync fails due to credentials, proxy, or network restrictions,")
    print("you can synchronize the wiki manually using the following standard commands:")
    print("\n--- Linux / macOS (Bash) ---")
    print("STAGING_DIR=$(mktemp -d)")
    print(f"git clone {remote_url} \"$STAGING_DIR\"")
    print(f"cp -r {wiki_dir}/* \"$STAGING_DIR/\"")
    print("cd \"$STAGING_DIR\"")
    print(f"git checkout -B {branch}")
    print("git add -A")
    print("git status")
    print(f'git commit -m "{message}"')
    print(f"git push origin {branch}")
    print("rm -rf \"$STAGING_DIR\"")

    print("\n--- Windows (PowerShell) ---")
    print('$StagingDir = Join-Path $env:TEMP ("glassbox_wiki_" + [guid]::NewGuid().ToString("N"))')
    print(f'git clone {remote_url} "$StagingDir"')
    print(f'Copy-Item -Path "{wiki_dir}\\*" -Destination "$StagingDir" -Recurse -Force')
    print('Set-Location "$StagingDir"')
    print(f'git checkout -B {branch}')
    print('git add -A')
    print('git status')
    print(f'git commit -m "{message}"')
    print(f'git push origin {branch}')
    print(f'Set-Location -Path "{Path.cwd()}"')
    print('Remove-Item -Recurse -Force "$StagingDir"')
    print("=" * 80 + "\n")


def execute_check(wiki_dir: Path, remote_url: str, branch: str) -> int:
    """Run comprehensive diagnostics on local setup and remote connectivity."""
    print("=" * 70)
    print("GlassBox Wiki Diagnostics & Environment Check")
    print("=" * 70)

    all_passed = True

    # 1. Check Git
    git_ok, git_info = check_git_installed()
    if git_ok:
        print(f"[PASS] Git CLI: {git_info}")
    else:
        print(f"[FAIL] Git CLI: {git_info}")
        all_passed = False

    # 2. Check local wiki directory
    wiki_ok, md_files, wiki_info = check_wiki_directory(wiki_dir)
    if wiki_ok:
        print(f"[PASS] Local Wiki: {wiki_info}")
        for f in md_files:
            size_kb = f.stat().st_size / 1024.0
            print(f"       * {f.name:<40} ({size_kb:6.1f} KB)")
    else:
        print(f"[FAIL] Local Wiki: {wiki_info}")
        all_passed = False

    # 3. Check Remote Connectivity
    if git_ok:
        remote_ok, remote_info = check_remote_connectivity(remote_url, branch)
        if remote_ok:
            print(f"[PASS] Remote Repository: {remote_info}")
        else:
            print(f"[WARN] Remote Repository: {remote_info}")
            print("       (Remote push may require authentication or active network connection)")
    else:
        print("[SKIP] Remote Repository: Skipped because git is not installed.")

    print("-" * 70)
    if all_passed:
        print("Status: All local prerequisite checks PASSED.")
        print("Ready for dry-run simulation or remote push.")
        return 0
    else:
        print("Status: Prerequisite checks FAILED. Review errors above.")
        return 1


def execute_sync(
    wiki_dir: Path,
    remote_url: str,
    branch: str,
    message: str,
    dry_run: bool,
    push: bool,
    verbose: bool = False,
) -> int:
    """Execute wiki synchronization via temporary clone sandbox."""
    # Pre-flight checks
    git_ok, git_info = check_git_installed()
    if not git_ok:
        print(f"[ERROR] Git is not available: {git_info}", file=sys.stderr)
        return 1

    wiki_ok, md_files, wiki_info = check_wiki_directory(wiki_dir)
    if not wiki_ok:
        print(f"[ERROR] Local wiki validation failed: {wiki_info}", file=sys.stderr)
        return 1

    mode_label = "DRY-RUN SIMULATION" if dry_run else ("PUSH TO REMOTE" if push else "PREVIEW")
    print("=" * 70)
    print(f"GlassBox Wiki Sync: {mode_label}")
    print(f"Local Directory:   {wiki_dir.resolve()}")
    print(f"Remote URL:        {remote_url}")
    print(f"Target Branch:     {branch}")
    print("=" * 70)

    # Create temporary sandbox
    staging_dir = Path(tempfile.mkdtemp(prefix="glassbox_wiki_sync_"))
    try:
        print("\n[1/4] Cloning remote repository to temporary sandbox...")
        try:
            run_cmd(
                ["git", "clone", "--depth", "1", "-b", branch, remote_url, str(staging_dir)],
                timeout=45,
            )
        except subprocess.CalledProcessError:
            # Fallback: clone without branch specification if fresh
            try:
                run_cmd(["git", "clone", remote_url, str(staging_dir)], timeout=45)
                run_cmd(["git", "checkout", "-B", branch], cwd=staging_dir)
            except subprocess.CalledProcessError as fallback_err:
                stderr_str = getattr(fallback_err, "stderr", str(fallback_err))
                print(
                    f"[ERROR] Failed to clone remote wiki repository:\n{stderr_str}",
                    file=sys.stderr,
                )
                print_fallback_instructions(wiki_dir, remote_url, branch, message, stderr_str)
                return 1

        print("[2/4] Mirroring local wiki files into sandbox...")
        copied_count = 0
        for src_file in wiki_dir.iterdir():
            if src_file.is_file() and not src_file.name.startswith("."):
                dest_file = staging_dir / src_file.name
                shutil.copy2(src_file, dest_file)
                copied_count += 1
                if verbose:
                    print(f"      Copied: {src_file.name}")
        print(f"      Synchronized {copied_count} files.")

        print("[3/4] Evaluating git status and diff...")
        run_cmd(["git", "add", "-A"], cwd=staging_dir)
        status_res = run_cmd(["git", "status", "--porcelain"], cwd=staging_dir)
        status_lines = [line.strip() for line in status_res.stdout.splitlines() if line.strip()]

        if not status_lines:
            print("\n[OK] Wiki documentation is already up to date with remote repository.")
            print("     No changes detected. Nothing to commit or push.")
            return 0

        print(f"\nDetected {len(status_lines)} changed file(s):")
        for line in status_lines:
            print(f"   {line}")

        diff_res = run_cmd(["git", "diff", "--cached", "--stat"], cwd=staging_dir)
        if diff_res.stdout.strip():
            print("\nSummary of staged modifications:")
            print(diff_res.stdout.strip())

        # Branching on dry-run vs push
        if dry_run or not push:
            print("\n" + "-" * 70)
            print("[DRY RUN] Simulation complete. Changes verified successfully.")
            print("No commits or network pushes were performed.")
            print("To commit and push these changes to GitHub Wiki, execute:")
            print("    python scripts/sync_wiki.py --push")
            print("-" * 70)
            return 0

        # Execute push
        print(f"\n[4/4] Committing and pushing changes to {remote_url} on branch '{branch}'...")
        # Ensure committer identity is configured in temporary clone
        try:
            p_name = run_cmd(["git", "config", "user.name"], cwd=REPO_ROOT).stdout.strip()
            p_email = run_cmd(["git", "config", "user.email"], cwd=REPO_ROOT).stdout.strip()
        except Exception:
            p_name, p_email = "", ""
        run_cmd(["git", "config", "user.name", p_name or "AI Black Box contributors"], cwd=staging_dir)
        run_cmd(["git", "config", "user.email", p_email or "blackbox@example.invalid"], cwd=staging_dir)

        try:
            run_cmd(["git", "commit", "-m", message], cwd=staging_dir)
        except subprocess.CalledProcessError as commit_err:
            stderr_str = getattr(commit_err, "stderr", str(commit_err))
            print(f"\n[ERROR] Failed to commit changes in staging sandbox:\n{stderr_str}", file=sys.stderr)
            return 1
        try:
            push_res = run_cmd(["git", "push", "origin", branch], cwd=staging_dir, timeout=60)
            if verbose and push_res.stdout:
                print(push_res.stdout.strip())
            print("\n[SUCCESS] Successfully pushed wiki documentation to GitHub remote!")
            print("URL: https://github.com/azzouzabdelhak68-ship-it/glassbox/wiki")
            return 0
        except subprocess.CalledProcessError as push_err:
            stderr_str = getattr(push_err, "stderr", str(push_err))
            print(f"\n[ERROR] Failed to push to remote:\n{stderr_str}", file=sys.stderr)
            print_fallback_instructions(wiki_dir, remote_url, branch, message, stderr_str)
            return 1

    finally:
        # Clean up sandbox
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def main() -> int:
    """CLI entrypoint for wiki synchronization script."""
    parser = argparse.ArgumentParser(
        description="Synchronize GlassBox wiki/ documentation with GitHub Wiki remote repository.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/sync_wiki.py --check
  python scripts/sync_wiki.py --dry-run
  python scripts/sync_wiki.py --push -m "Update mechanistic architecture guide"
        """,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run environment diagnostics on git, local files, and remote connectivity.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate clone, file sync, staging, and diff without committing or pushing.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Stage, commit, and push wiki documentation changes to the remote repository.",
    )
    parser.add_argument(
        "--message",
        "-m",
        default=DEFAULT_MESSAGE,
        help=f"Commit message (default: '{DEFAULT_MESSAGE}').",
    )
    parser.add_argument(
        "--wiki-dir",
        type=Path,
        default=Path(DEFAULT_WIKI_DIR),
        help=f"Path to local wiki directory (default: '{DEFAULT_WIKI_DIR}').",
    )
    parser.add_argument(
        "--remote",
        default=DEFAULT_REMOTE,
        help=f"Target Git remote URL (default: '{DEFAULT_REMOTE}').",
    )
    parser.add_argument(
        "--branch",
        default=DEFAULT_BRANCH,
        help=f"Target Git branch (default: '{DEFAULT_BRANCH}').",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed diagnostic and git output.",
    )

    args = parser.parse_args()

    # Route based on provided flags
    if args.check:
        return execute_check(args.wiki_dir, args.remote, args.branch)

    if args.dry_run or args.push:
        return execute_sync(
            wiki_dir=args.wiki_dir,
            remote_url=args.remote,
            branch=args.branch,
            message=args.message,
            dry_run=args.dry_run,
            push=args.push,
            verbose=args.verbose,
        )

    # Default action if neither --check, --dry-run, nor --push was specified:
    # Run diagnostic check and inform user
    print("No action specified (--check, --dry-run, or --push). Running status check:\n")
    check_code = execute_check(args.wiki_dir, args.remote, args.branch)
    print("\nNext steps:")
    print("  * To test the synchronization diff: python scripts/sync_wiki.py --dry-run")
    print("  * To push updates to GitHub Wiki:   python scripts/sync_wiki.py --push")
    return check_code


if __name__ == "__main__":
    sys.exit(main())
