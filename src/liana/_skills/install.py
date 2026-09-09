"""Install the bundled liana Agent Skill for Claude Code.

Exposes the ``liana-install-skills`` console script. Claude Code does not scan Python
``site-packages`` for skills, so the skill directory is copied to the personal skill root
``~/.claude/skills/liana`` (available in every project). The copy is opt-in rather than done on
import or install, because writing into a user's home configuration silently would be surprising.

``--print-path`` prints the bundled directory instead, for symlinking it into ``~/.claude/skills``
so Claude Code tracks the installed package in place.
"""

import argparse
import shutil
import sys
from pathlib import Path

SKILL_NAME = "liana"


def bundled_skill_dir() -> Path:
    """Return the ``data/`` directory holding ``SKILL.md`` and ``references/``."""
    return Path(__file__).resolve().parent / "data"


def default_dest() -> Path:
    """Return ``~/.claude/skills/liana``."""
    return Path.home() / ".claude" / "skills" / SKILL_NAME


def install_skill(dest: Path | None = None, force: bool = False) -> Path:
    """Copy the bundled skill to ``dest`` (default ``~/.claude/skills/liana``).

    An existing ``dest`` is left untouched unless ``force`` is set, in which case it is replaced so
    that an upgrade reflects the installed package version exactly.
    """
    src = bundled_skill_dir()
    if not (src / "SKILL.md").is_file():
        raise FileNotFoundError(f"Bundled skill not found at {src}.")
    dest = default_dest() if dest is None else dest
    if dest.exists():
        if not force:
            raise FileExistsError(f"{dest} already exists. Re-run with --force to overwrite.")
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints"))
    return dest


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``liana-install-skills``."""
    parser = argparse.ArgumentParser(prog="liana-install-skills", description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=None, help="destination (default: ~/.claude/skills/liana)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing installation")
    parser.add_argument("--print-path", action="store_true", help="print the bundled skill directory and exit")
    args = parser.parse_args(argv)

    if args.print_path:
        print(bundled_skill_dir())
        return 0
    try:
        dest = install_skill(dest=args.dest, force=args.force)
    except (FileExistsError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"Installed liana skill to {dest}")
    print("Claude Code picks it up automatically; type /skills there to confirm it is listed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
