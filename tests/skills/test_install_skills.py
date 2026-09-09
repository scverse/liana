import ast
import inspect
import re
from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture

import liana as li
from liana._skills.install import bundled_skill_dir, install_skill, main

DATA = bundled_skill_dir()
REFS = sorted((DATA / "references").glob("*.md"))
ALL_MD = [DATA / "SKILL.md", *REFS]


def test_bundled_skill_layout() -> None:
    assert (DATA / "SKILL.md").is_file()
    assert len(REFS) >= 5


def test_frontmatter() -> None:
    text = (DATA / "SKILL.md").read_text()
    assert text.startswith("---\n")
    fields = dict(re.findall(r"^(\w+):\s*(.*)$", text.split("---\n", 2)[1], flags=re.MULTILINE))
    assert fields["name"] == "liana"
    assert 0 < len(fields["description"]) <= 1024
    assert "<" not in fields["description"]


def test_skill_md_is_short() -> None:
    assert len((DATA / "SKILL.md").read_text().splitlines()) < 500


def test_links_resolve_one_level_deep() -> None:
    for md in ALL_MD:
        for target in re.findall(r"\]\(([^)#]+\.md)\)", md.read_text()):
            resolved = (md.parent / target).resolve()
            assert resolved.is_file(), f"{md.name} -> {target}"
            if md.name != "SKILL.md":
                # reference files may only point at siblings, never deeper
                assert resolved.parent == md.parent, f"{md.name} -> {target}"
    linked = {
        (DATA / t).resolve() for t in re.findall(r"\]\((references/[^)]+\.md)\)", (DATA / "SKILL.md").read_text())
    }
    assert linked == {r.resolve() for r in REFS}, "every reference file must be linked from SKILL.md"


@pytest.mark.parametrize("md", ALL_MD, ids=lambda p: p.name)
def test_public_symbols_exist(md: Path) -> None:
    # every `li.<sub>.<name>` mentioned in the skill must resolve on the installed package
    symbols = set(re.findall(r"`li\.(\w+)\.(\w+)", md.read_text()))
    missing = [f"li.{a}.{b}" for a, b in symbols if not hasattr(getattr(li, a, None), b)]
    assert not missing, missing


def _li_calls(block: str) -> list[tuple[str, ast.Call]]:
    calls = []
    for node in ast.walk(ast.parse(block)):
        if not isinstance(node, ast.Call):
            continue
        parts, f = [], node.func
        while isinstance(f, ast.Attribute):
            parts.append(f.attr)
            f = f.value
        if isinstance(f, ast.Name) and f.id == "li":
            calls.append((".".join(reversed(parts)), node))
    return calls


@pytest.mark.parametrize("md", REFS, ids=lambda p: p.name)
def test_code_blocks_carry_no_defaults(md: Path) -> None:
    # code blocks show decisions, not signatures: every block parses, every keyword exists,
    # and no constant keyword restates the function's default (docstrings already do that)
    problems = []
    for block in re.findall(r"```python\n(.*?)```", md.read_text(), re.S):
        for dotted, call in _li_calls(block):
            obj: object = li
            for part in dotted.split("."):
                obj = getattr(obj, part)
            params = inspect.signature(obj).parameters  # type: ignore[arg-type]
            var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
            for kw in call.keywords:
                if kw.arg is None:
                    continue
                if kw.arg not in params:
                    if not var_kw:
                        problems.append(f"li.{dotted}: unknown keyword {kw.arg}")
                    continue
                default = params[kw.arg].default
                if (
                    isinstance(kw.value, ast.Constant)
                    and default is not inspect.Parameter.empty
                    and kw.value.value == default
                ):
                    problems.append(f"li.{dotted}: {kw.arg}={default!r} is the default")
    assert not problems, problems


def test_install_and_force(tmp_path: Path) -> None:
    dest = install_skill(dest=tmp_path / "liana")
    assert (dest / "SKILL.md").is_file() and (dest / "references").is_dir()
    with pytest.raises(FileExistsError):
        install_skill(dest=dest)
    assert install_skill(dest=dest, force=True) == dest


def test_cli(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    assert main(["--print-path"]) == 0
    assert capsys.readouterr().out.strip() == str(DATA)
    assert main(["--dest", str(tmp_path / "s")]) == 0
    assert main(["--dest", str(tmp_path / "s")]) == 1
    assert Path(tmp_path / "s" / "SKILL.md").is_file()
