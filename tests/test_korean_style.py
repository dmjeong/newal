"""Korean prose in the repository.

The rules live in AGENTS.md; these are the ones a machine can check without
guessing. They are deliberately narrow. A style checker that cries wolf gets
switched off, and then it checks nothing at all.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HANGUL = re.compile(r"[가-힣]")

#: Phrases that read as English wearing Korean grammar, and what to write instead.
TRANSLATIONESE = {
    "것입니다": "~합니다",
    "을 통해": "~(으)로",
    "를 통해": "~(으)로",
    "에 의해": "능동으로",
    "되어집니다": "~됩니다",
    "가지고 있습니다": "있습니다",
    "제공합니다": "줍니다 / 씁니다",
    "수행합니다": "합니다",
    "존재합니다": "있습니다",
    "기여합니다": "~을 올립니다",
    "할 필요가 있습니다": "~해야 합니다",
}


def _korean_files() -> list[Path]:
    """Every tracked file that carries user-facing Korean."""
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, cwd=ROOT, check=True
    ).stdout.split()
    wanted = {".md", ".py", ".js", ".html"}
    return [
        ROOT / name
        for name in tracked
        if (ROOT / name).suffix in wanted and HANGUL.search((ROOT / name).read_text("utf-8"))
    ]


def _korean_prose(path: Path) -> list[tuple[int, str]]:
    """Korean lines, minus fenced code blocks where examples of bad style live."""
    out: list[tuple[int, str]] = []
    fenced = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced and HANGUL.search(line):
            out.append((number, line))
    return out


def test_there_is_korean_to_check():
    """Guard the guard: a broken file walk would make every test below vacuous."""
    assert len(_korean_files()) > 5


@pytest.mark.parametrize("phrase,better", sorted(TRANSLATIONESE.items()))
def test_translationese_is_absent(phrase, better):
    found = [
        f"{path.relative_to(ROOT)}:{number}: {line.strip()}"
        for path in _korean_files()
        for number, line in _korean_prose(path)
        if phrase in line and path.name != "AGENTS.md"  # the rule sheet quotes them
    ]
    assert not found, f"'{phrase}' 대신 '{better}':\n" + "\n".join(found)


#: A finished Korean clause: the polite verb endings a sentence stops on.
CLAUSE_END = re.compile(r"(습니다|합니다|입니다|됩니다|랍니다|납니다|십니다|었다|이다|한다)$")


def _ends_a_clause(text: str) -> bool:
    """Is this a finished sentence rather than a label?

    Strips the markup that would otherwise hide the verb ending: ``**굵게**``,
    ``</b>``, a closing quote. What is left is the last word the reader hears.
    """
    bare = re.sub(r"(\*\*|<[^>]+>|[`\"'*_)\]])+$", "", text.strip())
    return bool(CLAUSE_END.search(bare))


def test_the_em_dash_does_not_split_a_korean_sentence():
    """An em-dash mid-sentence is an English move; Korean ends the sentence.

    The test is what separates the two uses. Splitting a finished clause off
    with a dash is the English habit. Putting one between a label and its gloss
    (``SFT — 통과한 턴 따라하기``) is a separator, which AGENTS.md allows, so the
    rule fires only when the left-hand side is already a complete sentence.
    """
    offenders = []
    for path in _korean_files():
        if path.name == "AGENTS.md":  # the rule sheet quotes what it forbids
            continue
        for number, line in _korean_prose(path):
            stripped = line.strip()
            if "—" not in stripped or stripped.startswith(("|", "#")):
                continue
            if _ends_a_clause(stripped.split("—")[0]):
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {stripped}")

    assert not offenders, "문장 중간의 줄표를 마침표나 쉼표로 바꾸세요:\n" + "\n".join(offenders)


def test_the_em_dash_rule_tells_a_label_from_a_sentence():
    """Guard the heuristic itself, or it could quietly start passing everything."""
    assert _ends_a_clause("나머지는 다 됩니다")
    assert _ends_a_clause("**해상도보다 프레임 수를 먼저 줄입니다**")
    assert _ends_a_clause("<title>여기서 끝납니다</title>")
    assert not _ends_a_clause("SFT")
    assert not _ends_a_clause("1. **`embedding`** (~1GB)")
    assert not _ends_a_clause("QLoRA")


def test_the_rule_sheet_is_where_the_loader_looks():
    """prompts.py picks the first of these it finds, so the name matters."""
    from newal.agent.prompts import load_project_doc

    doc = load_project_doc(ROOT)
    assert doc is not None
    assert "줄표" in doc, "the Korean rules must be in the file the agent actually loads"
