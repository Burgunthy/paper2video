"""대본 수치의 원문 그라운딩 검증 — 렌더 전에 환각 수치를 차단한다.

2026-11-10 사고(TAE-56): research 단계가 표 텍스트 추출이 뒤섞인 상태에서
수치를 잘못 귀속(냉장고 zero-shot "12%→64%" — 실제로는 다른 과제 행의 값)해
대본·영상까지 실음.

검증 정책(텍스트 수준 v1):
- 강(firm): 원문에 "N%" 문자 그대로 존재 → 통과.
- 약(table-sourced): "N%"는 없으나 독립 숫자 토큰 N이 존재(표 셀 추정) →
  통과하되 grounding_report.json에 unverified로 기록해 검토 가능하게 한다.
- 부재(absent): 둘 다 없음 → 순수 환각이므로 해당 씬 교정(제거/근거 교체).

한계(정직한 고지): 표 행이 뒤섞여 '잘못된 과제에 귀속'된 수치는 토큰이
원문에 존재하므로 텍스트 게이트로 잡을 수 없다 — 그래프 원본을 VLM으로
대조하는 것이 v2 과제다. research 프롬프트의 수치 인용 규칙이 사전 차단을
담당하고, 본 게이트가 잔여 환각을 흡수한다.

1~2자리 맨숫자("3단계", "두 번째", "20개")는 원문 등장 빈도가 높아
검증 의미가 없어(오탐만 유발) 대상에서 제외한다.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .llm import LLMClient
from .types import ScriptDoc

PCT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")
BIG_RE = re.compile(r"(?<![\d.,])(?:\d{1,3}(?:,\d{3})+|\d{4,6})(?!\d)")


@dataclass(frozen=True)
class UngroundedNumber:
    scene_id: int
    number: str
    snippet: str


@dataclass(frozen=True)
class UnverifiedNumber:
    """표에서 온 것으로 보이나 % 문맥으로는 확인 안 된 수치 — 보존·기록."""

    scene_id: int
    number: str
    snippet: str


@dataclass(frozen=True)
class GroundingResult:
    ungrounded: list[UngroundedNumber]   # 부재 — 교정 대상
    unverified: list[UnverifiedNumber]   # 표 추정 — 보존·기록


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).replace(" %", "%")


def extract_claim_numbers(text: str) -> list[str]:
    """검증 대상 수치 토큰 추출 — 백분율(전부) + 큰 수(쉼표 묶음 또는 4자리+)."""
    out: list[str] = []
    out.extend(PCT_RE.findall(text))
    out.extend(BIG_RE.findall(text))
    seen: set[str] = set()
    res: list[str] = []
    for n in out:
        n = _norm(n)
        if n and n not in seen:
            seen.add(n)
            res.append(n)
    return res


def _variants(number: str) -> set[str]:
    vs = {_norm(number)}
    if "," in number:
        vs.add(_norm(number.replace(",", "")))
    return vs


def _standalone_in(src: str, n: str) -> bool:
    """독립 숫자 토큰(표 셀 등)로 존재하는지 — 자릿수 경계 엄격히."""
    return re.search(rf"(?<![\d.,]){re.escape(n)}(?![\d.,%])", src) is not None


def verify(script: ScriptDoc, source_text: str) -> GroundingResult:
    """대본의 각 수치가 원문에 존재하는지 검증."""
    src = _norm(source_text)
    ungrounded: list[UngroundedNumber] = []
    unverified: list[UnverifiedNumber] = []
    for sc in script.scenes:
        for n in extract_claim_numbers(sc.narration):
            if any(v in src for v in _variants(n)):
                continue  # 강 — % 문맥으로 확인됨
            if _standalone_in(src, n.rstrip("%")):
                unverified.append(UnverifiedNumber(sc.id, n, sc.narration[:80]))
            else:
                ungrounded.append(UngroundedNumber(sc.id, n, sc.narration[:80]))
    return GroundingResult(ungrounded=ungrounded, unverified=unverified)


CORRECT_PROMPT = """논문 해설 영상의 내레이션에 원문에 존재하지 않는 수치가 섞여 있다.

원문에 없는 수치: {numbers}

규칙:
- 원문에 없는 수치는 제거하거나, 확실히 원문에 있는 사실·수치로 바꾼다.
- 확신할 수 없는 수치는 정성 서술로 바꾼다 (예: "크게 올랐습니다").
- 그 외 내용·어투·분량은 최대한 유지한다.
- 출력은 JSON만: {{"narration": "..."}}"""


def correct_scenes(
    script: ScriptDoc, ungrounded: list[UngroundedNumber], llm: LLMClient,
) -> tuple[ScriptDoc, int]:
    """근거 없는 수치가 든 씬 내레이션을 LLM으로 교정. (script_doc, 교정 씬 수) 반환.

    LLM 교정 실패 시 최후 수단으로 해당 수치 토큰만 제거한다 —
    환각 수치가 남는 것보다 낫다.
    """
    by_scene: dict[int, list[str]] = {}
    for u in ungrounded:
        by_scene.setdefault(u.scene_id, []).append(u.number)
    fixed = 0
    for sc in script.scenes:
        nums = by_scene.get(sc.id)
        if not nums:
            continue
        new_narration = ""
        try:
            out = llm.complete_json(
                CORRECT_PROMPT.format(numbers=", ".join(nums), narration=sc.narration)
            )
            new_narration = str(out.get("narration", "")).strip()
        except Exception:
            new_narration = ""
        if new_narration and new_narration != sc.narration:
            sc.narration = new_narration
        else:
            for n in nums:
                sc.narration = re.sub(re.escape(n), "", sc.narration)
            sc.narration = re.sub(r"\s{2,}", " ", sc.narration)
        fixed += 1
    return script, fixed


def report_dict(
    initial: GroundingResult, remaining: GroundingResult,
) -> dict:
    return {
        "corrected": [asdict(u) for u in initial.ungrounded],
        "remaining_ungrounded": [asdict(u) for u in remaining.ungrounded],
        "unverified_table_sourced": [asdict(u) for u in initial.unverified],
    }
