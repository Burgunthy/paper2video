"""수치 그라운딩 검증 — 원문에 없는 환각 수치 탐지·교정 (TAE-56).

2026-11-10 실사고: 냉장고 zero-shot "12%→64%"가 논문 어디에도 없었음.
"""
from paper2video.grounding import (
    GroundingResult, UngroundedNumber, correct_scenes,
    extract_claim_numbers, report_dict, verify,
)
from paper2video.llm import FakeLLMClient
from paper2video.types import Scene, ScriptDoc

# 실제 논문 본문 발췌(수치 포함) — 12%, 64%는 없음
SOURCE = """Our method with few-shot adaptation achieves an average success rate of 72%.
After iterative refinement, the overall success rate improves to 81%.
The oracle with perfect plans reaches 98%. We train with 2048 parallel
environments across 100 initial conditions using 512-unit MLPs."""


def _script(*narrations: str) -> ScriptDoc:
    return ScriptDoc(
        title="t",
        summary="s",
        scenes=[
            Scene(id=i + 1, narration=n, visual_type="slide",
                  visual_spec={"title": "t"}, duration_hint_sec=10.0)
            for i, n in enumerate(narrations)
        ],
    )


def test_extract_targets_percentages_and_big_numbers():
    text = "zero-shot이 72%에 few-shot 81%, 병렬 2,048개와 2048환경, 3단계와 512유닛, 두 번째 반복"
    nums = extract_claim_numbers(text)
    assert "72%" in nums and "81%" in nums
    assert "2,048" in nums and "2048" in nums
    assert "512" not in nums       # 3자리 맨숫자는 오탐 방지 제외
    assert "3" not in nums and "2" not in nums


def test_verify_flags_fabricated_fridge_numbers():
    """실사고 회귀: 12%·64%는 원문에 없고 72%·81%·98%는 있다."""
    narration = ("성공한 계획을 되먹이자 냉장고가 12%에서 64%로 뛰어오르고 "
                 "전체 평균도 81%까지 오릅니다. oracle은 98%입니다.")
    g = verify(_script(narration), SOURCE)
    assert {b.number for b in g.ungrounded} == {"12%", "64%"}
    assert all(b.scene_id == 1 for b in g.ungrounded)


def test_verify_passes_grounded_numbers():
    narration = "zero-shot 72%, few-shot 81%, oracle 98%, 병렬 환경 2048개로 학습합니다."
    assert verify(_script(narration), SOURCE).ungrounded == []


def test_correct_scenes_rewrites_via_llm():
    bad = [
        UngroundedNumber(1, "12%", "냉장고가 12%에서"),
        UngroundedNumber(1, "64%", "64%로 뛰어오르고"),
    ]
    fixed_narration = "성공한 계획을 되먹이자 냉장고가 크게 뛰어오르고 전체 평균도 81%까지 오릅니다."
    llm = FakeLLMClient([f'{{"narration": "{fixed_narration}"}}'])
    script = _script("성공한 계획을 되먹이자 냉장고가 12%에서 64%로 뛰어오르고 평균 81%.")
    out, n = correct_scenes(script, bad, llm)
    assert n == 1
    assert out.scenes[0].narration == fixed_narration
    assert verify(out, SOURCE).ungrounded == []


def test_correct_scenes_fallback_strips_numbers():
    """LLM 실패 시에도 환각 수치는 제거된다."""
    class BrokenLLM(FakeLLMClient):
        def complete_json(self, prompt, system=None):
            raise RuntimeError("llm down")

    bad = [UngroundedNumber(1, "12%", "s"), UngroundedNumber(1, "64%", "s")]
    script = _script("냉장고가 12%에서 64%로, 평균은 81%입니다.")
    out, n = correct_scenes(script, bad, BrokenLLM([]))
    assert n == 1
    assert "12%" not in out.scenes[0].narration
    assert "64%" not in out.scenes[0].narration
    assert "81%" in out.scenes[0].narration  # 근거 있는 수치는 보존


def test_report_shape():
    g = GroundingResult(ungrounded=[UngroundedNumber(1, "12%", "s")], unverified=[])
    r = report_dict(g, GroundingResult([], []))
    assert r["corrected"][0]["number"] == "12%"
    assert r["remaining_ungrounded"] == [] and r["unverified_table_sourced"] == []


def test_table_sourced_number_is_unverified_not_corrected():
    """표에는 독립 숫자로 있으나 % 문맥이 없는 수치 — 보존하고 unverified 기록."""
    src_text = "Task rows: Drawer 5 19 12 64 100 Fridge 27 6 87 84 100 Mean 19 32 72 81 98"
    g = verify(_script("서랍은 12%, 냉장고는 87%로 기록됐습니다. 평균은 72%입니다."), src_text)
    assert g.ungrounded == []                     # 숫자 흔적이 있으니 교정 안 함
    assert {u.number for u in g.unverified} == {"12%", "87%", "72%"}
