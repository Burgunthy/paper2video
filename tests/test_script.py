from paper2video.llm import FakeLLMClient
from paper2video.research import ResearchResult
from paper2video.script import write_script
from paper2video.types import ScriptDoc


def test_write_script_returns_valid_script_doc():
    research_result = ResearchResult(
        concepts=["LLM", "wiki"],
        notes=[],
        key_points=["intro", "body", "outro"],
        source_text="article text",
    )
    llm_response = """{
      "title": "LLMs as Wiki Builders",
      "summary": "How LLMs maintain persistent wikis.",
      "scenes": [
        {"id": 1, "narration": "Welcome.", "visual_type": "slide",
         "visual_spec": {"title": "LLM Wikis", "bullets": ["a"]}, "duration_hint_sec": 5.0},
        {"id": 2, "narration": "Here's why.", "visual_type": "slide",
         "visual_spec": {"title": "Why", "bullets": ["b"]}, "duration_hint_sec": 6.0}
      ]
    }"""
    llm = FakeLLMClient(responses=[llm_response])
    doc = write_script(research_result, llm=llm)
    assert isinstance(doc, ScriptDoc)
    assert doc.title == "LLMs as Wiki Builders"
    assert len(doc.scenes) == 2
    assert doc.scenes[0].narration == "Welcome."
    assert doc.scenes[1].visual_spec["title"] == "Why"


def test_write_script_enforces_scene_ids():
    research_result = ResearchResult(concepts=[], notes=[], key_points=["x"], source_text="t")
    llm = FakeLLMClient(responses=["""{
      "title": "T", "summary": "S",
      "scenes": [
        {"narration": "a", "visual_type": "slide", "visual_spec": {"title": "A"}, "duration_hint_sec": 3.0},
        {"narration": "b", "visual_type": "slide", "visual_spec": {"title": "B"}, "duration_hint_sec": 3.0}
      ]
    }"""])
    doc = write_script(research_result, llm=llm)
    assert [s.id for s in doc.scenes] == [1, 2]


def test_figure_block_demands_visual_walkthrough_narration():
    """Figure scenes must be narrated THROUGH the figure: the prompt has to
    force explicit panel/axis/curve/table-row pointers, give the writer the
    full caption (not 150 chars) and the figure's pixel size."""
    from paper2video.script import _figure_block, VARIANTS

    manifest = [{
        "id": "fig2", "page": 4,
        "caption": "cap " * 120,          # 480 chars
        "width_px": 1134, "height_px": 584,
    }]
    block = _figure_block(manifest, VARIANTS["deepdive"])

    low = block.lower()
    # Narration-side rule: explicit visual pointers required
    assert "narration" in low
    assert "panel" in low or "subplot" in low
    assert "walk" in low
    # Writer sees the full caption (>=400 chars), not a 150-char stub
    assert "cap " * 120 in block or len([l for l in block.splitlines() if "fig2" in l and "cap" in l][0]) > 400
    # Writer sees figure geometry (helps infer multi-panel layouts)
    assert "1134" in block


def test_scene_chunk_prompt_carries_figure_narration_rule():
    from paper2video.script import SCENE_CHUNK_PROMPT, FIGURE_SCENE_SCHEMA

    # The chunk writer is what actually produces narration for long variants;
    # the rule must live in the shared figure schema it embeds.
    assert "panel" in FIGURE_SCENE_SCHEMA.lower()
    assert "narration" in FIGURE_SCENE_SCHEMA.lower()
    assert "{figure_block}" in SCENE_CHUNK_PROMPT
