from __future__ import annotations
import json
import os
from .llm import LLMClient
from .research import ResearchResult
from .types import Scene, ScriptDoc


LANG_RULES = {
    "ko": """

=== LANGUAGE RULES (OVERRIDES ANY LANGUAGE INSTRUCTION ABOVE) ===
Write ALL user-facing content in natural Korean (한국어): title, summary, narration,
and every visual_spec field (title, direction, elements, caption).
- Narration: 구어체 (~합니다/~요 스타일) — 친구에게 설명하듯 자연스럽게.
- Keep widely-used English technical terms as-is when natural (예: 'Keypoint', '로봇 핸드').
- On-screen labels (title, elements, caption) are short Korean phrases (<= 25 chars).
=== END LANGUAGE RULES ===
""",
}


SCRIPT_PROMPT = """You are writing the script for a 2-5 minute explainer video in the style of
3Blue1Brown: clear, progressive, and HEAVILY VISUAL. The audience is a smart novice.

CRITICAL: This is NOT a slide deck. Every scene will be rendered as a Manim animation —
shapes moving, text writing in, diagrams being constructed, transforms. You must think in
terms of MOTION and METAPHOR, not titles and bullet points.

INPUT ARTICLE:
{source_text}

KEY POINTS (story beats, in order):
{key_points}

RESEARCH NOTES:
{notes}

Produce a JSON object with this exact shape:
{{
  "title": "short video title",
  "summary": "one-sentence summary",
  "scenes": [
    {{
      "narration": "what the narrator says, 1-3 sentences, plain text no markdown",
      "visual_type": "manim",
      "visual_spec": {{
        "title": "optional short label or heading (<=40 chars) — may be empty",
        "direction": "REQUIRED. A concrete, visual description of the animation: what objects appear, how they move, what transforms into what, what the viewer should be LOOKING AT. Write it like a shot description. Use verbs like 'fade in', 'transform into', 'slide from left', 'highlight', 'draw arrow from X to Y'. 2-4 sentences.",
        "elements": ["short list of key visual objects to draw, e.g. 'a green box labeled RAG', 'three stacked rectangles', 'a curved arrow'"],
        "caption": "optional 1-line text callout shown during the scene"
      }},
      "duration_hint_sec": 5.0
    }}
  ]
}}

Constraints:
- 5 to 10 scenes total.
- Each narration 1-3 sentences, spoken English, no jargon without a plain-English gloss.
- Total narration should read in 2-5 minutes (~150 words per minute).
- Every scene must have a non-empty `direction` field describing what ANIMATES.
- Avoid static text-heavy scenes. If a scene would just be "a title and bullets", REDESIGN it
  as something visual — e.g., boxes being connected, an arrow transforming, items being
  grouped into categories, a process flowing step by step.
- Do NOT reference LaTeX, equations, SVGs, or external images — pure Manim shapes + Text only.
- Output JSON only, no prose, no code fences.

Examples of GOOD directions:
- "Three labeled rectangles fade in on the left: SOURCES, WIKI, SCHEMA. Arrows draw between them showing data flow from sources into the wiki, with the schema arrow pointing at the wiki from above. The wiki rectangle pulses yellow to emphasize it as the center."
- "A single box labeled 'QUERY' appears. A magnifying-glass circle sweeps across a field of document rectangles on the right. Then the circle fades and a new answer box writes in at the bottom. The whole process replays faster to show 'every time you ask'."
- "Text 'Ingest' writes in at top. Below it, a file icon slides from the left into an existing wiki diagram. Lines from the new file extend to connect with existing wiki nodes, showing integration."

Examples of BAD directions (DO NOT DO THIS):
- "Show the title and three bullet points"
- "Display a slide with the key concepts"
- "Animate the text appearing"
""" 

SOCIAL_SCRIPT_PROMPT = """You are writing a 60-second vertical social video about a technical article.
Platform: TikTok, Instagram Reels, Xiaohongshu. Channel: @paperin60
Audience: curious builders who want to UNDERSTAND the idea, not just be entertained.

=== PACING RULES ===

The goal is LEARNING, not just retention. The viewer should walk away understanding
the core idea well enough to explain it to a friend. That means: clear visuals,
readable text, and enough breathing room for each concept to land.

1. HOOK IN 6 SECONDS. Scene 1 must stop the scroll. Pick ONE hook formula:

   A — Contrarian: "[Everyone thinks X]... but actually [surprising opposite]."
   B — Shocking stat: "[Specific number] that changes everything."
   C — "What if" scenario: "What if [familiar thing] was actually [reframe]?"
   D — Direct challenge: "You're doing [common thing] wrong."
   E — Pattern interrupt: Start mid-thought, as if mid-conversation.

   NEVER start with: "In this video", "Today we'll explore", "Let me explain".

2. TARGET 55-65 SECONDS total narration across 6-7 scenes.
   At ~150 words/minute, that's 140-160 words total. Don't go under 130 or over 170.

3. EACH SCENE IS 8-10 SECONDS. Enough time to show ONE idea clearly.
   Scene 1 (hook): 6-8 seconds.
   Middle scenes: 8-10 seconds each.
   Last scene (payoff): 8-10 seconds.

4. ONE IDEA PER SCENE. Each scene explains exactly ONE concept with ONE clear visual.
   If a scene tries to do two things, split it. The viewer should never feel rushed.

5. MINIMAL TEXT ON SCREEN. The animation should show DIAGRAMS and SHAPES, not walls of
   text. Maximum 3 short text labels visible at any moment (e.g. "Generator", "Evaluator",
   "Output" — not full sentences). The narration carries the explanation; the visual
   illustrates it. Let the narrator do the talking.

6. LAST SCENE answers "why should I care?" in one sentence.

7. NARRATION: each scene gets 1-2 sentences in plain spoken English. Cut jargon.
   The narration should sound like you're explaining it to a smart friend over coffee.

=== END PACING RULES ===

INPUT ARTICLE:
{source_text}

KEY POINTS:
{key_points}

RESEARCH NOTES:
{notes}

Produce JSON only with this exact shape:
{{
  "title": "punchy title (max 8 words, no colons)",
  "summary": "one-sentence summary",
  "scenes": [
    {{
      "id": 1,
      "narration": "one short spoken sentence — this is the HOOK",
      "visual_type": "manim",
      "visual_spec": {{
        "title": "2-4 word on-screen title",
        "direction": "REQUIRED: describe 2-3 visual beats that happen during this scene. What appears first, what changes, what the viewer's eye follows. Use verbs: appears, slides, transforms, highlights, connects, pulses. Something must move every 3 seconds.",
        "elements": ["key visual objects"],
        "caption": "optional 1-line callout"
      }},
      "duration_hint_sec": 8.0
    }}
  ]
}}

Rules:
- 5 to 6 scenes. No more, no less.
- Scene 1: hook (6-8 sec). Scenes 2-4/5: explanation (8-12 sec). Last scene: payoff (8-10 sec).
- Total narration ~100-120 words (45 seconds at speaking pace).
- Every direction field must describe multiple visual beats, not a static layout.
- No jargon without a gloss. Plain English only.
- Output JSON only. No prose. No code fences.
"""

SOCIAL_REVISION_PROMPT = """You are revising an existing short-form social video script.

REVISION INSTRUCTION:
{instruction}

CURRENT SCRIPT:
{script_json}

Return JSON only with this exact shape:
{{
  "title": "updated title",
  "summary": "updated summary",
  "changed_scene_ids": [1, 3],
  "scenes": [
    {{
      "id": 1,
      "narration": "updated narration",
      "visual_type": "manim",
      "visual_spec": {{
        "title": "short title",
        "direction": "visual direction",
        "elements": ["a", "b"],
        "caption": "optional"
      }},
      "duration_hint_sec": 8.0
    }}
  ]
}}

Rules:
- Keep unchanged scenes semantically consistent.
- Prefer targeted edits to the hook, pacing, clarity, or layout.
- Output the full updated script, plus changed_scene_ids.
- Output JSON only.
"""


VARIANTS: dict[str, dict] = {
    # V1 — depth fix only, no figures
    "deep": dict(
        label="V1 딥스크립트",
        scenes=(12, 14), minutes="5-7분", narration="3-5문장",
        figures=False,
        structure=(
            "Cover the full arc of the paper: (1) problem and why it is hard, "
            "(2) why prior approaches fall short, (3) the method's components "
            "in detail — name each one and say what it contributes, "
            "(4) experiments with concrete numbers (tasks, baselines, success "
            "rates, training time), (5) limitations and takeaways."
        ),
    ),
    # V2 — figure walkthrough
    "figurewalk": dict(
        label="V2 피규어 워크스루",
        scenes=(9, 11), minutes="4-5분", narration="2-4문장",
        figures=True, figure_min=6,
        structure=(
            "Walk through the paper's own figures as the backbone of the video: "
            "overview/tasks → architecture → training procedure → learning "
            "curves/results → robustness/qualitative behavior. Most scenes show "
            "a real figure; use manim only for an opening hook and abstract "
            "ideas no figure captures."
        ),
    ),
    # V3 — hybrid
    "hybrid": dict(
        label="V3 하이브리드",
        scenes=(13, 16), minutes="6-8분", narration="3-5문장",
        figures=True, figure_min=5,
        structure=(
            "Animated manim scenes explain concepts and mechanisms step by step; "
            "the paper's REAL figures appear for the architecture diagram, "
            "experiment setups, and result plots. Alternate deliberately: "
            "concept (manim) → evidence (figure) → analysis (manim)."
        ),
    ),
    # V4 — full lecture deep-dive
    "deepdive": dict(
        label="V4 풀 딥다이브",
        scenes=(17, 21), minutes="10-12분", narration="3-6문장",
        figures=True, figure_min=8,
        structure=(
            "A full lecture: (1) problem & motivation, (2) related approaches "
            "and the gap, (3) method deep-dive — EVERY major component, the "
            "training objective, and how human input enters learning, "
            "(4) experiment protocols, tasks and baselines, (5) headline "
            "results AND ablations with numbers, (6) failure modes and "
            "limitations, (7) takeaways. Use the paper's figures generously "
            "for architecture, task photos, learning curves, ablation tables."
        ),
    ),
    # V5 — compact hybrid
    "compact": dict(
        label="V5 컴팩트 하이브리드",
        scenes=(8, 9), minutes="3-4분", narration="2-3문장",
        figures=True, figure_min=4,
        structure=(
            "Tight and fast: problem → core mechanism (manim) → key results "
            "(real figures) → takeaway. No detours, but keep concrete numbers."
        ),
    ),
}

FIGURE_SCENE_SCHEMA = """When the best visual is the paper's OWN figure, emit a figure scene:
  "visual_type": "figure",
  "visual_spec": {{
    "figure_id": "fig2",
    "focus": "full|left|right|top|bottom|center|top-left|top-right|bottom-left|bottom-right",
    "highlight": "same vocab as focus — the region to outline during the scene, or empty string",
    "caption": "short Korean on-screen callout (<= 40 chars)"
  }}
Rules for figure scenes:
- figure_id MUST be copied exactly from the AVAILABLE FIGURES list. Never invent one.
- focus points the camera at the part of the figure the narration discusses.
- TALL FIGURES (height/width >= 1.5 in the AVAILABLE FIGURES pixel sizes): focus "full" is FORBIDDEN — the whole-figure fit shrinks it to an unreadable sliver. Pick the band the narration walks (top/bottom for row-stacked panels; left/right for column-stacked) or split the figure across two scenes, one band each.
- highlight: leave empty (deprecated — the renderer ignores it).
- NARRATION MUST WALK THE VIEWER THROUGH THE FIGURE, not talk past it: name what
  is on screen in reading order — the panel ("왼쪽 (a) 패널"), the axis or curve
  ("세로축이 오류, 아래로 갈수록 좋습니다"), the table row ("표에서 W-MPJPE 행"),
  the color or arrow being discussed. Every figure scene contains at least one
  such explicit visual pointer, and explains what the viewer is SEEING while the
  numbers/mechanism are read out of it. A figure scene whose narration never
  refers to the figure's own visual content is a failed scene.
- Figure references in narration MUST use the possessive spoken form:
  write "그림 6의 (a) 패널" (always with 의, and pair the panel letter with a
  position word like 왼쪽/오른쪽). NEVER write "그림 6 (a) 패널" — spoken aloud
  it sounds like "그림 6개의 패널" (six panels), which confuses viewers.
- EXTERNAL TOOL GROUNDING: when narration names an external model, dataset, or
  tracker (SAM3, Depth Anything 3, HaWoR, Ego4D, ...), its identity must be
  stated on first mention — what it is and what it does (input → output), in one
  clause. If the tool appears as a block in this paper's own figure, point the
  viewer at it there ("그림 2의 왼쪽 깊이 모듈이 바로 그것"). Dropping a name
  with no explanation ("오른쪽 핸드 트래커", "depth 모델") is a failed scene.
- {figure_min_rule}"""

SCRIPT_PROMPT_V2 = """You are writing the script for a {minutes} explainer video in the style of
3Blue1Brown: clear, progressive, heavily visual — but with LECTURE-LEVEL substance.
The audience is a smart novice who wants to actually UNDERSTAND the paper, not just hear it exists.

INPUT ARTICLE (the paper, in full):
{source_text}

KEY POINTS (story beats, in order):
{key_points}

RESEARCH NOTES:
{notes}
{figure_block}
STRUCTURE FOR THIS VIDEO:
{structure}

Produce a JSON object with this exact shape:
{{
  "title": "short video title",
  "summary": "one-sentence summary",
  "scenes": [
    {{
      "id": 1,
      "narration": "spoken narration, {narration}",
      "visual_type": "manim",
      "visual_spec": {{
        "title": "optional short on-screen label (<=40 chars), may be empty",
        "direction": "REQUIRED. A concrete visual description of the animation: what objects appear, how they move, what transforms into what. Use verbs like 'fade in', 'transform into', 'slide from left', 'highlight', 'draw arrow from X to Y'. 2-4 sentences.",
        "elements": ["key visual objects to draw"],
        "caption": "optional 1-line Korean callout"
      }},
      "duration_hint_sec": 20.0
    }}
  ]
}}

SUBSTANCE RULES — this is the most important part:
- Narration must TEACH concrete content. Every method scene names the actual
  components and explains the mechanism. Every results scene cites actual
  numbers from the paper (success rates, training hours, comparison gaps).
- FORBIDDEN: vague filler like "다양한 작업에서 좋은 성능을 보였습니다" without
  numbers, "효율적인 방법을 제안합니다" without saying WHAT makes it efficient.
- The viewer should be able to explain the paper's core mechanism to a colleague
  after watching. Depth beats polish.
- Progressively build understanding: each scene assumes exactly the previous scenes.
- EXTERNAL TOOL GROUNDING: any external model, dataset, or tracker named in
  narration gets its identity (what it does, input → output) on first mention —
  never a bare name. Point at this paper's figure when the tool is visible there.

FORM RULES:
- {scene_count} scenes total.
- Narration {narration} per scene, no markdown.
- duration_hint_sec ≈ your narration's spoken length (Korean ~5 chars/sec).
- manim scenes: pure Manim shapes + Text only — no LaTeX, no equations, no images.{figure_rule}
- Output JSON only, no prose, no code fences."""


def _figure_block(manifest: list[dict], variant: dict) -> str:
    if not variant.get("figures") or not manifest:
        return ""
    lines = ["", "AVAILABLE FIGURES (extracted from the paper):"]
    for f in manifest:
        cap = str(f.get("caption", "")).replace("\n", " ")[:400]
        size = ""
        if f.get("width_px") and f.get("height_px"):
            size = f" [{f['width_px']}x{f['height_px']}px]"
        lines.append(f"- {f['id']} (page {f.get('page', '?')}){size}: {cap}")
    figure_min = variant.get("figure_min", 0)
    rule = (
        f"At least {figure_min} scenes must be figure scenes."
        if figure_min
        else "Use figure scenes where they beat abstract animation."
    )
    return "\n".join(lines) + "\n\n" + FIGURE_SCENE_SCHEMA.format(figure_min_rule=rule)


OUTLINE_PROMPT = """You are planning the scene outline for a {minutes} explainer video about a paper.
Audience: smart novice who wants to actually understand it.

INPUT ARTICLE (the paper, in full):
{source_text}

KEY POINTS (story beats, in order):
{key_points}

RESEARCH NOTES:
{notes}
{figure_block}
STRUCTURE:
{structure}

Produce ONLY a JSON outline with this exact shape:
{{
  "title": "short video title",
  "summary": "one-sentence summary",
  "scenes": [
    {{
      "id": 1,
      "beat": "what this scene teaches — ONE short sentence (under 30 characters of Korean). Do NOT elaborate here; details come later in the scene chunks.",
      "visual_type": "manim",
      "figure_id": "fig2 or empty"
    }}
  ]
}}

Rules:
- {scene_count} scenes.
- visual_type is "manim" or "figure" only.{figure_use}
- Each beat must carry concrete substance (mechanism, numbers, comparisons) — no filler.
- Output JSON only, no prose, no code fences."""


SCENE_CHUNK_PROMPT = """You are writing FULL scenes {first}–{last} of a {minutes} explainer video script.
The outline and earlier scenes are fixed. Write complete, final scenes for the IDs listed below.

INPUT ARTICLE (the paper, in full):
{source_text}

FULL OUTLINE:
{outline}

SCENES ALREADY WRITTEN (for continuity — match tone and build on these):
{previous}

NOW WRITE FULLY (outline entries):
{chunk}
{figure_block}
Each scene must be JSON:
{{
  "id": {first},
  "narration": "spoken narration, {narration}, concrete substance (names, numbers), no markdown",
  "visual_type": "manim",
  "visual_spec": {{
    "title": "optional short on-screen label (<=40 chars), may be empty",
    "direction": "REQUIRED. Concrete animation description: objects, motion, transforms. 2-4 sentences.",
    "elements": ["key visual objects"],
    "caption": "optional short Korean callout"
  }},
  "duration_hint_sec": 20.0
}}
For figure scenes instead use:
{{
  "id": {first},
  "narration": "...",
  "visual_type": "figure",
  "visual_spec": {{
    "figure_id": "fig2",
    "focus": "full|left|right|top|bottom|center|top-left|top-right|bottom-left|bottom-right",
    "highlight": "region to outline, same vocab, or empty string",
    "caption": "short Korean on-screen callout (<= 40 chars)"
  }},
  "duration_hint_sec": 20.0
}}

Rules:
- duration_hint_sec ≈ narration spoken length (Korean ~5 chars/sec).
- figure_id must come from the AVAILABLE FIGURES list only.
- TALL FIGURES (height/width >= 1.5): never focus "full" — pick the band the narration walks (top/bottom/left/right) or one band per scene across two scenes.
- Output ONLY: {{"scenes": [...]}} with exactly the {n} requested scenes, in order.
- No prose, no code fences."""


def _parse_scenes(raw: dict, start_id: int) -> list[Scene]:
    scenes: list[Scene] = []
    for i, s in enumerate(raw.get("scenes", []), start=start_id):
        scenes.append(
            Scene(
                id=s.get("id") or i,
                narration=s["narration"],
                visual_type=s.get("visual_type", "manim"),
                visual_spec=s.get("visual_spec", {}),
                duration_hint_sec=float(s.get("duration_hint_sec", 5.0)),
            )
        )
    return scenes


def write_script_chunked(research_result: ResearchResult, llm: LLMClient, v: dict,
                         figure_manifest: list[dict] | None = None) -> ScriptDoc:
    """Two-phase script writing for long variants: outline, then scenes in
    chunks of 4. Keeps every LLM response small enough to stay valid JSON."""
    notes_str = "\n".join(f"- {n.claim}" for n in research_result.notes) or "(none)"
    key_points_str = "\n".join(f"- {p}" for p in research_result.key_points) or "(none)"
    manifest = figure_manifest or []
    fblock = _figure_block(manifest, v)
    lang = LANG_RULES.get(os.environ.get("PAPER2VIDEO_LANG", "").lower(), "")

    outline_raw = llm.complete_json(
        OUTLINE_PROMPT.format(
            minutes=v["minutes"],
            source_text=research_result.source_text,
            key_points=key_points_str,
            notes=notes_str,
            figure_block=fblock,
            structure=v["structure"],
            scene_count=f"{v['scenes'][0]} to {v['scenes'][1]}",
            figure_use=(
                " Use the paper's real figures (figure_id from the list) where they beat animation."
                if v.get("figures") else " Do NOT use figures."
            ),
        ) + lang
    )
    title = outline_raw.get("title", "Untitled")
    summary = outline_raw.get("summary", "")
    outline_scenes = outline_raw.get("scenes", [])
    outline_json = json.dumps(outline_raw, ensure_ascii=False, indent=1)

    chunk_size = 4
    scenes: list[Scene] = []
    for cstart in range(0, len(outline_scenes), chunk_size):
        chunk = outline_scenes[cstart:cstart + chunk_size]
        if not chunk:
            continue
        first_id = chunk[0].get("id", cstart + 1)
        previous = "\n".join(
            f"- scene {s.id}: {s.narration[:120]}" for s in scenes[-4:]
        ) or "(none — this is the opening)"
        raw = llm.complete_json(
            SCENE_CHUNK_PROMPT.format(
                first=first_id,
                last=chunk[-1].get("id", first_id + len(chunk) - 1),
                minutes=v["minutes"],
                source_text=research_result.source_text,
                outline=outline_json,
                previous=previous,
                chunk=json.dumps(chunk, ensure_ascii=False, indent=1),
                figure_block=fblock,
                narration=v["narration"],
                n=len(chunk),
            ) + lang
        )
        scenes.extend(_parse_scenes(raw, first_id))

    return ScriptDoc(title=title, summary=summary, scenes=scenes)


def write_script(research_result: ResearchResult, llm: LLMClient,
                 variant: str | None = None, figure_manifest: list[dict] | None = None) -> ScriptDoc:
    variant_name = variant or os.environ.get("PAPER2VIDEO_VARIANT", "classic")
    notes_str = "\n".join(f"- {n.claim}" for n in research_result.notes) or "(none)"
    key_points_str = "\n".join(f"- {p}" for p in research_result.key_points) or "(none)"

    if variant_name in VARIANTS:
        v = VARIANTS[variant_name]
        # Long scripts (13+ scenes) exceed what GLM returns as valid JSON in one
        # shot — write those as outline + scene chunks instead.
        if v["scenes"][1] >= 13:
            return write_script_chunked(research_result, llm, v, figure_manifest)
        prompt = SCRIPT_PROMPT_V2.format(
            minutes=v["minutes"],
            source_text=research_result.source_text,
            key_points=key_points_str,
            notes=notes_str,
            figure_block=_figure_block(figure_manifest or [], v),
            structure=v["structure"],
            narration=v["narration"],
            scene_count=f"{v['scenes'][0]} to {v['scenes'][1]}",
            figure_rule=(
                " Figure scenes may reference figure_id from the list above — that is the ONLY image use allowed."
                if v.get("figures") else " Do NOT reference any images or figures."
            ),
        )
        prompt += LANG_RULES.get(os.environ.get("PAPER2VIDEO_LANG", "").lower(), "")
    else:
        prompt = SCRIPT_PROMPT.format(
            source_text=research_result.source_text[:10000],
            key_points=key_points_str,
            notes=notes_str,
        ) + LANG_RULES.get(os.environ.get("PAPER2VIDEO_LANG", "").lower(), "")

    raw = llm.complete_json(prompt)

    scenes: list[Scene] = []
    for i, s in enumerate(raw.get("scenes", []), start=1):
        scenes.append(
            Scene(
                id=s.get("id") or i,
                narration=s["narration"],
                visual_type=s.get("visual_type", "slide"),
                visual_spec=s.get("visual_spec", {}),
                duration_hint_sec=float(s.get("duration_hint_sec", 5.0)),
            )
        )

    return ScriptDoc(
        title=raw.get("title", "Untitled"),
        summary=raw.get("summary", ""),
        scenes=scenes,
    )


def write_social_script(research_result: ResearchResult, llm: LLMClient) -> ScriptDoc:
    notes_str = "\n".join(f"- {n.claim}" for n in research_result.notes) or "(none)"
    key_points_str = "\n".join(f"- {p}" for p in research_result.key_points[:4]) or "(none)"
    raw = llm.complete_json(
        SOCIAL_SCRIPT_PROMPT.format(
            source_text=research_result.source_text[:10000],
            key_points=key_points_str,
            notes=notes_str,
        )
    )
    scenes = [
        Scene(
            id=s.get("id") or i,
            narration=s["narration"],
            visual_type=s.get("visual_type", "manim"),
            visual_spec=s.get("visual_spec", {}),
            duration_hint_sec=float(s.get("duration_hint_sec", 8.0)),
        )
        for i, s in enumerate(raw.get("scenes", []), start=1)
    ]
    return ScriptDoc(
        title=raw.get("title", "Untitled"),
        summary=raw.get("summary", ""),
        scenes=scenes,
    )


def revise_social_script(script_doc: ScriptDoc, instruction: str, llm: LLMClient) -> tuple[ScriptDoc, list[int]]:
    raw = llm.complete_json(
        SOCIAL_REVISION_PROMPT.format(
            instruction=instruction.strip(),
            script_json=json.dumps(script_doc.model_dump(), ensure_ascii=False, indent=2),
        )
    )
    scenes = [
        Scene(
            id=s.get("id") or i,
            narration=s["narration"],
            visual_type=s.get("visual_type", "manim"),
            visual_spec=s.get("visual_spec", {}),
            duration_hint_sec=float(s.get("duration_hint_sec", 8.0)),
        )
        for i, s in enumerate(raw.get("scenes", []), start=1)
    ]
    return (
        ScriptDoc(
            title=raw.get("title", script_doc.title),
            summary=raw.get("summary", script_doc.summary),
            scenes=scenes,
        ),
        [int(scene_id) for scene_id in raw.get("changed_scene_ids", [])],
    )
