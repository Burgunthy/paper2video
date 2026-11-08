from paper2video.renderers.manim_r import _strip_code_fences, _compose_visual_direction, lint_manim_code


def test_strip_code_fences_removes_python_fence():
    raw = "```python\nfrom manim import *\nclass MainScene(Scene):\n    pass\n```"
    out = _strip_code_fences(raw)
    assert out.startswith("from manim")
    assert out.endswith("pass")
    assert "```" not in out


def test_strip_code_fences_handles_plain_fence():
    raw = "```\nx = 1\n```"
    assert _strip_code_fences(raw) == "x = 1"


def test_strip_code_fences_leaves_plain_code_alone():
    raw = "from manim import *\nclass MainScene(Scene):\n    pass"
    assert _strip_code_fences(raw) == raw


def test_compose_visual_direction_uses_all_fields():
    spec = {
        "title": "The Flip",
        "direction": "Transform an arrow pointing right into one pointing left",
        "elements": ["box A", "box B", "arrow"],
        "bullets": ["reveal A", "reveal B", "connect"],
        "caption": "this is the key insight",
    }
    out = _compose_visual_direction(spec)
    assert "The Flip" in out
    assert "Transform an arrow" in out
    assert "box A" in out
    assert "reveal A" in out
    assert "this is the key insight" in out


def test_compose_visual_direction_empty_spec():
    assert "invent" in _compose_visual_direction({})


# ---- linter tests ----

GOOD_CODE = """\
from manim import *

class MainScene(Scene):
    def construct(self):
        def fit(mobj, max_w=11.0, max_h=6.0):
            s = min(max_w / mobj.width, max_h / mobj.height, 1.0)
            if s < 1.0:
                mobj.scale(s)
            return mobj

        title = Text("Hello", font_size=44)
        fit(title, max_w=11.0)
        title.to_edge(UP, buff=0.6)
        self.play(Write(title))
        self.wait(1)
"""


def test_lint_passes_on_good_code():
    assert lint_manim_code(GOOD_CODE) == []


def test_lint_catches_scale_to_fit_width():
    code = GOOD_CODE.replace("fit(title, max_w=11.0)", "title.scale_to_fit_width(10)")
    errors = lint_manim_code(code)
    assert any("scale_to_fit_width" in e for e in errors)


def test_lint_catches_scale_to_fit_height():
    code = GOOD_CODE.replace("fit(title, max_w=11.0)", "title.scale_to_fit_height(5)")
    errors = lint_manim_code(code)
    assert any("scale_to_fit" in e for e in errors)


def test_lint_catches_huge_shift():
    bad = GOOD_CODE.replace("self.wait(1)", "title.shift(DOWN * 4)\n        self.wait(1)")
    errors = lint_manim_code(bad)
    assert any("shift" in e for e in errors)


def test_lint_catches_fly_in_anti_pattern():
    bad = GOOD_CODE.replace(
        "self.wait(1)",
        "title.shift(UP * 2 + LEFT * 2)\n        self.wait(1)",
    )
    errors = lint_manim_code(bad)
    assert any("fly in" in e.lower() or "shift" in e for e in errors)


def test_lint_catches_exact_1_5_shift():
    bad = GOOD_CODE.replace("self.wait(1)", "title.shift(UP * 1.5)\n        self.wait(1)")
    errors = lint_manim_code(bad)
    assert any("shift" in e for e in errors)


def test_lint_allows_small_shift():
    ok = GOOD_CODE.replace("self.wait(1)", "title.shift(UP * 0.3)\n        self.wait(1)")
    assert lint_manim_code(ok) == []


def test_lint_allows_animate_shift_large():
    # .animate.shift() is fine — it's a proper animation to a new position
    ok = GOOD_CODE.replace(
        "self.play(Write(title))",
        "self.play(title.animate.shift(UP * 3))"
    )
    assert lint_manim_code(ok) == []


def test_lint_catches_missing_main_scene():
    code = "from manim import *\n\nclass NotIt(Scene):\n    def construct(self):\n        pass\n"
    errors = lint_manim_code(code)
    assert any("MainScene" in e for e in errors)


def test_lint_catches_forbidden_import():
    code = "from manim import *\nimport numpy as np\n\nclass MainScene(Scene):\n    def construct(self):\n        pass\n"
    errors = lint_manim_code(code)
    assert any("forbidden import" in e for e in errors)


def test_lint_catches_mathtex():
    code = GOOD_CODE.replace("Text(\"Hello\", font_size=44)", "MathTex(r\"x^2\")")
    errors = lint_manim_code(code)
    assert any("LaTeX" in e or "MathTex" in e for e in errors)


def test_lint_catches_fadeout_at_end():
    bad = GOOD_CODE.replace(
        "self.wait(1)",
        "self.play(FadeOut(title))\n        self.wait(1)"
    )
    errors = lint_manim_code(bad)
    assert any("fade out" in e.lower() or "FadeOut" in e for e in errors)


def test_lint_allows_fadeout_in_middle():
    code = """\
from manim import *

class MainScene(Scene):
    def construct(self):
        def fit(mobj, max_w=11.0, max_h=6.0):
            s = min(max_w / mobj.width, max_h / mobj.height, 1.0)
            if s < 1.0:
                mobj.scale(s)
            return mobj
        a = Text("a", font_size=44)
        b = Text("b", font_size=44)
        self.play(Write(a))
        self.play(FadeOut(a))
        self.play(Write(b))
        self.wait(2)
"""
    assert lint_manim_code(code) == []


def test_lint_catches_tiny_scale():
    bad = GOOD_CODE.replace("self.play(Write(title))", "title.scale(0.1)\n        self.play(Write(title))")
    errors = lint_manim_code(bad)
    assert any("scale" in e and "0.1" in e for e in errors)


def test_lint_catches_huge_scale():
    bad = GOOD_CODE.replace("self.play(Write(title))", "title.scale(10)\n        self.play(Write(title))")
    errors = lint_manim_code(bad)
    assert any("scale" in e and "10" in e for e in errors)


def test_lint_catches_animate_scale_tiny():
    bad = GOOD_CODE.replace(
        "self.play(Write(title))",
        "self.play(title.animate.scale(0.2))"
    )
    errors = lint_manim_code(bad)
    assert any("scale" in e for e in errors)


def test_lint_allows_small_scale_adjustments():
    ok = GOOD_CODE.replace(
        "self.play(Write(title))",
        "title.scale(1.1)\n        self.play(Write(title))"
    )
    assert lint_manim_code(ok) == []


def test_lint_catches_uncreate_at_end():
    bad = GOOD_CODE.replace(
        "self.wait(1)",
        "self.play(Uncreate(title))\n        self.wait(1)"
    )
    errors = lint_manim_code(bad)
    assert any("Uncreate" in e for e in errors)


def test_lint_catches_fragile_progress_bar_alignment():
    bad = GOOD_CODE.replace(
        "self.play(Write(title))",
        "fill.move_to(bg.get_left() + RIGHT * (w / 2))\n        self.play(Write(title))",
    )
    errors = lint_manim_code(bad)
    assert any("progress bar" in e.lower() or "align_to" in e for e in errors)


def test_lint_catches_fit_call_without_definition():
    bad = """\
from manim import *

class MainScene(Scene):
    def construct(self):
        title = Text("Hi", font_size=44)
        fit(title, 10)
        self.play(Write(title))
"""
    errors = lint_manim_code(bad)
    assert any("def fit" in e for e in errors)


def test_codegen_prompt_requires_korean_text_and_no_overlap():
    """Generated manim scenes came out with English labels overlapping bars."""
    from paper2video.renderers.manim_r import CODEGEN_SYSTEM
    low = CODEGEN_SYSTEM.lower()
    assert "korean" in low
    assert "overlap" in low


def test_lint_rejects_text_mirroring():
    """Rotate(axis=UP/DOWN) flips text to a mirror image (DPP 14:16 bug),
    and .flip(/scale(- mirror too. The static check must reject them."""
    from paper2video.renderers.manim_r import lint_manim_code
    code = """
from manim import *
class MainScene(Scene):
    def construct(self):
        card = VGroup(Text("한글", font="Noto Sans CJK KR"))
        self.play(Rotate(card, angle=PI, axis=UP), run_time=0.6)
"""
    errors = lint_manim_code(code, social_mode=False)
    assert any("axis=UP" in e or "mirror" in e.lower() for e in errors), errors


def test_codegen_prompt_reserves_subtitle_band():
    """The bottom of the frame is reserved for burned narration subtitles —
    manim content must stay above it (viewer: '자막 쓸 공간은 남기도록')."""
    from paper2video.renderers.manim_r import CODEGEN_SYSTEM
    assert "y=[-2.6" in CODEGEN_SYSTEM
    assert "subtitle" in CODEGEN_SYSTEM.lower() or "자막" in CODEGEN_SYSTEM


def test_subtitle_band_occupancy_detector(tmp_path):
    """Frame-level detector: content in y640-716 must be measurable so the
    renderer can retry scenes that invade the subtitle band."""
    import numpy as np
    from PIL import Image
    from paper2video.renderers.manim_r import subtitle_band_occupancy

    # 빈 프레임(배경만) — 0에 가깝게
    bg = Image.new("RGB", (1280, 720), (16, 16, 20))
    bg.save(tmp_path / "clean.png")
    assert subtitle_band_occupancy(tmp_path / "clean.png") < 0.01

    # 하단 밴드에 흰 사각형 — 높게
    dirty = bg.copy()
    from PIL import ImageDraw
    ImageDraw.Draw(dirty).rectangle((300, 660, 900, 710), fill=(240, 240, 240))
    dirty.save(tmp_path / "dirty.png")
    assert subtitle_band_occupancy(tmp_path / "dirty.png") > 0.02
