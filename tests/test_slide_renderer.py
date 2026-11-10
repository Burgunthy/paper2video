from PIL import Image
from paper2video.types import Scene
from paper2video.renderers.slide import render_slide


def test_render_slide_produces_png_at_target_resolution(tmp_path):
    scene = Scene(
        id=1,
        narration="Hello.",
        visual_type="slide",
        visual_spec={"title": "Test Title", "bullets": ["alpha", "beta", "gamma"]},
        duration_hint_sec=4.0,
    )
    out = render_slide(scene, tmp_path, size=(1920, 1080))
    assert out.exists()
    assert out.suffix == ".png"
    with Image.open(out) as img:
        assert img.size == (1920, 1080)


def test_render_slide_handles_missing_bullets(tmp_path):
    scene = Scene(
        id=2, narration="X", visual_type="slide",
        visual_spec={"title": "Only Title"}, duration_hint_sec=3.0,
    )
    out = render_slide(scene, tmp_path)
    assert out.exists()


def test_render_slide_handles_caption(tmp_path):
    scene = Scene(
        id=3, narration="X", visual_type="slide",
        visual_spec={"title": "T", "bullets": ["b"], "caption": "a caption"},
        duration_hint_sec=3.0,
    )
    out = render_slide(scene, tmp_path)
    assert out.exists()


def test_render_slide_theme_seed_produces_different_images(tmp_path):
    scene = Scene(
        id=1, narration="X", visual_type="slide",
        visual_spec={"title": "Test", "bullets": ["a"]},
        duration_hint_sec=3.0,
    )
    out_a = render_slide(scene, tmp_path / "a", theme_seed="openai_blog:abc123")
    out_b = render_slide(scene, tmp_path / "b", theme_seed="arxiv:xyz789")
    assert out_a.exists() and out_b.exists()
    # Different seeds should produce different pixel content
    assert out_a.read_bytes() != out_b.read_bytes()


def test_render_slide_same_seed_is_deterministic(tmp_path):
    scene = Scene(
        id=1, narration="X", visual_type="slide",
        visual_spec={"title": "Test", "bullets": ["a"]},
        duration_hint_sec=3.0,
    )
    out_a = render_slide(scene, tmp_path / "a", theme_seed="same_seed")
    out_b = render_slide(scene, tmp_path / "b", theme_seed="same_seed")
    assert out_a.read_bytes() == out_b.read_bytes()


def test_landscape_slide_renders_body_text(tmp_path):
    """Landscape slides must draw the direction/body like portrait does —
    a title-only slide reads as 'broken' to viewers (empty black frame)."""
    import numpy as np
    from PIL import Image
    from paper2video.types import Scene
    from paper2video.renderers.slide import render_slide

    scene = Scene(
        id=1, narration="n", visual_type="slide",
        visual_spec={"title": "제목", "direction": "본문 설명 텍스트가 여기 들어간다",
                     "caption": "캡션"},
        duration_hint_sec=5.0,
    )
    png = render_slide(scene, tmp_path, size=(1280, 720))
    im = np.array(Image.open(png).convert("RGB"))
    bg = im[0, 0]
    nb = (np.abs(im.astype(int) - bg.astype(int)).sum(axis=2) > 40)
    # body zone (between title ~y<250 and caption ~y>620) must have content
    assert nb[300:580].mean() > 0.005, "landscape slide rendered no body text"


def test_slide_korean_font_and_clear_band(tmp_path):
    """Slides must (a) render Hangul with a CJK font (not DejaVu tofu boxes)
    and (b) keep the bottom subtitle band (y>=650) free — the old layout put
    the caption and scene number inside it."""
    import numpy as np
    from PIL import Image
    from paper2video.types import Scene
    from paper2video.renderers.slide import render_slide

    scene = Scene(id=7, narration="한글 내레이션", visual_type="slide",
                  visual_spec={"title": "한글 제목 테스트",
                               "direction": "본문 방향 설명 한글 문장",
                               "caption": "하단 캡션"},
                  duration_hint_sec=5.0)
    png = render_slide(scene, tmp_path, size=(1280, 720))
    a = np.array(Image.open(png).convert("RGB")).astype(int)
    band = a[650:718, 150:1130]
    ref = band[0, 0]
    occ = float((np.abs(band - ref).sum(axis=2) > 40).mean())
    assert occ < 0.02, f"slide content in subtitle band: {occ:.3f}"


def test_slide_many_long_bullets_stay_clear_of_band(tmp_path):
    """Long bullets wrap and used to spill into the subtitle band. The layout
    must measure text heights and shrink the font until everything fits above
    the band."""
    import numpy as np
    from PIL import Image
    from paper2video.types import Scene
    from paper2video.renderers.slide import render_slide

    scene = Scene(id=9, narration="n", visual_type="slide",
                  visual_spec={
                      "title": "아주 긴 제목이 들어가는 경우의 슬라이드 테스트",
                      "bullets": [
                          "첫 번째 불릿인데 내용이 매우 길어서 두 줄 이상으로 줄바꿈이 일어나는 경우를 테스트합니다",
                          "두 번째 불릿도 길게 작성하여 여러 줄로 감싸지는 것을 확인하기 위한 문장입니다",
                          "세 번째 불릿 — 역시 길게 작성해서 전체 높이가 슬라이드를 넘으려는 상황을 만듭니다",
                          "네 번째 불릿 마지막 문장도 충분히 깁니다",
                          "다섯 번째",
                          "여섯 번째 불릿",
                      ],
                      "caption": "캡션"},
                  duration_hint_sec=5.0)
    png = render_slide(scene, tmp_path, size=(1280, 720))
    a = np.array(Image.open(png).convert('RGB')).astype(int)
    band = a[650:716, 150:1130]
    occ = float((np.abs(band - band[0, 0]).sum(axis=2) > 40).mean())
    assert occ < 0.02, f"long bullets spilled into band: {occ:.3f}"
    # 본문은 존재해야 함 (폰트를 과도하게 줄여 없애버리면 안 됨)
    body = a[200:600, 100:1180]
    assert (np.abs(body - body[0, 0]).sum(axis=2) > 40).mean() > 0.005


def test_slide_long_direction_stays_clear_of_band(tmp_path):
    """Manim-fallback specs carry a LONG visual direction — anchored at 0.62
    with wrap it spills hundreds of pixels into the subtitle band."""
    import numpy as np
    from PIL import Image
    from paper2video.types import Scene
    from paper2video.renderers.slide import render_slide

    scene = Scene(id=2, narration="n", visual_type="slide",
                  visual_spec={
                      "title": "카메라 좌표와 월드 좌표",
                      "direction": ("머리에 카메라를 단 작업자 아이콘이 등장하고 왼쪽에서 손 모양이 "
                                    "나타나 카메라 좌표계 안에 갇혀 있는 모습을 보여준다. 이어서 월드 "
                                    "좌표계 격자가 펼쳐지며 손이 그 위로 옮겨지는 변환을 화살표로 "
                                    "보여주고, 오른쪽에는 두 방법의 비교 표가 위에서부터 차례로 "
                                    "내려오면서 각 셀의 숫자가 카운트업된다. 마지막에 두 축의 스케일 "
                                    "차이를 강조하는 확대 애니메이션으로 마무리한다. 이 문장은 실제 "
                                    "manim direction 필드처럼 매우 길게 작성되었다."),
                      "caption": "캡션"},
                  duration_hint_sec=5.0)
    png = render_slide(scene, tmp_path, size=(1280, 720))
    a = np.array(Image.open(png).convert('RGB')).astype(int)
    band = a[650:716, 150:1130]
    occ = float((np.abs(band - band[0, 0]).sum(axis=2) > 40).mean())
    assert occ < 0.02, f"long direction spilled into band: {occ:.3f}"
