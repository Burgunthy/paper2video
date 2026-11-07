from __future__ import annotations
import json
import os
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

from .ingest import extract_from_url, extract_from_file, IngestedDoc
from .research import research, ResearchResult
from .script import write_script, VARIANTS
from .figures import extract_figures
from .renderers.slide import render_slide
from .renderers.manim_r import render_manim_scene, ManimRenderError
from .renderers.figure_r import render_figure_scene, FigureRenderError
from .tts import TTSEngine, synthesize_scene_audio, SceneAudio
from .assemble import mux_scene_clip, concat_clips, build_scene_clip_from_image
from .llm import LLMClient
from .types import ScriptDoc, Scene


@dataclass
class PipelineConfig:
    out_dir: Path = Path("out")
    run_id: str = ""
    enable_search: bool = True
    width: int = 1920
    height: int = 1080
    fps: int = 30
    manim_quality: str = "m"  # l=480p15, m=720p30, h=1080p60
    manim_max_retries: int = 3
    use_manim: bool = True


@dataclass
class PipelineResult:
    run_dir: Path
    video_path: Path
    script: ScriptDoc


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False), encoding="utf-8")


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _log(run_dir: Path, msg: str) -> None:
    print(f"[pipeline] {msg}")
    (run_dir / "pipeline.log").open("a", encoding="utf-8").write(msg + "\n")


def _render_scene_visual(
    scene: Scene,
    duration_sec: float,
    run_dir: Path,
    llm: LLMClient,
    cfg: PipelineConfig,
    theme_seed: str | None = None,
    fig_dir: Path | None = None,
) -> tuple[Path, str]:
    """Render one scene's visual track. Returns (mp4_or_png_path, kind) where kind is 'video' or 'image'.

    theme_seed: deterministic seed for slide color/layout variation (e.g. source_id).
    """
    if scene.visual_type == "figure":
        try:
            mp4 = render_figure_scene(
                scene=scene,
                duration_sec=duration_sec,
                fig_dir=fig_dir or (run_dir / "figures"),
                out_dir=run_dir / "figure_out",
                width=cfg.width,
                height=cfg.height,
                fps=cfg.fps,
            )
            _log(run_dir, f"  scene {scene.id}: figure OK -> {mp4.name}")
            return mp4, "video"
        except (FigureRenderError, Exception) as e:
            _log(run_dir, f"  scene {scene.id}: figure FAILED, falling back to slide. err: {e}")
            img_dir = run_dir / "images"
            png = render_slide(scene, img_dir, size=(cfg.width, cfg.height), theme_seed=theme_seed)
            return png, "image"

    if cfg.use_manim:
        try:
            manim_dir = run_dir / "manim"
            # Always render Manim at landscape. For portrait social videos,
            # the social pipeline reframes the output to 9:16 after render.
            mp4 = render_manim_scene(
                scene=scene,
                duration_sec=duration_sec,
                out_dir=manim_dir,
                llm=llm,
                quality=cfg.manim_quality,
                max_retries=cfg.manim_max_retries,
                resolution=None,
            )
            _log(run_dir, f"  scene {scene.id}: manim OK -> {mp4.name}")
            return mp4, "video"
        except ManimRenderError as e:
            _log(run_dir, f"  scene {scene.id}: manim FAILED after retries, falling back to slide. err: {e.message}")
            (run_dir / "manim" / f"scene_{scene.id:03d}.err.txt").write_text(
                e.stderr + "\n\n---LAST CODE---\n" + e.last_code, encoding="utf-8"
            )
        except Exception as e:
            _log(run_dir, f"  scene {scene.id}: manim crashed: {e}\n{traceback.format_exc()}")

    # Fallback: matplotlib slide
    img_dir = run_dir / "images"
    png = render_slide(scene, img_dir, size=(cfg.width, cfg.height), theme_seed=theme_seed)
    return png, "image"


def run_pipeline(
    source: str,
    config: PipelineConfig,
    llm: LLMClient,
    tts: TTSEngine,
    codegen_llm: LLMClient | None = None,
) -> PipelineResult:
    # codegen_llm: optional separate client for manim codegen (see cli.py).
    cg_llm = codegen_llm or llm
    run_id = config.run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _log(run_dir, f"=== run_id={run_id} ===")

    # 1-3. Ingest / Figures / Research / Script — cached as script.json for resumability
    script_cache = run_dir / "script.json"
    if script_cache.exists():
        _log(run_dir, "stage: script (CACHED — reusing script.json)")
        script_doc = ScriptDoc.model_validate_json(script_cache.read_text(encoding="utf-8"))
        _log(run_dir, f"  title={script_doc.title!r} scenes={len(script_doc.scenes)}")
    else:
        _log(run_dir, "stage: ingest")
        pdf_path: Path | None = None
        if _is_url(source):
            pdf_path = run_dir / "paper.pdf"
            doc: IngestedDoc = extract_from_url(source, keep_pdf=pdf_path)
            if not pdf_path.exists():
                pdf_path = None  # HTML source — no PDF to mine
        else:
            src = Path(source)
            doc = extract_from_file(src)
            if src.suffix.lower() == ".pdf":
                pdf_path = src
        _write_json(run_dir / "ingest.json", asdict(doc))
        _log(run_dir, f"  title={doc.title!r} chars={len(doc.text)}")

        # Figures are extracted BEFORE the script so the writer can reference them.
        variant_name = os.environ.get("PAPER2VIDEO_VARIANT", "classic")
        figure_manifest: list[dict] = []
        if variant_name in VARIANTS and VARIANTS[variant_name].get("figures") and pdf_path:
            try:
                _log(run_dir, "stage: figures")
                figs = extract_figures(pdf_path, run_dir / "figures")
                figure_manifest = [
                    {"id": f.id, "page": f.page, "caption": f.caption,
                     "width_px": f.width_px, "height_px": f.height_px}
                    for f in figs
                ]
                _log(run_dir, f"  extracted {len(figs)} figures: {[f.id for f in figs]}")
            except Exception as e:
                _log(run_dir, f"  figure extraction failed (continuing without): {e}")

        _log(run_dir, "stage: research")
        research_result: ResearchResult = research(doc, llm=llm, enable_search=config.enable_search)
        _write_json(run_dir / "research.json", {
            "concepts": research_result.concepts,
            "notes": [n.model_dump() for n in research_result.notes],
            "key_points": research_result.key_points,
        })
        _log(run_dir, f"  concepts={research_result.concepts}")

        _log(run_dir, "stage: script")
        script_doc = write_script(research_result, llm=llm, figure_manifest=figure_manifest)
        _write_json(script_cache, script_doc.model_dump())
        _log(run_dir, f"  title={script_doc.title!r} scenes={len(script_doc.scenes)}")

    # 4. TTS FIRST — so we know per-scene durations before rendering visuals.
    _log(run_dir, "stage: tts")
    audio_dir = run_dir / "audio"
    scene_audios: list[SceneAudio] = [
        synthesize_scene_audio(s, tts, audio_dir) for s in script_doc.scenes
    ]
    for s, a in zip(script_doc.scenes, scene_audios):
        _log(run_dir, f"  scene {s.id}: audio {a.duration_sec:.2f}s")

    # 5. Render visuals per scene (figure scenes first-class, Manim primary, slide fallback)
    _log(run_dir, "stage: render visuals")
    visual_tracks: list[tuple[Path, str]] = []
    for s, a in zip(script_doc.scenes, scene_audios):
        path_kind = _render_scene_visual(s, a.duration_sec, run_dir, cg_llm, config)
        visual_tracks.append(path_kind)

    # 6. Mux audio + video per scene (with pad/trim to audio duration), then concat
    _log(run_dir, "stage: mux + concat")
    work_dir = run_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    scene_clips: list[Path] = []
    for s, a, (visual_path, kind) in zip(script_doc.scenes, scene_audios, visual_tracks):
        clip_out = work_dir / f"scene_{s.id:03d}.mp4"
        if kind == "video":
            mux_scene_clip(
                video_path=visual_path,
                audio_path=a.audio_path,
                duration_sec=a.duration_sec,
                out_path=clip_out,
                width=config.width,
                height=config.height,
                fps=config.fps,
            )
        else:
            # Image: loop image for duration + attach audio
            build_scene_clip_from_image(
                image_path=visual_path,
                audio_path=a.audio_path,
                duration_sec=a.duration_sec,
                out_path=clip_out,
                width=config.width,
                height=config.height,
                fps=config.fps,
            )
        scene_clips.append(clip_out)

    video_path = run_dir / "video.mp4"
    concat_clips(scene_clips, video_path, work_dir)

    # 내레이션 대사 자막 번닝 — 하단 배치(피규어 캡션 스트립과 무겹침)
    try:
        from .captions import build_srt, burn_subtitles

        srt_path = run_dir / "narration.srt"
        srt_path.write_text(
            build_srt(script_doc.scenes, [a.duration_sec for a in scene_audios]),
            encoding="utf-8",
        )
        burned = run_dir / "video_subtitled.mp4"
        burn_subtitles(video_path, srt_path, burned)
        burned.replace(video_path)
        _log(run_dir, "stage: narration subtitles burned")
    except Exception as e:
        _log(run_dir, f"subtitle burn failed (non-fatal): {e}")

    _log(run_dir, f"DONE: {video_path}")

    return PipelineResult(run_dir=run_dir, video_path=video_path, script=script_doc)
