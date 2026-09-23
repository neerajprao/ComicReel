"""ComicReel command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from comicreel import stages  # noqa: F401  (registers stage backends)
from comicreel.config import PipelineConfig, registered_backends
from comicreel.pipeline.factory import build_stage
from comicreel.pipeline.orchestrator import Orchestrator

app = typer.Typer(help="ComicReel: turn scanned comics into animated motion pictures.")

DEFAULT_CONFIG = Path("configs/pipeline.yaml")
DEFAULT_CACHE_DIR = Path("data/cache")
PAGE_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def _resolve_page_image(comic_id: str, page_id: str) -> Path:
    """Find a page's raw scan under data/raw_scans/<comic_id>/<page_id>.<ext>."""
    raw_dir = Path("data/raw_scans") / comic_id
    for ext in PAGE_IMAGE_EXTENSIONS:
        candidate = raw_dir / f"{page_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No page image found for {comic_id}/{page_id} in {raw_dir} "
        f"(tried extensions: {', '.join(PAGE_IMAGE_EXTENSIONS)})"
    )


@app.command()
def run(
    input_path: Path = typer.Argument(
        ..., help="Path to a comic PDF or a directory of page images."
    ),
    config: Path = typer.Option(DEFAULT_CONFIG, help="Path to pipeline.yaml."),
    comic_id: str = typer.Option(
        None, help="Identifier for this comic (defaults to the input filename stem)."
    ),
) -> None:
    """Run the full pipeline end-to-end on a comic."""
    typer.echo(f"[stub] Would run full pipeline on {input_path} using config {config}")


@app.command("run-stage")
def run_stage(
    stage: str = typer.Argument(..., help="Stage name, e.g. panel_detection."),
    comic_id: str = typer.Argument(...),
    page_id: str = typer.Argument(...),
    config: Path = typer.Option(DEFAULT_CONFIG),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, help="Where stage artifacts are cached."),
    force: bool = typer.Option(False, help="Recompute even if a cached result exists."),
) -> None:
    """Run a single stage for one page."""
    pipeline_config = PipelineConfig.from_yaml(config)
    image_path = _resolve_page_image(comic_id, page_id)
    stage_obj = build_stage(stage, pipeline_config)
    orchestrator = Orchestrator(pipeline_config, cache_dir, {stage: stage_obj})
    artifacts = orchestrator.run_page(
        comic_id, page_id, image_path, from_stage=stage, to_stage=stage, force=force
    )
    typer.echo(json.dumps(artifacts[stage], indent=2))


@app.command()
def resume(
    comic_id: str = typer.Argument(...),
    config: Path = typer.Option(DEFAULT_CONFIG),
) -> None:
    """Resume a partially completed run using cached artifacts."""
    typer.echo(f"[stub] Would resume {comic_id}")


@app.command()
def inspect(
    artifact_path: Path = typer.Argument(..., help="Path to a cached stage artifact JSON."),
) -> None:
    """Pretty-print a cached artifact for debugging."""
    if not artifact_path.exists():
        typer.echo(f"No such artifact: {artifact_path}", err=True)
        raise typer.Exit(1)
    typer.echo(artifact_path.read_text())


@app.command()
def doctor(
    config: Path = typer.Option(DEFAULT_CONFIG),
) -> None:
    """Check config validity and print resolved device/backends per stage."""
    if not config.exists():
        typer.echo(f"Config not found: {config}", err=True)
        raise typer.Exit(1)
    pipeline_config = PipelineConfig.from_yaml(config)
    for stage_name, stage_cfg in pipeline_config.stages.items():
        device = pipeline_config.resolve_device(stage_name)
        available = registered_backends(stage_name)
        typer.echo(
            f"{stage_name}: backend={stage_cfg.backend} device={device} "
            f"available={available or '(none registered yet)'}"
        )


if __name__ == "__main__":
    app()
