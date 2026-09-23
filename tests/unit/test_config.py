from comicreel.config import (
    PipelineConfig,
    get_backend_class,
    register_backend,
    registered_backends,
)


def test_register_and_get_backend():
    @register_backend("dummy_stage", "dummy_backend")
    class DummyBackend:
        pass

    assert get_backend_class("dummy_stage", "dummy_backend") is DummyBackend
    assert "dummy_backend" in registered_backends("dummy_stage")


def test_pipeline_config_from_yaml(tmp_path):
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        """
stages:
  panel_detection:
    backend: cv_heuristic
    device: cpu
"""
    )
    config = PipelineConfig.from_yaml(config_path)
    assert config.stages["panel_detection"].backend == "cv_heuristic"
    assert config.resolve_device("panel_detection") == "cpu"
