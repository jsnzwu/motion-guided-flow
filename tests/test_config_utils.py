from pathlib import Path

from config import (
    MFRRTaskConfig,
    load_yaml_with_replacements,
    parse_config,
    parse_config_to_dict,
)
from wickit.utils.basic.dict import deep_update


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_create_config_loads_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "simple.yaml"
    _write_yaml(cfg_path, "foo: bar\nnum: 3\n")
    data = load_yaml_with_replacements(str(cfg_path))
    assert data["foo"] == "bar"
    assert data["num"] == 3


def test_merge_from_another_deep_update() -> None:
    target = {"a": 1, "b": {"c": 2}}
    source = {"b": {"d": 3}, "e": 4}
    deep_update(target, source)
    assert target == {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}


def test_parse_config_to_dict_with_base_and_pipeline(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    pipe = tmp_path / "pipe.yaml"
    child = tmp_path / "child.yaml"
    _write_yaml(base, "value: 1\nnested:\n  base_only: true\n")
    _write_yaml(pipe, "value: 2\nnested:\n  piped: true\n")
    _write_yaml(child, "base: base.yaml\npipeline: [pipe.yaml]\nvalue: 3\n")
    data = parse_config_to_dict("child.yaml", root_path=str(tmp_path) + "/")
    assert data["value"] == 3
    assert data["nested"]["base_only"] is True
    assert data["nested"]["piped"] is True


def test_parse_config_returns_task_config(tmp_path: Path) -> None:
    cfg = tmp_path / "task.yaml"
    _write_yaml(
        cfg,
        "\n".join(
            [
                "job_name: unit_test",
                "entry: MFRRTaskConfig",
                "runner:",
                "  entry: DummyRunner",
                "  num_gpu: 1",
                "dataset:",
                "  entry: DummyDataset",
                "  train_num_worker_sum: 2",
                "  history_config:",
                "    num: 1",
                "model:",
                "  entry: DummyModel",
                "  model_name: dummy",
                "  inference_precision: fp32",
                "  dummy_input_size_h: 1",
                "  dummy_input_size_w: 1",
                "train_parameter:",
                "  batch_size: 2",
                "job_config:",
                "  export_path: /tmp",
                "buffer_config:",
                "  scale_regex:",
                "    ds_scale:",
                "      pattern: \"%ds\"",
                "      target: \"ds{}\"",
                "      value: 1",
                "      enable: false",
                "  augmented_data_recipe__demodulate_template: []",
                "  augmented_data_recipe__history_template: []",
                "  augmented_data_recipe__future_template: []",
                "  augmented_data_recipe__history_warped_template: []",
                "  augmented_data_recipe__future_num: 0",
                "  augmented_data_recipe:",
                "    merged_motion_vector:",
                "      num_history: 0",
                "  data_attribute:",
                "    scene_color:",
                "      type: image",
                "      channel: 3",
            ]
        ),
    )
    config = parse_config(str(cfg), root_path="")
    assert isinstance(config, MFRRTaskConfig)
    assert config.job_name == "unit_test"
