from __future__ import annotations

from pathlib import Path

import pytest

from wickit.config import load_task_config


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_entry_scripts_use_load_task_config() -> None:
    files = [
        "external/wickit/wickit/runtime/launcher.py",
        "src/test/test_train.py",
        "src/test/test_trainer.py",
        "src/test/test_inference.py",
    ]
    for file_path in files:
        text = _read_text(file_path)
        assert "load_task_config(" in text


def test_entry_scripts_do_not_use_legacy_parse_apis() -> None:
    files = [
        "external/wickit/wickit/runtime/launcher.py",
        "src/test/test_train.py",
        "src/test/test_trainer.py",
        "src/test/test_inference.py",
    ]
    forbidden = ("parse_config(", "parse_config_to_dict(", "load_yaml_with_replacements(")
    for file_path in files:
        text = _read_text(file_path)
        for token in forbidden:
            assert token not in text


def test_entry_scripts_do_not_use_struct_mutation_apis() -> None:
    files = [
        "external/wickit/wickit/runtime/launcher.py",
        "src/test/test_train.py",
        "src/test/test_trainer.py",
        "src/test/test_inference.py",
    ]
    forbidden = ("unfreeze(", "freeze(", ".merge(")
    for file_path in files:
        text = _read_text(file_path)
        for token in forbidden:
            assert token not in text


def test_loader_has_no_runtime_domain_imports() -> None:
    text = _read_text("external/wickit/wickit/config/pyconfig_loader.py")
    forbidden_prefixes = (
        "wickit.models",
        "wickit.datasets",
        "wickit.runner",
        "wickit.losses",
        "wickit.lr_schedulers",
        "wickit.optimizers",
        "wickit.logging",
    )
    for prefix in forbidden_prefixes:
        assert prefix not in text


def test_loader_has_no_registry_build_calls() -> None:
    text = _read_text("external/wickit/wickit/config/pyconfig_loader.py")
    assert "Registry" not in text
    assert ".build(" not in text


def test_inference_uses_single_loader_call_site() -> None:
    text = _read_text("src/test/test_inference.py")
    assert text.count("load_task_config(") >= 2
    assert "config/dataset/infer_dataset_v6_moflow_ess" in text


def test_load_task_config_accepts_canonical_spec() -> None:
    cfg = load_task_config("config/entry/single_moflow_debug")
    assert cfg.entry == "MFRRTaskConfig"


def test_load_task_config_accepts_py_spec() -> None:
    cfg = load_task_config("config/entry/single_moflow_debug.py")
    assert cfg.entry == "MFRRTaskConfig"


def test_runtime_entry_rejects_yaml_and_yml_spec() -> None:
    with pytest.raises(Exception):
        _ = load_task_config("config/entry/single_moflow_debug.yaml")
    with pytest.raises(Exception):
        _ = load_task_config("config/entry/single_moflow_debug.yml")
