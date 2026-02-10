from config_defines import MFRRTaskConfig
from utils.config_enhancer import enhance_train_config, update_config


def _minimal_buffer_config() -> dict:
    return {
        "scale_regex": {
            "ds_scale": {
                "pattern": "%ds",
                "target": "ds{}",
                "value": 1,
                "enable": False,
            }
        },
        "augmented_data_recipe": {
            "augmented_data_recipe__demodulate_template": [],
            "augmented_data_recipe__history_template": [],
            "augmented_data_recipe__future_template": [],
            "augmented_data_recipe__history_warped_template": [],
            "augmented_data_recipe__future_num": 0,
            "augmented_data_recipe": {
                "merged_motion_vector": {"num_history": 0},
            },
            "data_attribute": {
                "scene_color": {"type": "image", "channel": 3},
            },
        },
    }


def _minimal_config_dict() -> dict:
    return {
        "entry": "MFRRTaskConfig",
        "job_name": "enhancer_test",
        "runner": {"entry": "DummyRunner", "num_gpu": 2, "wait_to_start": "0"},
        "dataset": {
            "entry": "DummyDataset",
            "train_num_worker_sum": 4,
            "train_num_worker": 4,
            "history_config": {"num": 1},
            "scale_config": {},
            "augment_loader": True,
        },
        "model": {"entry": "DummyModel", "require_data": ["scene_color"]},
        "train_parameter": {"batch_size": 4},
        "job_config": {"export_path": "/tmp"},
        "buffer_config": _minimal_buffer_config(),
    }


def test_update_config_sets_runtime_fields() -> None:
    config = MFRRTaskConfig.from_dict(_minimal_config_dict())
    updated = update_config(config)
    assert updated.runtime.use_gpu is True
    assert updated.runtime.use_ddp is True
    assert updated.train_parameter.batch_size == 2
    assert updated.dataset.train_num_worker == 2


def test_enhance_train_config_updates_dataset_and_buffer() -> None:
    config = MFRRTaskConfig.from_dict(_minimal_config_dict())
    updated = update_config(config)
    enhanced = enhance_train_config(updated)
    assert enhanced.dataset.require_list == ["scene_color"]
    assert enhanced.buffer_config["history_config"] == enhanced.dataset.history_config
    assert enhanced.dataset.path == "/tmp"
