from utils.config_adapter import BufferConfig, TaskConfig, _cfg_node_to_dict, dict_to_config


def test_buffer_config_defaults() -> None:
    cfg = BufferConfig()
    assert cfg.max_luminance == 1.0
    assert cfg.min_luminance == 0.0
    assert cfg.scale_regex == {}
    assert cfg.augmented_data_recipe is None
    assert cfg.history_config is None


def test_cfg_node_to_dict_handles_structbase() -> None:
    raw = {
        "job_name": "adapter_test",
        "trainer": {"type": "DummyTrainer"},
        "dataset": {"type": "DummyDataset", "train_num_worker_sum": 1, "history_config": {"num": 1}},
        "model": {"type": "DummyModel"},
        "train_parameter": {"batch_size": 1},
    }
    config = dict_to_config(raw)
    converted = _cfg_node_to_dict(config)
    assert isinstance(converted, dict)
    assert converted["job_name"] == "adapter_test"
    assert isinstance(config, TaskConfig)
