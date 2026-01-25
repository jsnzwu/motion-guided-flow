import copy
from dataclasses import dataclass, field, fields
from typing import Any

from wickit.config.components import (
    DatasetConfig as WickitDatasetConfig,
    JobConfig as WickitJobConfig,
    LearningRateConfig,
    LossConfig,
    ModelConfig as WickitModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    TaskConfig as WickitTaskConfig,
    TrainParameterConfig,
    TrainerConfig as WickitTrainerConfig,
)
from wickit.logging.config import LoggingConfig

try:
    from yacs.config import CfgNode
except Exception:  # pragma: no cover - optional dependency
    CfgNode = None


@dataclass
class DatasetConfig(WickitDatasetConfig):
    _allow_new_keys = True


@dataclass
class ModelConfig(WickitModelConfig):
    _allow_new_keys = True


@dataclass
class TrainerConfig(WickitTrainerConfig):
    _allow_new_keys = True


@dataclass
class JobConfig(WickitJobConfig):
    _allow_new_keys = True


@dataclass
class TaskConfig(WickitTaskConfig):
    _allow_new_keys = True
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    job_config: JobConfig = field(default_factory=JobConfig)
    exp_name: str = ""


def _cfg_node_to_dict(cfg_node: Any) -> dict:
    if isinstance(cfg_node, list):
        return [_cfg_node_to_dict(item) for item in cfg_node]
    if isinstance(cfg_node, tuple):
        return tuple(_cfg_node_to_dict(item) for item in cfg_node)
    if CfgNode is None or not isinstance(cfg_node, CfgNode):
        return cfg_node
    cfg_dict = dict(cfg_node)
    for key, value in cfg_dict.items():
        cfg_dict[key] = _cfg_node_to_dict(value)
    return cfg_dict


def _filter_dataclass_fields(data: dict, cls: type) -> dict:
    field_names = {field_item.name for field_item in fields(cls)}
    return {key: value for key, value in data.items() if key in field_names}


def _normalize_type_field(data: dict) -> dict:
    normalized = dict(data)
    if not normalized.get("type") and normalized.get("class"):
        normalized["type"] = normalized["class"]
    normalized.pop("class", None)
    return normalized


def _normalize_cuda_visible_devices(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(item) for item in value if str(item).strip() != ""]
    if isinstance(value, str):
        cleaned = value.replace("\\,", ",").replace(" ", "")
        if not cleaned:
            return []
        return [int(item) for item in cleaned.split(",") if item]
    return [int(value)]


def _build_optimizer_config(optimizer_cfg: dict) -> dict:
    optimizer_cfg = _normalize_type_field(optimizer_cfg)
    if "betas" in optimizer_cfg and isinstance(optimizer_cfg["betas"], list):
        optimizer_cfg["betas"] = tuple(optimizer_cfg["betas"])
    return _filter_dataclass_fields(optimizer_cfg, OptimizerConfig)


def _build_lr_config(lr_cfg: dict, total_epoch: int) -> dict:
    lr_cfg = _normalize_type_field(lr_cfg)
    if "first_cycle_epochs" not in lr_cfg and "first_cycle_epoch" in lr_cfg:
        lr_cfg["first_cycle_epochs"] = lr_cfg["first_cycle_epoch"]
    if "initial_lr" not in lr_cfg:
        lr_cfg["initial_lr"] = lr_cfg.get("max_lr", LearningRateConfig.initial_lr)
    if "min_lr" not in lr_cfg and "max_lr" in lr_cfg:
        lr_cfg["min_lr"] = lr_cfg["max_lr"]
    if "total_epoch" not in lr_cfg:
        lr_cfg["total_epoch"] = total_epoch
    return _filter_dataclass_fields(lr_cfg, LearningRateConfig)


def _build_train_parameter_config(train_cfg: dict) -> dict:
    train_cfg = dict(train_cfg)
    optimizer_cfg = _build_optimizer_config(train_cfg.get("optimizer", {}))
    total_epoch = int(train_cfg.get("epoch", TrainParameterConfig.epoch))
    lr_cfg = _build_lr_config(train_cfg.get("lr_config", {}), total_epoch)
    train_cfg["optimizer"] = optimizer_cfg
    train_cfg["lr_config"] = lr_cfg
    return _filter_dataclass_fields(train_cfg, TrainParameterConfig)


def _build_trainer_config(config: dict) -> dict:
    trainer_cfg = _normalize_type_field(config.get("trainer", {}))
    if "num_gpu" not in trainer_cfg:
        trainer_cfg["num_gpu"] = int(config.get("num_gpu", trainer_cfg.get("num_gpu", 0)))
    return trainer_cfg


def _build_dataset_config(config: dict) -> dict:
    dataset_cfg = _normalize_type_field(config.get("dataset", {}))
    if "train_num_worker" not in dataset_cfg and "train_num_worker_sum" in dataset_cfg:
        dataset_cfg["train_num_worker"] = dataset_cfg["train_num_worker_sum"]
    return dataset_cfg


def _build_model_config(config: dict) -> dict:
    model_cfg = _normalize_type_field(config.get("model", {}))
    return model_cfg


def _build_loss_config(config: dict) -> dict:
    loss_cfg = _normalize_type_field(config.get("loss", {}))
    return _filter_dataclass_fields(loss_cfg, LossConfig)


def _build_runtime_config(config: dict) -> dict:
    runtime_cfg = dict(config.get("runtime", {}))
    num_gpu = int(config.get("num_gpu", runtime_cfg.get("num_gpu", 0)))
    if "cuda_visible_devices" in config and "cuda_visible_devices" not in runtime_cfg:
        runtime_cfg["cuda_visible_devices"] = config.get("cuda_visible_devices")
    runtime_cfg["cuda_visible_devices"] = _normalize_cuda_visible_devices(runtime_cfg.get("cuda_visible_devices"))
    if "use_gpu" in config:
        runtime_cfg["use_gpu"] = bool(config.get("use_gpu"))
    runtime_cfg.setdefault("use_gpu", num_gpu > 0)
    if "use_ddp" in config:
        runtime_cfg["use_ddp"] = bool(config.get("use_ddp"))
    runtime_cfg.setdefault("use_ddp", num_gpu > 1)
    runtime_cfg.setdefault("local_rank", int(config.get("local_rank", runtime_cfg.get("local_rank", 0))))
    runtime_cfg.setdefault("world_size", int(config.get("world_size", runtime_cfg.get("world_size", 1))))
    if "device" not in runtime_cfg:
        runtime_cfg["device"] = "cuda" if runtime_cfg.get("use_gpu", False) else "cpu"
    return _filter_dataclass_fields(runtime_cfg, RuntimeConfig)


def _build_logging_config(config: dict) -> dict:
    logging_cfg = dict(config.get("logging", {}))
    return _filter_dataclass_fields(logging_cfg, LoggingConfig)


def _build_job_config(config: dict) -> dict:
    job_cfg = dict(config.get("job_config", {}))
    return job_cfg


def dict_to_config(config: dict) -> TaskConfig:
    if not isinstance(config, dict):
        raise TypeError(f"config must be a dict, got {type(config)}")
    config = copy.deepcopy(config)
    train_param_cfg = _build_train_parameter_config(config.get("train_parameter", {}))
    task_payload = {
        "job_name": config.get("job_name", ""),
        "config_root_path": config.get("config_root_path", ""),
        "base": config.get("base"),
        "pipeline": config.get("pipeline", []),
        "include": config.get("include", []),
        "output_root_path": config.get("output_root_path", ""),
        "debug_data_flow": config.get("debug_data_flow", config.get("trainer", {}).get("debug_data_flow", False)),
        "runtime": _build_runtime_config(config),
        "trainer": _build_trainer_config(config),
        "dataset": _build_dataset_config(config),
        "model": _build_model_config(config),
        "train_parameter": train_param_cfg,
        "debug": config.get("debug", False),
        "clear_output_path": config.get("clear_output_path", False),
        "detect_anomaly": config.get("detect_anomaly", False),
        "initial_inference": config.get("initial_inference", True),
        "include_name": config.get("include_name"),
        "args": config.get("args", {}),
        "log": config.get("log", {}),
        "logging": _build_logging_config(config),
        "loss": _build_loss_config(config),
        "job_config": _build_job_config(config),
        "time_string": config.get("time_string", ""),
        "pre_model": config.get("pre_model"),
        "buffer_config": config.get("buffer_config", {}),
        "_input_config": config,
        "_trainer_config": config.get("_trainer_config", config),
        "tensorboard_info_step": config.get("tensorboard_info_step", {}),
        "tensorboard_info_epoch": config.get("tensorboard_info_epoch", {}),
        "bar_info_step": config.get("bar_info_step", {}),
        "bar_info_epoch": config.get("bar_info_epoch", {}),
        "text_info_step": config.get("text_info_step", {}),
        "text_info_epoch": config.get("text_info_epoch", {}),
        "avg_info_epoch": config.get("avg_info_epoch", {}),
    }
    task_cfg = TaskConfig.from_dict(_filter_dataclass_fields(task_payload, TaskConfig))
    if not task_cfg.exp_name:
        object.__setattr__(task_cfg, "exp_name", task_cfg.job_name)
    return task_cfg


class DictToDataclassAdapter:
    def __init__(self, config: dict) -> None:
        config_dict = _cfg_node_to_dict(config)
        if not isinstance(config_dict, dict):
            raise TypeError(f"config must be a dict, got {type(config_dict)}")
        self._config_dict = config_dict

    @property
    def config_dict(self) -> dict:
        return self._config_dict

    def to_task_config(self) -> TaskConfig:
        return dict_to_config(self._config_dict)

    def to_dataset_config(self) -> DatasetConfig:
        dataset_cfg = _build_dataset_config(self._config_dict)
        return DatasetConfig.from_dict(dataset_cfg)

    def to_trainer_config(self) -> TrainerConfig:
        trainer_cfg = _build_trainer_config(self._config_dict)
        return TrainerConfig.from_dict(trainer_cfg)

    def to_model_config(self) -> ModelConfig:
        model_cfg = _build_model_config(self._config_dict)
        return ModelConfig.from_dict(model_cfg)
