from dataclasses import dataclass, field
from typing import Any, Dict, List

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
from wickit.config import CONFIGS
from wickit.logging.config import LoggingConfig


# ========== Dataset Configs ==========

@dataclass
class FGDatasetConfig(WickitDatasetConfig):
    scale_config: Dict[str, Any] = field(default_factory=dict)
    require_list: List[str] = field(default_factory=list)
    demodulation_mode: str = ""
    part_size: int = 0
    future_config: Dict[str, Any] = field(default_factory=dict)


# ========== Job Configs ==========

@dataclass
class MFRRJobConfig(WickitJobConfig):
    export_path: str = ""
    num_thread: int = 0
    import_path: str = ""
    dataset_path: Dict[str, str] = field(default_factory=dict)
    dataset_format: str = ""
    overwrite: bool = False
    scene: List[str] = field(default_factory=list)
    test_config: Dict[str, Any] = field(default_factory=dict)
    scene_info_name: str = ""
    pattern: str = ""
    num_history_frame: int = 0


# ========== Model Configs ==========

@dataclass
class FGModelConfig(WickitModelConfig):
    export_onnx: bool = False
    config: Dict[str, Any] = field(default_factory=dict)
    debug: List[str] = field(default_factory=list)
    feature: List[str] = field(default_factory=list)
    feature_config: Dict[str, Any] = field(default_factory=dict)
    loss_config: Dict[str, Any] = field(default_factory=dict)
    loss: List[str] = field(default_factory=list)


@dataclass
class MFRRModelConfig(FGModelConfig):
    require_data: List[str] = field(default_factory=list)
    input_buffer: List[str] = field(default_factory=list)
    arch: str = ""
    gt_alias: str = ""
    method: str = ""
    tonemap_in_his_encoder: bool = False
    residual_item: str = ""
    st_color_names: List[str] = field(default_factory=list)
    st_history_names: List[str] = field(default_factory=list)
    pred_buffers: List[str] = field(default_factory=list)
    gbuffer_encoder: Dict[str, Any] = field(default_factory=dict)
    scene_color_encoder: Dict[str, Any] = field(default_factory=dict)
    scene_color_encoder_no_st: Dict[str, Any] = field(default_factory=dict)
    st_color_encoder: Dict[str, Any] = field(default_factory=dict)
    shade_decoder__residual: Dict[str, Any] = field(default_factory=dict)
    shade_decoder: Dict[str, Any] = field(default_factory=dict)
    history_encoders: Dict[str, Any] = field(default_factory=dict)
    history_no_st_encoders: Dict[str, Any] = field(default_factory=dict)
    history_st_encoders: Dict[str, Any] = field(default_factory=dict)
    scene_color_encoder_output_prefix: str = ""
    st_color_encoder_output_prefix: str = ""


# ========== Trainer Configs ==========

@dataclass
class MFRRTrainerConfig(WickitTrainerConfig):
    recurrent_train_start: float = 0.0
    recurrent_train: Dict[str, Any] = field(default_factory=dict)
    recurrent_test: Dict[str, Any] = field(default_factory=dict)


# ========== Task Configs (registered to CONFIGS) ==========

@CONFIGS.register_module(name="MFRRTaskConfig")
@dataclass
class MFRRTaskConfig(WickitTaskConfig):
    log_to_file: bool = False
    dataset: FGDatasetConfig = field(default_factory=FGDatasetConfig)
    model: MFRRModelConfig = field(default_factory=MFRRModelConfig)
    trainer: MFRRTrainerConfig = field(default_factory=MFRRTrainerConfig)
    job_config: MFRRJobConfig = field(default_factory=MFRRJobConfig)
    vars: Dict[str, Any] = field(default_factory=dict)
    write_path: str = ""
    exp_name: str = ""


__all__ = [
    "FGDatasetConfig",
    "FGModelConfig",
    "MFRRJobConfig",
    "MFRRModelConfig",
    "MFRRTrainerConfig",
    "MFRRTaskConfig",
    "LearningRateConfig",
    "LoggingConfig",
    "LossConfig",
    "OptimizerConfig",
    "RuntimeConfig",
    "TrainParameterConfig",
]
