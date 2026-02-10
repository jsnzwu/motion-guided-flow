from __future__ import annotations

from typing import Any, Dict, List

from pydantic import Field

from wickit.config.components import (
    ConfigStruct,
    DatasetConfig as WickitDatasetConfig,
    JobConfig as WickitJobConfig,
    LearningRateConfig,
    LossConfig,
    ModelConfig as WickitModelConfig,
    OptimizerConfig,
    RunnerConfig as WickitRunnerConfig,
    RuntimeConfig,
    TaskConfig as WickitTaskConfig,
    TrainParameterConfig,
)
from wickit.config import CONFIGS
from wickit.logging.config import LoggingConfig


class FGDatasetConfig(WickitDatasetConfig):
    scale_config: Dict[str, Any] = Field(default_factory=dict)
    require_list: List[str] = Field(default_factory=list)
    demodulation_mode: str = ""
    part_size: int = 0
    future_config: Dict[str, Any] = Field(default_factory=dict)


class MFRRJobConfig(WickitJobConfig):
    export_path: str = ""
    num_thread: int = 0
    import_path: str = ""
    dataset_path: Dict[str, str] = Field(default_factory=dict)
    dataset_format: str = ""
    overwrite: bool = False
    scene: List[str] = Field(default_factory=list)
    test_config: Dict[str, Any] = Field(default_factory=dict)
    scene_info_name: str = ""
    pattern: str = ""
    num_history_frame: int = 0


class FGModelConfig(WickitModelConfig):
    export_onnx: bool = False
    config: Dict[str, Any] = Field(default_factory=dict)
    debug: List[str] = Field(default_factory=list)
    feature: List[str] = Field(default_factory=list)
    feature_config: Dict[str, Any] = Field(default_factory=dict)
    loss_config: Dict[str, Any] = Field(default_factory=dict)
    loss: List[str] = Field(default_factory=list)


class MFRRModelConfig(FGModelConfig):
    require_data: List[str] = Field(default_factory=list)
    input_buffer: List[str] = Field(default_factory=list)
    arch: str = ""
    gt_alias: str = ""
    method: str = ""
    tonemap_in_his_encoder: bool = False
    residual_item: str = ""
    st_color_names: List[str] = Field(default_factory=list)
    st_history_names: List[str] = Field(default_factory=list)
    pred_buffers: List[str] = Field(default_factory=list)
    gbuffer_encoder: Dict[str, Any] = Field(default_factory=dict)
    scene_color_encoder: Dict[str, Any] = Field(default_factory=dict)
    scene_color_encoder_no_st: Dict[str, Any] = Field(default_factory=dict)
    st_color_encoder: Dict[str, Any] = Field(default_factory=dict)
    shade_decoder__residual: Dict[str, Any] = Field(default_factory=dict)
    shade_decoder: Dict[str, Any] = Field(default_factory=dict)
    history_encoders: Dict[str, Any] = Field(default_factory=dict)
    history_no_st_encoders: Dict[str, Any] = Field(default_factory=dict)
    history_st_encoders: Dict[str, Any] = Field(default_factory=dict)
    scene_color_encoder_output_prefix: str = ""
    st_color_encoder_output_prefix: str = ""


class MFRRRunnerConfig(WickitRunnerConfig):
    recurrent_train_start: float = 0.0
    recurrent_train: Dict[str, Any] = Field(default_factory=dict)
    recurrent_test: Dict[str, Any] = Field(default_factory=dict)


@CONFIGS.register_module(name="MFRRTaskConfig")  # type: ignore[arg-type]
class MFRRTaskConfig(WickitTaskConfig):
    log_to_file: bool = False
    dataset: FGDatasetConfig = Field(default_factory=FGDatasetConfig)
    model: MFRRModelConfig = Field(default_factory=MFRRModelConfig)
    runner: MFRRRunnerConfig = Field(default_factory=MFRRRunnerConfig)
    job_config: MFRRJobConfig = Field(default_factory=MFRRJobConfig)
    vars: Dict[str, Any] = Field(default_factory=dict)
    write_path: str = ""
    exp_name: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MFRRTaskConfig:
        return cls.model_validate(data)


__all__ = [
    "ConfigStruct",
    "FGDatasetConfig",
    "FGModelConfig",
    "MFRRJobConfig",
    "MFRRModelConfig",
    "MFRRRunnerConfig",
    "MFRRTaskConfig",
    "LearningRateConfig",
    "LoggingConfig",
    "LossConfig",
    "OptimizerConfig",
    "RuntimeConfig",
    "TrainParameterConfig",
]
