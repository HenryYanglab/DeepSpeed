# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Configuration models for the FastOffload observer."""

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FastOffloadMode(str, Enum):
    """Supported FastOffload execution modes."""

    observe = "observe"
    sync_offload = "sync_offload"
    async_offload = "async_offload"


class TelemetryRankMode(str, Enum):
    """Controls which ranks emit local telemetry reports."""

    rank0_local = "rank0_local"
    per_rank = "per_rank"


class ObserverFailurePolicy(str, Enum):
    """Controls how an observer callback failure affects training."""

    raise_error = "raise"
    disable_observer = "disable_observer"
    warn = "warn"


class FastOffloadConfigModel(BaseModel):
    """Strict and immutable base model for FastOffload configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class PolicyType(str, Enum):
    """Gradient policies available in the current implementation."""

    all_offload = "all_offload"


class SchedulerType(str, Enum):
    """Scheduler implementations available in the current implementation."""

    synchronous = "synchronous"
    overlap = "overlap"


class WorkerType(str, Enum):
    """CPU worker implementations available in the current implementation."""

    inline = "inline"


class PolicyConfig(FastOffloadConfigModel):
    """Policy settings for the synchronous baseline."""

    type: PolicyType = PolicyType.all_offload


class SchedulerConfig(FastOffloadConfigModel):
    """Scheduler concurrency and bounded-queue settings."""

    type: SchedulerType = SchedulerType.synchronous
    max_inflight_tasks: int = Field(default=1, ge=1)
    max_inflight_bytes: int = Field(default=536870912, ge=1)


class TransferConfig(FastOffloadConfigModel):
    """Gradient D2H and optional staging-buffer settings."""

    pin_memory: Literal[True] = True
    cpu_staging: bool = False
    async_strategy: Literal["producer_stream", "dedicated_stream"] = "producer_stream"
    buffer_count: int = Field(default=1, ge=1)
    buffer_size: int = Field(default=134217728, ge=1)


class WorkerConfig(FastOffloadConfigModel):
    """Worker settings for the synchronous baseline."""

    type: WorkerType = WorkerType.inline


class ImportanceConfig(FastOffloadConfigModel):
    """Streaming column-importance selection settings."""

    enabled: bool = False
    algorithm: str = Field(default="pretrained_delta_topk", min_length=1)
    warmup_steps: int = Field(default=10, ge=1)
    topk_ratio: float = Field(default=0.1, gt=0.0, le=0.5)
    comparison_chunk_rows: int = Field(default=4096, ge=1)
    sparse_backward: bool = False
    sparse_backward_min_tokens: int = Field(default=1024, ge=1)
    sparse_backward_min_output_features: int = Field(default=1024, ge=0)
    output_path: Optional[str] = None


class HybridUpdateConfig(FastOffloadConfigModel):
    """Hybrid sparse/dense update scheduling settings."""

    enabled: bool = False
    update_interval: int = Field(default=8, ge=2)
    accumulation_device: Literal["cpu", "gpu"] = "cpu"
    second_reduce_scatter_device: Literal["cpu", "gpu"] = "gpu"
    dense_boundary_enabled: Literal[True] = True
    double_buffer: Literal[True] = True
    second_gradient_reduction: Literal["mean", "sum"] = "mean"
    max_async_lag: int = Field(default=2, ge=1)
    pt_reserved_cores_perc: float = Field(default=0.25, gt=0.0, lt=1.0)
    overdue_policy: Literal["wait"] = "wait"
    compressed_collective_shadow: bool = False
    zero2_takeover: bool = False
    compressed_bucket_bytes: int = Field(default=134217728, ge=1)
    parity_atol: float = Field(default=1e-3, ge=0.0)
    parity_rtol: float = Field(default=1e-3, ge=0.0)


class TelemetryConfig(FastOffloadConfigModel):
    """Controls metadata collection and reporting for observer mode."""

    log_interval: int = Field(default=10, ge=1)
    parameter_events: bool = True
    bucket_events: bool = True
    host_timing: bool = True
    device_timing: bool = False
    memory_metrics: bool = True
    distributed_summary: bool = False
    rank_mode: TelemetryRankMode = TelemetryRankMode.rank0_local
    debug_event_buffer_size: int = Field(default=0, ge=0)
    max_parameter_name_length: int = Field(default=128, ge=1)
    jsonl_path: Optional[str] = None
    csv_path: Optional[str] = None


class FastOffloadConfig(FastOffloadConfigModel):
    """Top-level configuration for observer and synchronous offload modes."""

    enabled: bool = False
    mode: FastOffloadMode = FastOffloadMode.observe
    zero_stage: Literal[2] = 2
    failure_policy: ObserverFailurePolicy = ObserverFailurePolicy.raise_error
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    transfer: TransferConfig = Field(default_factory=TransferConfig)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    importance: ImportanceConfig = Field(default_factory=ImportanceConfig)
    hybrid_update: HybridUpdateConfig = Field(default_factory=HybridUpdateConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    @model_validator(mode="after")
    def validate_mode(self) -> "FastOffloadConfig":
        if self.mode != FastOffloadMode.observe and not self.enabled:
            raise ValueError("Offload modes require enabled=true")
        if self.importance.enabled and not self.enabled:
            raise ValueError("Importance selection requires enabled=true")
        if self.hybrid_update.enabled and not self.importance.enabled:
            raise ValueError("Hybrid update requires importance selection")
        if self.hybrid_update.enabled and self.mode != FastOffloadMode.observe:
            raise ValueError("Hybrid update requires mode=observe")
        if self.hybrid_update.compressed_collective_shadow and self.hybrid_update.zero2_takeover:
            raise ValueError("Hybrid Shadow and ZeRO-2 takeover cannot be enabled together")
        if self.hybrid_update.second_reduce_scatter_device == "cpu":
            if not self.hybrid_update.enabled or not self.hybrid_update.zero2_takeover:
                raise ValueError("CPU B reduce-scatter requires enabled ZeRO-2 takeover")
            if self.hybrid_update.accumulation_device != "cpu":
                raise ValueError("CPU B reduce-scatter requires CPU accumulation")
        if self.mode == FastOffloadMode.sync_offload:
            if self.scheduler.type != SchedulerType.synchronous or self.scheduler.max_inflight_tasks != 1:
                raise ValueError("sync_offload requires the synchronous scheduler with max_inflight_tasks=1")
        if self.mode == FastOffloadMode.async_offload:
            if self.scheduler.type != SchedulerType.overlap:
                raise ValueError("async_offload requires the overlap scheduler")
            if (self.transfer.async_strategy == "dedicated_stream"
                    and self.transfer.buffer_count < self.scheduler.max_inflight_tasks):
                raise ValueError("dedicated_stream requires buffer_count >= max_inflight_tasks")
        return self

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "FastOffloadConfig":
        """Validate a configuration dictionary."""
        return cls.model_validate(config)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "FastOffloadConfig":
        """Load and validate a UTF-8 JSON configuration file."""
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        return cls.model_validate(config)
