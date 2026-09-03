# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Controller that isolates DeepSpeed hooks from observer implementation."""

from typing import Any, Callable, Optional

from deepspeed.utils import logger

from deepspeed.runtime.fastoffload.adapters.base import ObserverAdapter
from deepspeed.runtime.fastoffload.config import ObserverFailurePolicy
from deepspeed.runtime.fastoffload.telemetry.observer import FastOffloadObserver


class NullFastOffloadController:
    """No-op controller used when FastOffload is not installed."""

    @property
    def importance_registry(self) -> None:
        return None

    def takeover_active(self) -> bool:
        return False

    def takeover_native_boundary(self) -> bool:
        return False

    def on_backward_begin(self) -> None:
        pass

    def on_gradient_ready(self, parameter: Any, group_id: int) -> bool:
        return False

    def transfer_gradient(self, parameter: Any) -> bool:
        return False

    def validate_native_gradient(self, parameter: Any, group_id: int) -> None:
        pass

    def on_gradient_reduced(self, parameter: Any, group_id: int) -> None:
        pass

    def prepare_forward(self) -> None:
        pass

    def prepare_step(self) -> None:
        pass

    def takeover_step(self, loss_scale: float, clip_grad: float) -> None:
        return None

    def hybrid_state_dict(self) -> None:
        return None

    def load_hybrid_state_dict(self, state_dict: Any) -> None:
        if state_dict is not None:
            raise RuntimeError("Cannot load hybrid state without an active takeover runtime")

    def on_gradient_bucket(self, communication_dtype: Any, bucket: Any) -> None:
        pass

    def on_backward_end(self) -> None:
        pass

    def on_step_begin(self) -> None:
        pass

    def on_step_end(self) -> None:
        pass

    def close(self) -> None:
        pass


class FastOffloadController:
    """Build stable contexts and apply the configured observer failure policy."""

    def __init__(self,
                 adapter: ObserverAdapter,
                 observer: FastOffloadObserver,
                 failure_policy: ObserverFailurePolicy,
                 scheduler: Optional[Any] = None,
                 importance_selector: Optional[Any] = None,
                 hybrid_shadow: Optional[Any] = None,
                 hybrid_runtime: Optional[Any] = None,
                 takeover_runtime: Optional[Any] = None) -> None:
        self._adapter = adapter
        self._observer = observer
        self._failure_policy = failure_policy
        self._scheduler = scheduler
        self._importance_selector = importance_selector
        self._hybrid_shadow = hybrid_shadow
        self._hybrid_runtime = hybrid_runtime
        self._takeover_runtime = takeover_runtime
        self._takeover_batch = None
        self._disabled = False
        self._closed = False

    @property
    def observer(self) -> FastOffloadObserver:
        return self._observer

    @property
    def disabled(self) -> bool:
        return self._disabled

    def takeover_active(self) -> bool:
        return self._takeover_runtime is not None and self._takeover_runtime.active

    def takeover_native_boundary(self) -> bool:
        return self.takeover_active() and self._takeover_runtime.native_dense_boundary

    @property
    def importance_registry(self) -> Optional[Any]:
        if self._importance_selector is None:
            return None
        return self._importance_selector.registry

    def on_backward_begin(self) -> None:
        self._invoke(lambda: self._observer.on_backward_begin(self._adapter.create_backward_begin_context()))

    def on_gradient_ready(self, parameter: Any, group_id: int) -> bool:
        self._invoke(
            lambda: self._observer.on_gradient_ready(self._adapter.create_gradient_context(parameter, group_id)))
        if self._hybrid_shadow is not None:
            self._invoke(lambda: self._hybrid_shadow.capture(parameter, group_id))
        if self._takeover_runtime is None or not self._takeover_runtime.active:
            return False
        if self._takeover_runtime.native_dense_boundary:
            return False
        handled = self._takeover_runtime.capture_gradient(parameter, group_id)
        if handled:
            self._adapter.clear_gradient(parameter)
        return handled

    def transfer_gradient(self, parameter: Any) -> bool:
        """Synchronously transfer one local shard when sync mode is active."""
        if self._takeover_runtime is not None and self._takeover_runtime.capture_native_reduced_gradient(parameter):
            return True
        if self._disabled or self._scheduler is None:
            return False
        stable_source = getattr(self._scheduler, "requires_stable_source", False)
        transfer_view = self._adapter.create_gradient_transfer_view(parameter, stable_source=stable_source)
        handled = self._scheduler.submit(transfer_view)
        if handled:
            self._adapter.clear_gradient(parameter)
        return handled

    def validate_native_gradient(self, parameter: Any, group_id: int) -> None:
        if self._hybrid_shadow is not None:
            self._invoke(lambda: self._hybrid_shadow.validate_native_gradient(parameter, group_id))

    def on_gradient_reduced(self, parameter: Any, group_id: int) -> None:
        if self._hybrid_shadow is not None:
            self._invoke(lambda: self._hybrid_shadow.validate_owner_partition(parameter, group_id))
        self._invoke(
            lambda: self._observer.on_gradient_reduced(self._adapter.create_gradient_context(parameter, group_id)))

    def on_gradient_bucket(self, communication_dtype: Any, bucket: Any) -> None:
        self._invoke(lambda: self._observer.on_gradient_bucket(
            self._adapter.create_bucket_context(communication_dtype, bucket)))

    def on_backward_end(self) -> None:
        if self._scheduler is not None:
            self._scheduler.progress()
        if self._hybrid_shadow is not None:
            self._invoke(self._hybrid_shadow.reduce)
        if self._takeover_runtime is not None and self._takeover_runtime.active:
            batch = self._takeover_runtime.finish_microbatch()
            if batch is not None:
                if self._takeover_batch is not None:
                    raise RuntimeError("Takeover gradient batch was not consumed by optimizer step")
                self._takeover_batch = batch
        self._invoke(lambda: self._observer.on_backward_end(self._adapter.create_step_context()))

    def prepare_forward(self) -> None:
        if self._takeover_runtime is not None and self._takeover_runtime.active:
            self._takeover_runtime.prepare_forward()

    def prepare_step(self) -> None:
        if self._scheduler is not None:
            self._scheduler.prepare_step()

    def takeover_step(self, loss_scale: float, clip_grad: float) -> Optional[Any]:
        if self._takeover_runtime is None or not self._takeover_runtime.active:
            return None
        if self._takeover_batch is None:
            raise RuntimeError("Takeover optimizer step has no completed GAS gradient batch")
        batch = self._takeover_batch
        self._takeover_batch = None
        return self._takeover_runtime.step(batch, loss_scale, clip_grad)

    def hybrid_state_dict(self) -> Optional[Any]:
        runtime = self._takeover_runtime if self._takeover_runtime is not None else self._hybrid_runtime
        if runtime is None:
            return None
        return runtime.state_dict()

    def load_hybrid_state_dict(self, state_dict: Any) -> None:
        if state_dict is None:
            return
        runtime = self._takeover_runtime if self._takeover_runtime is not None else self._hybrid_runtime
        if runtime is None:
            raise RuntimeError("Checkpoint contains hybrid state but takeover runtime is inactive")
        runtime.load_state_dict(state_dict)

    def on_step_begin(self) -> None:
        self._invoke(lambda: self._observer.on_step_begin(self._adapter.create_step_context()))

    def on_step_end(self) -> None:
        context = self._adapter.create_step_context()
        self._invoke(lambda: self._observer.on_step_end(context))
        if self._importance_selector is not None:
            self._invoke(lambda: self._importance_selector.on_step_end(context.global_step + 1))
        if self._hybrid_shadow is not None:
            self._invoke(self._hybrid_shadow.complete_step)
        self._adapter.complete_step()

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._scheduler is not None:
                self._scheduler.close()
        finally:
            try:
                if self._importance_selector is not None:
                    self._importance_selector.close()
                if self._hybrid_shadow is not None:
                    self._hybrid_shadow.close()
                if self._takeover_runtime is not None:
                    self._takeover_runtime.close()
            finally:
                if not self._disabled:
                    self._observer.close()
                self._disabled = True
                self._closed = True

    def _invoke(self, callback: Callable[[], None]) -> None:
        if self._disabled:
            return
        try:
            callback()
        except Exception as error:
            if self._failure_policy == ObserverFailurePolicy.raise_error:
                raise
            if self._failure_policy == ObserverFailurePolicy.disable_observer:
                logger.warning(f"FastOffload observer disabled after callback failure: {error}")
                self._disabled = True
            else:
                logger.warning(f"FastOffload observer callback failed: {error}")
