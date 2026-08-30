from __future__ import annotations

import time
from collections import deque
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch.amp import GradScaler, autocast

from ..models.task_modes import OUTPUT_KEYS
from ..config_dataclasses import TrainerConfig
from .checkpoint import CheckpointManager
from .evaluator import (
    MulticlassSemanticEvaluator,
    extract_class_names_from_batch,
    extract_semantic_targets_from_batch,
    inference_with_tta,
)
from .hooks import Hook, HookManager
from .runtime_state import capture_rng_state, restore_rng_state
from .visualization import VisualizationManager


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        criterion: torch.nn.Module,
        train_dataloader: Optional[Iterable],
        checkpoint_manager: CheckpointManager,
        val_dataloader: Optional[Iterable] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        cfg: Optional[TrainerConfig] = None,
        hooks: Optional[Sequence[Hook]] = None,
        visualizer: Optional[VisualizationManager] = None,
        raw_cfg_for_logging=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.lr_scheduler = lr_scheduler
        self.cfg = cfg or TrainerConfig()
        self.raw_cfg_for_logging = raw_cfg_for_logging

        self.device = torch.device(self.cfg.device)
        self.scaler = GradScaler(
            device="cuda",
            enabled=self.cfg.use_amp and self.device.type == "cuda",
        )

        self.save_dir = Path(self.cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)
        self.visualizer = visualizer

        self.hook_manager = HookManager(hooks or [])
        self.checkpoint_manager = checkpoint_manager

        self.global_iter = 0

        self.iters_per_cycle = None
        if self.train_dataloader is not None:
            batch_sampler = getattr(self.train_dataloader, "batch_sampler", None)
            if batch_sampler is not None and hasattr(batch_sampler, "__len__"):
                self.iters_per_cycle = len(batch_sampler)

        self.val_iters_per_epoch = None
        if self.val_dataloader is not None and hasattr(self.val_dataloader, "__len__"):
            self.val_iters_per_epoch = len(self.val_dataloader)

        val_max_iters = self._get_val_max_iters()
        if (
            self.val_iters_per_epoch is not None
            and val_max_iters is not None
        ):
            self.val_iters_per_epoch = min(self.val_iters_per_epoch, val_max_iters)

        self.log_state: Dict[str, object] = {}
        self._log_getters = []

        self._iter_time_history = deque(maxlen=self.cfg.log_window_size)
        self._data_time_history = deque(maxlen=self.cfg.log_window_size)
        self._train_stat_history = deque(maxlen=self.cfg.log_window_size)

        self._val_iter_time_history = deque(maxlen=self.cfg.log_window_size)
        self._val_data_time_history = deque(maxlen=self.cfg.log_window_size)
        self._val_metric_history = deque(maxlen=self.cfg.log_window_size)

        self._train_iterator = None
        self.is_resumed = False
        self._pending_rng_state = None
        self._pending_val_from_resume = False

    # ------------------------------------------------------------------
    # State dict (for checkpoint runtime_state)
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, object]:
        train_sampler_state = None
        if self.train_dataloader is not None:
            batch_sampler = getattr(self.train_dataloader, "batch_sampler", None)
            if batch_sampler is not None and hasattr(batch_sampler, "state_dict"):
                train_sampler_state = batch_sampler.state_dict()

        return {
            "rng": capture_rng_state(),
            "data": {
                "sampler": train_sampler_state,
            },
            "hooks": self.hook_manager.state_dict(),
        }

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def _get_val_max_iters(self) -> Optional[int]:
        value = getattr(self.cfg, "val_max_iters", None)
        if value is None:
            return None
        value = int(value)
        if value <= 0:
            return None
        return value

    def _restore_runtime_state(self, runtime_state: Dict) -> None:
        """Restore sampler and hook state from checkpoint.

        RNG state is deferred — saved to ``_pending_rng_state`` and restored
        after the iterator and W&B are set up, so the resume process itself
        does not consume the RNG.
        """
        if not runtime_state:
            return

        data_state = runtime_state.get("data", {}) or {}
        sampler_state = data_state.get("sampler", None)
        if sampler_state is not None and self.train_dataloader is not None:
            batch_sampler = getattr(self.train_dataloader, "batch_sampler", None)
            if batch_sampler is not None and hasattr(batch_sampler, "load_state_dict"):
                batch_sampler.load_state_dict(sampler_state)

        hooks_state = runtime_state.get("hooks", {}) or {}
        self.hook_manager.load_state_dict(hooks_state)

        rng_state = runtime_state.get("rng", None)
        if rng_state is not None:
            self._pending_rng_state = rng_state

    def resume_from(self, path: str) -> Dict:
        ckpt = self.checkpoint_manager.load(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            scheduler=self.lr_scheduler,
        )
        self.global_iter = int(ckpt.get("global_iter", 0))
        self.is_resumed = True

        extra = ckpt.get("extra", {}) or {}
        if extra.get("val_status") == "pending":
            self._pending_val_from_resume = True

        self._restore_runtime_state(ckpt.get("runtime_state", {}) or {})
        print(f"Resumed from {path}, starting at iter={self.global_iter}")
        return ckpt

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    def _move_to_device(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device, non_blocking=True)

        if is_dataclass(obj):
            for field in fields(obj):
                setattr(obj, field.name, self._move_to_device(getattr(obj, field.name)))
            return obj

        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._move_to_device(v) for v in obj]

        if isinstance(obj, tuple):
            return tuple(self._move_to_device(v) for v in obj)

        return obj

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def _get_loss_train_progress(self) -> float:
        """Return 0 at the first step and 1 at the final optimizer step."""
        max_iters = int(self.cfg.max_iters)
        if max_iters <= 1:
            return 0.0
        return min(max(float(self.global_iter) / (max_iters - 1), 0.0), 1.0)

    def _compute_train_loss_streaming(
        self,
        batch,
    ) -> Dict[str, torch.Tensor]:
        """Streaming per-chunk loss/backward with proxy leaf gradient isolation.

        Low-resolution Refiner runs once on all classes. High-resolution
        decoding, loss and backward run per chunk. A detached proxy leaf
        accumulates gradients from all chunks; after all chunks, the
        accumulated gradient is handed back to the real Refiner graph in
        a single backward call.
        """
        if not hasattr(self.model, "build_encoder_refiner_cache"):
            raise AttributeError(
                "Model must provide build_encoder_refiner_cache(batch)."
            )

        label_map = batch.find_targets[0].semantic_label_map
        use_amp = self.cfg.use_amp and self.device.type == "cuda"

        # ---- Build cache and run low-res Refiner once ----
        with autocast(device_type=self.device.type, enabled=use_amp):
            encoder_refiner_cache = self.model.build_encoder_refiner_cache(batch)

            refiner_out = (
                self.model.run_encoder_refiner_lowres_from_cache(
                    encoder_refiner_cache,
                    batch,
                    return_debug=False,
                )
            )

        refiner_features_36 = refiner_out["refiner_features_36"]
        del refiner_out

        C_total = int(refiner_features_36.shape[1])

        metadata = batch.find_metadatas[0]

        num_prompt_channels = int(metadata.num_prompts)
        num_label_classes = int(metadata.num_classes)

        if C_total != num_prompt_channels:
            raise ValueError(
                "Refiner prompt channel count does not match metadata: "
                f"refiner={C_total}, "
                f"metadata.num_prompts={num_prompt_channels}."
            )

        prompt_to_class_id = torch.as_tensor(
            metadata.prompt_to_class_id,
            dtype=torch.long,
            device=refiner_features_36.device,
        )

        if prompt_to_class_id.ndim != 1:
            raise ValueError(
                "metadata.prompt_to_class_id must be 1D."
            )

        if int(prompt_to_class_id.numel()) != C_total:
            raise ValueError(
                "metadata.prompt_to_class_id length must equal the "
                f"Refiner prompt count. Got "
                f"{prompt_to_class_id.numel()} and {C_total}."
            )

        # Detached proxy leaf — accumulates gradients from all chunks.
        refiner_proxy = refiner_features_36.detach().requires_grad_(True)

        # Build streaming loss context once.
        train_progress = self._get_loss_train_progress()
        with autocast(device_type=self.device.type, enabled=use_amp):
            loss_context = self.criterion.prepare_streaming_context(
                targets={"label_map": label_map},
                num_prompt_channels=C_total,
                prompt_to_class_id=prompt_to_class_id,
                num_label_classes=num_label_classes,
                target_hw=(288, 288),
                train_progress=train_progress,
            )

        need_teacher_logits = (
            loss_context.sam3_mask_distill_weight > 0.0
            and loss_context.teacher_mix_ratio > 0.0
            and loss_context.distill_pixel_denom > 0
        )

        chunk_class_counts = encoder_refiner_cache["chunk_class_counts"]

        # GPU-side accumulators (detached scalars, .item() deferred).
        accum: Dict[str, torch.Tensor] = {}

        chunk_start = 0
        for num_chunk_classes in chunk_class_counts:
            chunk_end = chunk_start + num_chunk_classes

            chunk_has_present_pair = bool(
                loss_context.presence_target[
                    :, chunk_start:chunk_end
                ].any().item()
            )

            refiner_proxy_chunk = refiner_proxy[
                :, chunk_start:chunk_end
            ]

            with autocast(device_type=self.device.type, enabled=use_amp):
                chunk_outputs = (
                    self.model.decode_encoder_refiner_chunk_from_cache(
                        encoder_refiner_cache=encoder_refiner_cache,
                        refiner_feature_36_chunk=refiner_proxy_chunk,
                        class_start=chunk_start,
                        class_end=chunk_end,
                        return_teacher_logits=(
                            need_teacher_logits
                            and chunk_has_present_pair
                        ),
                    )
                )

                chunk_loss_dict = self.criterion.forward_chunk(
                    outputs=chunk_outputs,
                    context=loss_context,
                    class_start=chunk_start,
                    class_end=chunk_end,
                )

            chunk_total_loss = chunk_loss_dict["total_loss"]

            # Backward outside autocast.
            self.scaler.scale(chunk_total_loss).backward()

            # Accumulate detached stats on GPU.
            for key in (
                "loss_final_bce",
                "loss_final_dice",
                "loss_sam3_mask_distill_bce",
                "loss_sam3_mask_distill_weighted",
                "total_loss",
            ):
                val = chunk_loss_dict.get(key)
                if val is not None and torch.is_tensor(val):
                    detached = val.detach()
                    if key not in accum:
                        accum[key] = detached
                    else:
                        accum[key] = accum[key] + detached

            # Release chunk intermediates.
            del chunk_outputs, chunk_loss_dict, chunk_total_loss, refiner_proxy_chunk

            chunk_start = chunk_end

        if chunk_start != C_total:
            raise ValueError(
                f"Chunk index mismatch: final chunk_start={chunk_start}, "
                f"expected C_total={C_total}."
            )

        accum["sam3_mask_distill_train_progress"] = (
            refiner_features_36.new_tensor(train_progress)
        )
        accum["sam3_mask_distill_gt_ratio"] = (
            refiner_features_36.new_tensor(loss_context.gt_mix_ratio)
        )
        accum["sam3_mask_distill_teacher_ratio"] = (
            refiner_features_36.new_tensor(
                loss_context.teacher_mix_ratio
            )
        )

        # ---- Hand proxy gradient back to real Refiner graph ----
        if refiner_proxy.grad is None:
            raise RuntimeError(
                "No gradient was accumulated on refiner_proxy. "
                "Check that chunk losses have non-zero contribution."
            )

        scaled_refiner_grad = refiner_proxy.grad.detach()

        torch.autograd.backward(
            tensors=refiner_features_36,
            grad_tensors=scaled_refiner_grad,
        )

        return accum

    def train_step(
        self,
        batch,
    ) -> Dict[str, float]:
        if self.optimizer is None:
            raise RuntimeError("Optimizer is None, cannot run train_step().")

        batch = self._move_to_device(batch)
        self.optimizer.zero_grad(set_to_none=True)

        accum = self._compute_train_loss_streaming(batch)

        # Optimizer step (once per batch).
        self.scaler.unscale_(self.optimizer)

        if self.cfg.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.grad_clip_norm,
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # Convert GPU accumulators to float (single CPU sync).
        stats = {}
        for key, value in accum.items():
            stats[key] = float(value.item())

        return stats

    # ------------------------------------------------------------------
    # Sampler commit
    # ------------------------------------------------------------------

    def _commit_sampler_batch(self) -> None:
        if self.train_dataloader is None:
            return
        batch_sampler = getattr(self.train_dataloader, "batch_sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "commit_batch"):
            batch_sampler.commit_batch()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _forward_val_outputs(self, batch) -> Dict[str, torch.Tensor]:
        use_amp = self.cfg.use_amp and self.device.type == "cuda"
        with autocast(device_type=self.device.type, enabled=use_amp):
            outputs = inference_with_tta(self.model, batch, tta_cfg=self.cfg.tta_cfg)
        return outputs

    @staticmethod
    def _average_stats(stats_list: list[Dict[str, float]]) -> Dict[str, float]:
        if not stats_list:
            return {}

        keys = sorted({k for stats in stats_list for k in stats.keys()})
        out: Dict[str, float] = {}
        for k in keys:
            vals = [s[k] for s in stats_list if k in s]
            if vals:
                out[k] = sum(vals) / len(vals)
        return out

    def _get_current_lrs(self) -> list[float]:
        if self.optimizer is None:
            return []
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def _get_memory_mb(self) -> Optional[int]:
        if self.device.type != "cuda":
            return None
        return int(torch.cuda.max_memory_allocated(self.device) / 1024 / 1024)

    @staticmethod
    def _mean_of_history(values) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def register_log_getter(self, fn):
        if fn is None:
            return
        self._log_getters.append(fn)

    @staticmethod
    def _extract_prompt_names_from_dataloader(
        dataloader,
    ) -> Optional[list[str]]:
        if dataloader is None:
            return None

        dataset = getattr(dataloader, "dataset", None)

        while dataset is not None and hasattr(dataset, "dataset"):
            dataset = dataset.dataset

        if dataset is None:
            return None

        prompt_names = getattr(dataset, "prompt_names", None)
        if prompt_names is None:
            return None

        prompt_names = [str(x) for x in prompt_names]
        if len(prompt_names) == 0:
            return None

        return prompt_names

    def _prepare_text_cache_for_dataloader(
        self,
        dataloader,
        force: bool = False,
    ) -> None:
        if dataloader is None:
            return

        if not hasattr(self.model, "prepare_text_cache"):
            return

        prompt_names = self._extract_prompt_names_from_dataloader(
            dataloader
        )
        if prompt_names is None:
            return

        self.model.prepare_text_cache(
            class_names=prompt_names,
            device=self.device,
            force=force,
        )

    def _to_loggable_scalar(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().item())
            return None
        if isinstance(value, (float, int, bool, str)):
            return value
        return None

    def _collect_extra_log_vars(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for fn in self._log_getters:
            try:
                values = fn(self)
            except Exception as e:
                out[f"log_getter_error_{len(out)}"] = str(e)
                continue

            if not isinstance(values, dict):
                continue

            for k, v in values.items():
                vv = self._to_loggable_scalar(v)
                if vv is not None:
                    out[str(k)] = vv
        return out

    # ------------------------------------------------------------------
    # Log state
    # ------------------------------------------------------------------

    def _update_train_log_state(
        self,
        stats: Dict[str, float],
        data_time: float,
        iter_time: float,
    ) -> None:
        self._data_time_history.append(float(data_time))
        self._iter_time_history.append(float(iter_time))
        self._train_stat_history.append(dict(stats))

        avg_data_time = self._mean_of_history(self._data_time_history)
        avg_iter_time = self._mean_of_history(self._iter_time_history)
        avg_stats = self._average_stats(list(self._train_stat_history))

        remaining_iters = max(self.cfg.max_iters - self.global_iter, 0)
        eta_seconds = avg_iter_time * remaining_iters

        self.log_state = {
            "mode": "train",
            "iter": int(self.global_iter),
            "max_iters": int(self.cfg.max_iters),
            "data_cycle": self._get_sampler_cycle(),
            "iters_per_cycle": self.iters_per_cycle,
            "lrs": self._get_current_lrs(),
            "eta_seconds": eta_seconds,
            "iter_time": avg_iter_time,
            "data_time": avg_data_time,
            "memory_mb": self._get_memory_mb(),
            "log_vars": avg_stats,
            "extra_log_vars": self._collect_extra_log_vars(),
        }

    def _update_val_log_state(
        self,
        val_step: int,
        metric_stats_snapshot: Dict[str, float],
        data_time: float,
        iter_time: float,
    ) -> None:
        self._val_data_time_history.append(float(data_time))
        self._val_iter_time_history.append(float(iter_time))
        self._val_metric_history.append(dict(metric_stats_snapshot))

        avg_data_time = self._mean_of_history(self._val_data_time_history)
        avg_iter_time = self._mean_of_history(self._val_iter_time_history)
        avg_metrics = self._average_stats(list(self._val_metric_history))

        eta_seconds = None
        if self.val_iters_per_epoch is not None:
            remaining_iters = max(self.val_iters_per_epoch - val_step, 0)
            eta_seconds = avg_iter_time * remaining_iters

        self.log_state = {
            "mode": "val",
            "iter": int(self.global_iter),
            "max_iters": int(self.cfg.max_iters),
            "val_iter": int(val_step),
            "val_total_iters": self.val_iters_per_epoch,
            "eta_seconds": eta_seconds,
            "iter_time": avg_iter_time,
            "data_time": avg_data_time,
            "log_vars": avg_metrics,
            "extra_log_vars": self._collect_extra_log_vars(),
        }

    # ------------------------------------------------------------------
    # Data iterator
    # ------------------------------------------------------------------

    def _get_sampler_cycle(self) -> Optional[int]:
        if self.train_dataloader is None:
            return None
        batch_sampler = getattr(self.train_dataloader, "batch_sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "cycle"):
            return int(batch_sampler.cycle)
        return None

    def _build_train_iterator(self) -> None:
        if self.train_dataloader is None:
            self._train_iterator = None
            return

        self._train_iterator = iter(self.train_dataloader)

    def _next_train_batch(self):
        if self.train_dataloader is None:
            raise RuntimeError("train_dataloader is None, cannot fetch training batch.")

        if self._train_iterator is None:
            self._build_train_iterator()

        try:
            return next(self._train_iterator)
        except StopIteration:
            self._train_iterator = iter(self.train_dataloader)
            return next(self._train_iterator)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        train_stats: Dict[str, float],
        val_stats: Optional[Dict[str, float]] = None,
        checkpoint_reason: str = "periodic",
        val_status: str = "not_due",
    ) -> Path:
        return self.checkpoint_manager.save(
            global_iter=self.global_iter,
            model=self.model,
            checkpoint_reason=checkpoint_reason,
            val_status=val_status,
            runtime_state=self.state_dict(),
            optimizer=self.optimizer,
            scaler=self.scaler,
            scheduler=self.lr_scheduler,
            train_stats=train_stats,
            val_stats=val_stats or {},
            extra={
                "monitor": self.cfg.monitor,
                "monitor_mode": self.cfg.monitor_mode,
            },
        )

    # ------------------------------------------------------------------
    # Validation loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def val(self) -> Dict[str, float]:
        if self.val_dataloader is None:
            return {}

        self._prepare_text_cache_for_dataloader(self.val_dataloader, force=False)
        self.hook_manager.call("before_val", self, self.global_iter)

        self.model.eval()
        self._val_iter_time_history.clear()
        self._val_data_time_history.clear()
        self._val_metric_history.clear()

        eval_cfg = dict(self.cfg.eval_cfg or {})
        evaluator = MulticlassSemanticEvaluator(**eval_cfg)
        class_names = None

        end = time.perf_counter()

        val_max_iters = self._get_val_max_iters()

        for it, batch in enumerate(self.val_dataloader, start=1):
            if val_max_iters is not None and it > val_max_iters:
                break

            data_time = time.perf_counter() - end

            batch = self._move_to_device(batch)

            outputs = self._forward_val_outputs(batch)

            targets = extract_semantic_targets_from_batch(batch)
            evaluator.update(outputs, targets)

            if class_names is None:
                class_names = extract_class_names_from_batch(batch)

            if self.visualizer is not None:
                self.visualizer.run(
                    model=self.model,
                    batch=batch,
                    semantic_outputs=outputs,
                    semantic_targets=targets,
                    epoch=self.global_iter,
                    stage="val",
                )

            metric_snapshot = evaluator.compute(eval_class_names=class_names)
            iter_time = time.perf_counter() - end

            self._update_val_log_state(
                val_step=it,
                metric_stats_snapshot=metric_snapshot,
                data_time=data_time,
                iter_time=iter_time,
            )

            self.hook_manager.call("after_val_iter", self, self.global_iter, it, batch, metric_snapshot)

            end = time.perf_counter()

        stats = evaluator.compute(eval_class_names=class_names)
        if class_names is not None:
            stats["_class_names"] = class_names

        self.hook_manager.call("after_val", self, self.global_iter, stats)
        return stats

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        if self.train_dataloader is None:
            raise RuntimeError("train_dataloader is None, cannot run train().")

        # 1. Build iterator from (possibly restored) sampler state.
        self._build_train_iterator()

        # 2. Call before_run — W&B initializes or resumes here.
        self.hook_manager.call("before_run", self)

        self._prepare_text_cache_for_dataloader(self.train_dataloader, force=False)

        self.model.train()
        self._iter_time_history.clear()
        self._data_time_history.clear()
        self._train_stat_history.clear()

        # 3. Restore RNG *last*, so the resume process itself does not
        #    consume RNG state.
        if self._pending_rng_state is not None:
            restore_rng_state(self._pending_rng_state)
            self._pending_rng_state = None

        train_stats_window: list[Dict[str, float]] = []

        end = time.perf_counter()

        try:
            self._run_training_loop(train_stats_window, end)
        except KeyboardInterrupt as exc:
            self.hook_manager.call("on_exception", self, exc)
            raise
        except BaseException as exc:
            self._save_checkpoint(
                self._average_stats(train_stats_window),
                checkpoint_reason="exception",
            )
            self.hook_manager.call("on_exception", self, exc)
            raise
        else:
            self._finish_training(train_stats_window)

    def _run_training_loop(
        self,
        train_stats_window: list[Dict[str, float]],
        end: float,
    ) -> None:
        # Replay pending validation that was interrupted before finalization.
        if self._pending_val_from_resume:
            self._pending_val_from_resume = False
            if self.val_dataloader is not None:
                ckpt_path = self.save_dir / f"iter_{int(self.global_iter):07d}.pth"
                print(f"Replaying pending validation at iter={self.global_iter}...")
                val_stats = self.val()
                self.model.train()
                if ckpt_path.exists():
                    self.checkpoint_manager.finalize_after_validation(
                        ckpt_path=ckpt_path,
                        val_stats=val_stats,
                        runtime_state=self.state_dict(),
                        extra={
                            "monitor": self.cfg.monitor,
                            "monitor_mode": self.cfg.monitor_mode,
                        },
                    )
                    print(f"Finalized pending validation checkpoint at iter={self.global_iter}")

        while self.global_iter < self.cfg.max_iters:
            data_time = time.perf_counter() - end

            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            batch = self._next_train_batch()
            next_iter = self.global_iter + 1

            self.hook_manager.call("before_train_iter", self, next_iter, batch)

            stats = self.train_step(batch)

            self._commit_sampler_batch()

            train_stats_window.append(stats)

            self.global_iter = next_iter

            iter_time = time.perf_counter() - end

            self._update_train_log_state(
                stats=stats,
                data_time=data_time,
                iter_time=iter_time,
            )

            self.hook_manager.call("after_train_iter", self, self.global_iter, batch, stats)

            should_eval = (
                self.val_dataloader is not None
                and self.cfg.eval_interval > 0
                and self.global_iter % self.cfg.eval_interval == 0
            )
            should_save = (
                self.cfg.save_interval > 0
                and self.global_iter % self.cfg.save_interval == 0
            )

            averaged_train_stats = self._average_stats(train_stats_window)

            if should_save:
                val_status = "pending" if should_eval else "not_due"
                ckpt_path = self._save_checkpoint(
                    averaged_train_stats,
                    val_status=val_status,
                )
                print(f"saved training-state checkpoint at iter={self.global_iter}: {ckpt_path}")
            else:
                ckpt_path = None

            if should_eval:
                val_stats = self.val()
                self.model.train()
            else:
                val_stats = {}

            if ckpt_path is not None and should_eval:
                self.checkpoint_manager.finalize_after_validation(
                    ckpt_path=ckpt_path,
                    val_stats=val_stats,
                    runtime_state=self.state_dict(),
                    extra={
                        "monitor": self.cfg.monitor,
                        "monitor_mode": self.cfg.monitor_mode,
                    },
                )
                print(f"finalized checkpoint with validation stats at iter={self.global_iter}: {ckpt_path}")

            end = time.perf_counter()

    def _finish_training(
        self,
        train_stats_window: list[Dict[str, float]],
    ) -> None:
        final_train_stats = self._average_stats(train_stats_window)

        need_final_save = (
            self.cfg.save_interval <= 0
            or self.global_iter % self.cfg.save_interval != 0
        )

        final_ckpt_path = None
        if need_final_save:
            final_ckpt_path = self._save_checkpoint(final_train_stats, checkpoint_reason="final")
            print(f"saved final training-state checkpoint at iter={self.global_iter}: {final_ckpt_path}")

        need_final_eval = (
            self.val_dataloader is not None
            and (
                self.cfg.eval_interval <= 0
                or self.global_iter % self.cfg.eval_interval != 0
            )
        )

        if need_final_eval:
            final_val_stats = self.val()
            self.model.train()
        else:
            final_val_stats = {}

        if final_ckpt_path is not None and need_final_eval:
            self.checkpoint_manager.finalize_after_validation(
                ckpt_path=final_ckpt_path,
                val_stats=final_val_stats,
                runtime_state=self.state_dict(),
                extra={
                    "monitor": self.cfg.monitor,
                    "monitor_mode": self.cfg.monitor_mode,
                },
            )
            print(f"finalized final checkpoint with validation stats at iter={self.global_iter}: {final_ckpt_path}")

        self.hook_manager.call("after_run", self)
