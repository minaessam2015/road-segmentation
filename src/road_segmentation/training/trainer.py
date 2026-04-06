"""Main training loop with AMP, gradient accumulation, checkpointing, and logging."""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt  # noqa: E402

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from road_segmentation.config import ExperimentConfig, config_to_dict, save_config
from road_segmentation.models.factory import (
    freeze_encoder,
    get_decoder_parameters,
    get_encoder_parameters,
    unfreeze_encoder,
)
from road_segmentation.training.callbacks import EarlyStopping, ModelEMA
from road_segmentation.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    verify_config_compatibility,
)
from road_segmentation.training.metrics import MetricTracker
from road_segmentation.training.visualization import (
    plot_prediction_samples,
    plot_training_curves,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Full training loop with best practices.

    Features:
        - Mixed precision training (AMP) on CUDA
        - Gradient accumulation for larger effective batch sizes
        - Two-phase encoder freeze/unfreeze with differential LR
        - Checkpoint save/resume with full state
        - Early stopping on validation metric
        - Optional Model EMA
        - CSV logging + optional TensorBoard
        - Periodic prediction visualizations
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
    ) -> None:
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn.to(device)
        self.device = device

        self.history: List[Dict[str, float]] = []
        self.start_epoch = 0
        self.best_metric = -float("inf") if config.training.early_stopping_mode == "max" else float("inf")

        # Set up directories
        self.log_dir = Path(config.logging.log_dir) / config.logging.experiment_name
        self.ckpt_dir = Path(config.checkpoint.save_dir) / config.logging.experiment_name
        self.viz_dir = self.log_dir / "visualizations"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        save_config(config, self.log_dir / "config.yaml")

        # Set up components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_amp()
        self._setup_early_stopping()
        self._setup_ema()
        self._setup_tensorboard()
        self.metric_tracker = MetricTracker(threshold=0.5)

        # Collect fixed val samples for visualization
        self._viz_batch = self._collect_viz_samples()

        # Resume from checkpoint if specified
        if config.checkpoint.resume_from:
            self._resume()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_optimizer(self) -> None:
        cfg = self.config
        frozen = cfg.training.freeze_encoder_epochs > 0

        if frozen:
            params = get_decoder_parameters(self.model)
        else:
            params = [
                {"params": get_decoder_parameters(self.model), "lr": cfg.optimizer.lr},
                {"params": get_encoder_parameters(self.model), "lr": cfg.optimizer.lr * cfg.training.encoder_lr_factor},
            ]

        self.optimizer = self._build_optimizer(params)

    def _setup_optimizer_unfrozen(self) -> None:
        """Create optimizer with differential LR (encoder + decoder param groups)."""
        cfg = self.config
        params = [
            {"params": get_decoder_parameters(self.model), "lr": cfg.optimizer.lr},
            {"params": get_encoder_parameters(self.model), "lr": cfg.optimizer.lr * cfg.training.encoder_lr_factor},
        ]
        self.optimizer = self._build_optimizer(params)

    def _build_optimizer(self, params) -> torch.optim.Optimizer:
        cfg = self.config
        if cfg.optimizer.type == "adamw":
            return torch.optim.AdamW(params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
        if cfg.optimizer.type == "adam":
            return torch.optim.Adam(params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
        if cfg.optimizer.type == "sgd":
            return torch.optim.SGD(
                params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay, momentum=cfg.optimizer.momentum,
            )
        raise ValueError(f"Unknown optimizer: {cfg.optimizer.type}")

    def _setup_scheduler(self) -> None:
        cfg = self.config.scheduler
        epochs = self.config.training.epochs

        if cfg.type == "cosine":
            self.scheduler: Optional[Any] = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.params.get("T_max", epochs), eta_min=cfg.params.get("eta_min", 1e-6),
            )
        elif cfg.type == "cosine_warm_restarts":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=cfg.params.get("T_0", 10), T_mult=cfg.params.get("T_mult", 2),
                eta_min=cfg.params.get("eta_min", 1e-6),
            )
        elif cfg.type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=cfg.params.get("step_size", 15), gamma=cfg.params.get("gamma", 0.1),
            )
        elif cfg.type == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=self.config.training.early_stopping_mode,
                patience=cfg.params.get("patience", 5), factor=cfg.params.get("factor", 0.5),
            )
        elif cfg.type == "one_cycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.config.optimizer.lr,
                epochs=epochs, steps_per_epoch=len(self.train_loader),
            )
        else:
            self.scheduler = None

    def _setup_amp(self) -> None:
        use_amp = self.config.training.mixed_precision and self.device.type == "cuda"
        self.use_amp = use_amp
        self.scaler = GradScaler("cuda") if use_amp else None
        self.amp_device_type = "cuda" if use_amp else "cpu"

    def _setup_early_stopping(self) -> None:
        patience = self.config.training.early_stopping_patience
        if patience and patience > 0:
            self.early_stopping: Optional[EarlyStopping] = EarlyStopping(
                patience=patience,
                mode=self.config.training.early_stopping_mode,
            )
        else:
            self.early_stopping = None

    def _setup_ema(self) -> None:
        if self.config.training.ema:
            self.ema = ModelEMA(self.model, decay=self.config.training.ema_decay)
        else:
            self.ema: Optional[ModelEMA] = None

    def _setup_tensorboard(self) -> None:
        self.tb_writer = None
        if self.config.logging.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
            except ImportError:
                logger.warning("TensorBoard not installed. Skipping TB logging.")

    def _collect_viz_samples(self) -> Optional[Dict[str, torch.Tensor]]:
        """Collect a fixed batch of validation samples for visualization."""
        n = self.config.logging.num_visualization_samples
        if n <= 0:
            return None
        batch = next(iter(self.val_loader))
        return {
            "image": batch["image"][:n],
            "mask": batch["mask"][:n],
        }

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------

    def _resume(self) -> None:
        path = Path(self.config.checkpoint.resume_from)
        logger.info(f"Resuming from checkpoint: {path}")

        # Load model weights first (always safe)
        state = load_checkpoint(
            path, self.model,
            optimizer=None,  # load optimizer separately below
            scheduler=None,
            scaler=None,
            ema_model=self.ema.shadow if self.ema else None,
            device=self.device,
        )
        self.start_epoch = state.epoch + 1
        self.best_metric = state.best_metric
        verify_config_compatibility(state.config_dict, config_to_dict(self.config))

        # If resuming into a different training phase (frozen vs unfrozen),
        # the optimizer param groups won't match — rebuild optimizer for the
        # current phase instead of loading stale state.
        freeze_epochs = self.config.training.freeze_encoder_epochs
        if self.start_epoch >= freeze_epochs > 0:
            unfreeze_encoder(self.model)
            self._setup_optimizer_unfrozen()
            self._setup_scheduler()
            logger.info("Resumed past freeze phase — rebuilt optimizer with differential LR.")
        else:
            # Try to restore optimizer/scheduler state
            try:
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                if "optimizer_state_dict" in ckpt:
                    self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if self.scheduler is not None and "scheduler_state_dict" in ckpt:
                    self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                if self.scaler is not None and "scaler_state_dict" in ckpt:
                    self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Could not restore optimizer state ({e}). Using fresh optimizer.")

        logger.info(f"Resumed at epoch {self.start_epoch}, best {state.best_metric_name}={state.best_metric:.4f}")

    # ------------------------------------------------------------------
    # Encoder freeze/unfreeze transition
    # ------------------------------------------------------------------

    def _transition_to_unfrozen(self) -> None:
        """Unfreeze encoder and rebuild optimizer with differential LR."""
        logger.info("Unfreezing encoder — switching to differential learning rates.")
        unfreeze_encoder(self.model)
        self._setup_optimizer_unfrozen()

        # Recreate scheduler for remaining epochs
        remaining = self.config.training.epochs - self.config.training.freeze_encoder_epochs
        if self.config.scheduler.type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=remaining,
                eta_min=self.config.scheduler.params.get("eta_min", 1e-6),
            )

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, float]:
        """Run the full training loop. Returns the best validation metrics."""
        logger.info(
            f"Starting training: {self.config.training.epochs} epochs, "
            f"device={self.device}, AMP={self.use_amp}"
        )

        for epoch in range(self.start_epoch, self.config.training.epochs):
            # Phase transition
            if epoch == self.config.training.freeze_encoder_epochs and epoch > 0:
                self._transition_to_unfrozen()

            t0 = time.time()
            train_metrics = self._train_one_epoch(epoch)
            val_metrics = self._validate(epoch)
            epoch_time = time.time() - t0

            lr = self.optimizer.param_groups[0]["lr"]
            record = {
                "epoch": epoch,
                "train_loss": train_metrics["train_loss"],
                "val_loss": val_metrics["val_loss"],
                "val_iou": val_metrics["val_iou"],
                "val_dice": val_metrics["val_dice"],
                "val_precision": val_metrics["val_precision"],
                "val_recall": val_metrics["val_recall"],
                "lr": lr,
                "epoch_time_s": epoch_time,
            }
            self.history.append(record)

            # Logging
            self._log_epoch(record)

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    monitor = val_metrics[self.config.training.early_stopping_metric]
                    self.scheduler.step(monitor)
                elif not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()

            # Checkpointing
            self._save_checkpoints(epoch, val_metrics)

            # Visualization
            if self._should_visualize(epoch):
                self._generate_visualizations(epoch)

            # Early stopping
            if self.early_stopping is not None:
                monitor = val_metrics[self.config.training.early_stopping_metric]
                if self.early_stopping(monitor):
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break

        self._save_final_artifacts()
        return self._get_best_metrics()

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        n_batches = 0
        grad_accum = self.config.training.grad_accumulation_steps

        self.optimizer.zero_grad()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config.training.epochs - 1} [train]",
            leave=False,
        )

        for step, batch in enumerate(pbar):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with autocast(device_type=self.amp_device_type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.loss_fn(logits, masks) / grad_accum

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(self.train_loader):
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

                # OneCycleLR steps per batch
                if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()

            if self.ema is not None:
                self.ema.update(self.model)

            running_loss += loss.item() * grad_accum
            n_batches += 1
            pbar.set_postfix({"loss": f"{running_loss / n_batches:.4f}"})

        return {"train_loss": running_loss / max(n_batches, 1)}

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        eval_model = self.ema.module if self.ema is not None else self.model
        eval_model.eval()

        self.metric_tracker.reset()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch}/{self.config.training.epochs - 1} [val]",
            leave=False,
        )

        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with autocast(device_type=self.amp_device_type, enabled=self.use_amp):
                logits = eval_model(images)
                loss = self.loss_fn(logits, masks)

            running_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits)
            self.metric_tracker.update(probs, masks)

        metrics = self.metric_tracker.compute()
        metrics["val_loss"] = running_loss / max(n_batches, 1)

        return {f"val_{k}" if not k.startswith("val_") else k: v for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_epoch(self, record: Dict[str, float]) -> None:
        epoch = int(record["epoch"])
        logger.info(
            f"Epoch {epoch:>3d} | "
            f"train_loss={record['train_loss']:.4f} | "
            f"val_loss={record['val_loss']:.4f} | "
            f"val_iou={record['val_iou']:.4f} | "
            f"val_dice={record['val_dice']:.4f} | "
            f"lr={record['lr']:.2e} | "
            f"{record['epoch_time_s']:.1f}s"
        )

        # CSV logging
        csv_path = self.log_dir / "metrics.csv"
        write_header = not csv_path.exists() or epoch == 0
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in record.items()})

        # TensorBoard
        if self.tb_writer is not None:
            for key, value in record.items():
                if key != "epoch" and isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, epoch)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoints(self, epoch: int, val_metrics: Dict[str, float]) -> None:
        metric_name = self.config.training.early_stopping_metric
        current = val_metrics.get(metric_name, val_metrics.get(f"val_{metric_name}", 0))

        is_best = False
        if self.config.training.early_stopping_mode == "max":
            is_best = current > self.best_metric
        else:
            is_best = current < self.best_metric

        if is_best:
            self.best_metric = current

        ckpt_kwargs = dict(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            best_metric=self.best_metric,
            best_metric_name=metric_name,
            config_dict=config_to_dict(self.config),
            ema_model=self.ema.shadow if self.ema else None,
        )

        if self.config.checkpoint.save_last:
            save_checkpoint(self.ckpt_dir / "last.pth", **ckpt_kwargs)

        if self.config.checkpoint.save_best and is_best:
            save_checkpoint(self.ckpt_dir / "best.pth", **ckpt_kwargs)
            logger.info(f"New best {metric_name}: {current:.4f}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _should_visualize(self, epoch: int) -> bool:
        freq = self.config.logging.save_visualizations_every_n_epochs
        return freq > 0 and (epoch % freq == 0 or epoch == self.config.training.epochs - 1)

    @torch.no_grad()
    def _generate_visualizations(self, epoch: int) -> None:
        if self._viz_batch is None:
            return

        eval_model = self.ema.module if self.ema is not None else self.model
        eval_model.eval()

        images = self._viz_batch["image"].to(self.device)
        masks_gt = self._viz_batch["mask"]

        with autocast(device_type=self.amp_device_type, enabled=self.use_amp):
            logits = eval_model(images)
        probs = torch.sigmoid(logits).cpu()

        save_path = self.viz_dir / f"epoch_{epoch:04d}.png"
        plot_prediction_samples(
            images=self._viz_batch["image"],
            masks_gt=masks_gt,
            masks_pred=probs,
            mean=self.config.normalization.mean,
            std=self.config.normalization.std,
            save_path=save_path,
        )
        plt.close("all")
        logger.info(f"Visualizations saved: {save_path}")

    # ------------------------------------------------------------------
    # Final artifacts
    # ------------------------------------------------------------------

    def _save_final_artifacts(self) -> None:
        if self.config.logging.save_training_curves and self.history:
            curves_path = self.log_dir / "training_curves.png"
            plot_training_curves(self.history, save_path=curves_path)
            plt.close("all")
            logger.info(f"Training curves saved: {curves_path}")

        if self.tb_writer is not None:
            self.tb_writer.close()

        best = self._get_best_metrics()
        logger.info(
            f"Training complete. Best val_iou={best.get('val_iou', 0):.4f}, "
            f"val_dice={best.get('val_dice', 0):.4f}"
        )

    def _get_best_metrics(self) -> Dict[str, float]:
        if not self.history:
            return {}
        metric = self.config.training.early_stopping_metric
        if self.config.training.early_stopping_mode == "max":
            best_record = max(self.history, key=lambda h: h.get(metric, 0))
        else:
            best_record = min(self.history, key=lambda h: h.get(metric, float("inf")))
        return best_record

