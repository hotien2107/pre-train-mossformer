import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# Custom modules
from utils.dataset import SpeechSeparationDataset, DatasetConfig
from utils.loss import SI_SDRLoss
from utils.metric import SeparationMetrics, MetricTracker

class Mossformer2Trainer:
    def __init__(
            self,
            model_name: str,
            dataset_config: DatasetConfig,
            save_dir: str = "checkpoints",
            # Training parameters
            epochs: int = 100,
            batch_size: int = 16,
            lr: float = 3e-4,
            weight_decay: float = 1e-2,
            grad_norm: float = 5.0,
            grad_accum_steps: int = 4,
            # Device settings
            num_gpus: int = None,
            # Early stopping
            early_stop_patience: int = 15,
            # Checkpoint management
            max_checkpoints: int = 3
    ):
        self.device = self._setup_device()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Training hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_norm = grad_norm
        self.grad_accum_steps = grad_accum_steps
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.max_checkpoints = max_checkpoints

        # Initialize components
        self.dataset_config = dataset_config
        self._init_datasets()
        self._init_model(model_name)
        self._init_training(lr, weight_decay, early_stop_patience)

    def _setup_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        return device

    def _init_datasets(self):
        """Initialize datasets with validation"""
        try:
            self.train_set = SpeechSeparationDataset(
                split='train',
                config=self.dataset_config
            )
            self.valid_set = SpeechSeparationDataset(
                split='validation',
                config=self.dataset_config
            )
        except KeyError as e:
            raise ValueError(f"Invalid dataset split: {e}") from e

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=SpeechSeparationDataset.collate_fn,
            persistent_workers=True
        )

        self.valid_loader = DataLoader(
            self.valid_set,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            collate_fn=SpeechSeparationDataset.collate_fn,
            persistent_workers=True
        )

    def _init_model(self, model_name):
        """Initialize model with multi-GPU support"""
        separation_pipe = pipeline(
            Tasks.speech_separation,
            model=model_name,
            device=0 if self.device.type == 'cuda' else -1
        )

        if not hasattr(separation_pipe, 'model'):
            raise ValueError("Invalid pipeline structure - missing 'model' attribute")

        self.model = separation_pipe.model.to(self.device)

        # Multi-GPU setup
        if torch.cuda.device_count() > 1 and self.num_gpus > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)

    def _init_training(self, lr, weight_decay, early_stop_patience):
        """Initialize training components"""
        self.global_step = 0
        self.loss_fn = SI_SDRLoss().to(self.device)
        self.metric_fn = SeparationMetrics().to(self.device)
        self.metric_tracker = MetricTracker()

        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=len(self.train_loader) // self.grad_accum_steps,
            eta_min=1e-6
        )

        # Early stopping and logging
        self.early_stopper = EarlyStopper(patience=early_stop_patience)
        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, 'logs'))

    def _save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint with smart management"""
        state_dict = self.model.module.state_dict() if isinstance(
            self.model, nn.DataParallel) else self.model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state': state_dict,
            'optim_state': self.optimizer.state_dict(),
            'sched_state': self.scheduler.state_dict(),
            'best_loss': self.early_stopper.best_loss
        }

        # Save best checkpoint
        if is_best:
            filename = 'mossformer2_best.pt'
            torch.save(checkpoint, os.path.join(self.save_dir, filename))
            return

        # Save regular checkpoint with rotation
        filename = f'mossformer2_epoch{epoch:03d}.pt'
        torch.save(checkpoint, os.path.join(self.save_dir, filename))

        # Remove old checkpoints
        checkpoints = sorted(
            [f for f in os.listdir(self.save_dir)
             if f.startswith("mossformer2_epoch")],
            key=lambda x: os.path.getmtime(os.path.join(self.save_dir, x))
        )
        while len(checkpoints) > self.max_checkpoints:
            os.remove(os.path.join(self.save_dir, checkpoints[0]))
            checkpoints.pop(0)

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint to resume training"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint['model_state'])

        # Load training state
        self.optimizer.load_state_dict(checkpoint['optim_state'])
        self.scheduler.load_state_dict(checkpoint['sched_state'])
        self.early_stopper.best_loss = checkpoint['best_loss']
        self.global_step = checkpoint['global_step']

        return checkpoint['epoch']

    def _train_epoch(self, epoch):
        """Training loop with proper gradient accumulation"""
        self.model.train()
        self.metric_tracker.reset()
        total_raw_loss = 0.0
        self.optimizer.zero_grad()

        with tqdm(self.train_loader, desc=f'Train Epoch {epoch}') as pbar:
            for step, batch in enumerate(pbar):
                mix = batch['mix'].to(self.device, non_blocking=True)
                sources = batch['sources'].to(self.device, non_blocking=True)

                # Forward pass
                est_sources = self.model(mix)
                raw_loss = self.loss_fn(est_sources, sources)
                loss = raw_loss / self.grad_accum_steps

                # Backward pass
                loss.backward()
                total_raw_loss += raw_loss.item()

                # Gradient accumulation step
                if (step + 1) % self.grad_accum_steps == 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # Update global step
                    self.global_step += 1

                    # Logging
                    current_loss = total_raw_loss / (step + 1)
                    pbar.set_postfix({'loss': current_loss})
                    self.writer.add_scalar(
                        'train/lr',
                        self.scheduler.get_last_lr()[0],
                        self.global_step
                    )

        # Calculate epoch metrics
        avg_loss = total_raw_loss / len(self.train_loader)
        self.writer.add_scalar('train/loss', avg_loss, epoch)
        return avg_loss

    def _validate(self, epoch):
        """Validation loop with proper multi-GPU handling"""
        self.model.eval()
        self.metric_tracker.reset()

        with torch.no_grad(), tqdm(self.valid_loader, desc=f'Valid Epoch {epoch}') as pbar:
            for batch in pbar:
                mix = batch['mix'].to(self.device)
                sources = batch['sources'].to(self.device)

                # Synchronize GPUs and forward pass
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                est_sources = self.model(mix)

                # Calculate metrics
                batch_metrics = self.metric_fn(est_sources, sources)
                self.metric_tracker.update(batch_metrics)
                pbar.set_postfix(self.metric_tracker.get_means())

        # Log validation metrics
        final_metrics = self.metric_tracker.get_means()
        for metric, value in final_metrics.items():
            self.writer.add_scalar(f'valid/{metric}', value, epoch)

        return final_metrics['sisdr']

    def train(self):
        """Main training loop"""
        best_metric = -np.inf

        try:
            for epoch in range(1, self.epochs + 1):
                # Train and validate
                train_loss = self._train_epoch(epoch)
                val_metric = self._validate(epoch)

                # Save checkpoints
                self._save_checkpoint(epoch)
                if val_metric > best_metric:
                    self._save_checkpoint(epoch, is_best=True)
                    best_metric = val_metric
                    print(f"New best model at epoch {epoch} ({val_metric:.2f} dB)")

                # Early stopping check
                if self.early_stopper.should_stop(-val_metric):
                    print(f"Early stopping at epoch {epoch}")
                    break

        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving latest checkpoint...")
            self._save_checkpoint(epoch)

        finally:
            self.writer.close()

class EarlyStopper:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf

    def should_stop(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

if __name__ == '__main__':
    # Configuration
    dataset_config = DatasetConfig(
        dataset_name="speech-separation-benchmark/wsj0-2mix",
        sample_rate=8000,
        max_duration=5.0,
        num_sources=2
    )

    trainer = Mossformer2Trainer(
        model_name='damo/speech_mossformer2_separation_temporal_8k',
        dataset_config=dataset_config,
        save_dir='checkpoints/mossformer2',
        epochs=100,
        batch_size=16,
        lr=5e-4,
        grad_accum_steps=4,
        num_gpus=torch.cuda.device_count(),
        max_checkpoints=3
    )

    # To resume training from checkpoint:
    # trainer.load_checkpoint("checkpoints/mossformer2/mossformer2_epoch010.pt")
    trainer.train()