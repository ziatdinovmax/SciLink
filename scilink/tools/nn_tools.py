import os
import time
from dataclasses import dataclass
from abc import abstractmethod
from typing import List, Tuple, Union, Callable, Any, Type, TypeVar, Optional, Dict, Any, Callable
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor as Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

# TODO: add shift and rotation invarient VAEs

#=================================
# Utils definitions
#=================================
logger = logging.getLogger(__name__)

pooling = {
    'max': lambda d : nn.MaxPool1d if d == 1 else nn.MaxPool2d,
    'avg': lambda d : nn.AvgPool1d if d == 1 else nn.AvgPool2d,
    'adaptMax': lambda d : nn.AdaptiveMaxPool1d if d == 1 else nn.AdaptiveMaxPool2d,
    'adaptAvg': lambda d : nn.AdaptiveAvgPool1d if d == 1 else nn.AdaptiveAvgPool2d,
}

activations = {
    'relu': nn.ReLU(),
    'lrelu': lambda a : nn.LeakyReLU(negative_slope = a),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh()
}

def get_activation(activation_fn, leaky_negative_slope):
    if activation_fn == 'lrelu':
        return activations[activation_fn](leaky_negative_slope)
    else:
        return activations[activation_fn]
    
def CNN_output_size(in_dims, kernel, padding, stride):
    output_size = ((in_dims-kernel+(2*padding))//stride)+1
    return output_size

def make_deterministic(seed: int = 1):
    """
    For reproducibility. Sets the manual seed of rng for numpy, torch, and cuda. Sets torch cudnn to deterministic (not benchmark).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def resolve_training_device(device):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    return device

#=================================
# Neural network block definitions
#=================================
class NetBlock(nn.Module):
    def __init__(self,
                 n_dims: int,
                 n_layers: int = 1,
                 activation_fn: str = 'lrelu',
                 pooling_type: str = None,
                 dropout_prob: float = 0.0,
                 leaky_negative_slope: float = 0.01,
                 batch_norm: bool = False,
                 ) -> None:
        super().__init__()
        self.n_dims = n_dims
        self.n_layers = n_layers
        self.pooling_type = pooling_type
        self.batch_norm = batch_norm
        self.dropout_prob = dropout_prob
        self.activation = get_activation(activation_fn, leaky_negative_slope)
        self.block = []

    def append_pooling(self):
        pooling_fn = pooling[self.pooling_type](self.n_dims)
        pooling_kernel = 3
        self.block.append(pooling_fn(pooling_kernel))

    def append_dropout(self):
        self.block.append(nn.Dropout(self.dropout_prob))
    
    def append_batchnorm(self, n_features):
        bn = nn.BatchNorm1d if self.n_dims == 1 else nn.BatchNorm2d
        self.block.append(bn(n_features))

    def append_pooling_dropout_activation_batchnorm(self, out_features):
        if self.pooling_type is not None: 
            self.append_pooling()
        
        if self.dropout_prob > 0.0: 
            self.append_dropout()

        self.block.append(self.activation)

        if self.batch_norm: 
            self.append_batchnorm(out_features)

    @abstractmethod
    def forward(self, *x: Tensor) -> Tensor:
        pass
    
class LinearBlock(NetBlock):
    def __init__(self,                  
                 in_features: int,
                 out_features: int = None,
                 *args, **kwargs) -> None:
        super().__init__(n_dims=1, *args, **kwargs)

        if out_features is None:
            out_features = in_features

        for i in range(self.n_layers):
            self.block.append(nn.Linear(in_features, out_features))
            in_features = out_features

        self.append_pooling_dropout_activation_batchnorm(out_features)
        self.block = nn.Sequential(*self.block)

    def forward(self, x: Tensor) -> Tensor:
        output = self.block(x)
        return output

class ConvBlock(NetBlock):
    def __init__(self, 
                 n_dims: int,
                 in_channels: int = 1,
                 out_channels: int = 16,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 *args, **kwargs) -> None:
        super().__init__(n_dims=n_dims, *args, **kwargs)
        conv = nn.Conv1d if self.n_dims == 1 else nn.Conv2d
        self.output_size = None

        for i in range(self.n_layers):
            self.block.append(
                conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
            
            in_channels = out_channels

        self.append_pooling_dropout_activation_batchnorm(out_channels)
        self.block = nn.Sequential(*self.block)

    def calc_output_size(self, in_dims):
        next_input = in_dims[0]

        for layer in self.block:
            # print(layer)
            next_input = CNN_output_size(next_input, self.kernel, self.padding, self.stride)

        if self.pooling_type is not None:
            next_input = CNN_output_size(next_input, 3, 0, 3)

        self.output_size = next_input
        return self.output_size

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        output = self.block(x)
        return output

class ConvTransposeBlock(NetBlock):
    def __init__(self, 
                 n_dims: int,
                 in_channels: int = 16,
                 out_channels: int = 1,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 *args, **kwargs) -> None:
        super().__init__(n_dims=n_dims, *args, **kwargs)
        convTranspose = nn.ConvTranspose1d if self.n_dims == 1 else nn.ConvTranspose2d

        for i in range(self.n_layers):
            self.block.append(
                convTranspose(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
            )
            in_channels = out_channels

        self.append_pooling_dropout_activation_batchnorm(out_channels)
        self.block = nn.Sequential(*self.block)

    def forward(self, x: Tensor) -> Tensor:
        output = self.block(x)
        return output
    
class UpSampleBlock(NetBlock):
    def __init__(self,
                 n_dims: int,
                 in_channels: int,
                 out_channels: int,
                 scale_factor: int = 2,
                 mode: str = 'bilinear',
                 *args, **kwargs) -> None:
        super().__init__(n_dims=n_dims, *args, **kwargs)
        self.scale_factor = scale_factor
        self.mode = mode if self.n_dims == 2 else "nearest"
        conv = nn.Conv1d if self.n_dims == 1 else nn.Conv2d
        self.conv_1x1 = conv(
            in_channels, 
            out_channels,
            kernel_size=1, 
            stride=1, 
            padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        output = self.conv_1x1(x)
        return output
    
#=================================
# Encoder definitions
#=================================
class Encoder(nn.Module):
    def __init__(self, 
                 in_dims: tuple[int, ...],
                 block_configs: List[dict] = [{}],
                 skip: bool = False,
                 softplus: bool = False) -> None:
        super().__init__()
        self.in_dims = in_dims
        self.block_configs = block_configs
        self.n_blocks = len(block_configs)
        self.in_dims = in_dims
        self.skip = skip
        self.softplus = softplus
        self.encoder = []

    @abstractmethod
    def forward(self, x: Tensor) -> None:
        pass
    
class LinearEncoder(Encoder):
    default_config = [{
        'n_layers': 1,
        'in_features': 20,
        'out_features': 4,
        'batch_norm': False,
        'dropout_prob': 0.0,
        'pooling_type': None,
        'activation_fn': 'lrelu',
        'leaky_negative_slope': 0.001
    }]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.blocks = [LinearBlock(**self.block_configs[i]) for i in range(self.n_blocks)]   
        self.encoder_blocks = nn.Sequential(*self.blocks)

    def forward(self, x: Tensor) -> Tensor:
        output = self.encoder_blocks(x)
        return output
    
class ConvEncoder(Encoder):
    default_config = [{
        'in_channels': 1,
        'out_channels': 16,
        'n_layers': 1,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'batch_norm': True,
        'dropout_prob': 0.2,
        'pooling_type': 'avg',
        'activation_fn': 'lrelu',
        'leaky_negative_slope': 0.01
    }]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.blocks = [ConvBlock(n_dims=len(self.in_dims), **self.block_configs[i]) for i in range(self.n_blocks)]        
        self.encoder_blocks = nn.Sequential(*self.blocks)

    def calc_output_size(self):
            output_size = 0
            in_dims = self.in_dims[0]

            for block_config in self.block_configs:
                kernel = block_config['kernel_size']
                padding = block_config['padding']
                stride = block_config['stride']
                output_size = CNN_output_size(in_dims, kernel, padding, stride)
                in_dims = output_size

                if 'pooling_type' in block_config:
                    if block_config['pooling_type'] is not None:
                        # pooling layer params are hard coded to kernel size 3, padding 0, stride 3
                        output_size = CNN_output_size(in_dims, 3, 0, 3)
                        in_dims = output_size

            return output_size
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder_blocks(x)
        return x
    
class LinearVariationalEncoder(LinearEncoder):
    def __init__(self, n_latent: int = 2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_latent = n_latent
        self.reshape = self.block_configs[-1]['out_features']
        self.fc_mu = nn.Linear(self.reshape, n_latent)
        self.fc_logvar = nn.Linear(self.reshape, n_latent) 

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder_blocks(x)
        z_mean = self.fc_mu(x)
        z_logvar = self.fc_logvar(x)
        return z_mean, z_logvar
    
class ConvVariationalEncoder(ConvEncoder):
    def __init__(self, 
                 n_latent: int = 2,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_latent = n_latent
        self.flatten = nn.Flatten()

        if len(self.in_dims) == 1:
            self.reshape = self.block_configs[-1]['out_channels'] * self.calc_output_size()
        else:
            self.reshape = self.block_configs[-1]['out_channels'] * self.calc_output_size()**2

        self.fc_mu = nn.Linear(self.reshape, n_latent)
        self.fc_logvar = nn.Linear(self.reshape, n_latent) 

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder_blocks(x)
        x = self.flatten(x)
        z_mean, z_log_var = self.fc_mu(x), self.fc_logvar(x)
        return z_mean, z_log_var

#=================================
# Decoder definitions
#=================================
class Decoder(nn.Module):
    def __init__(self, 
                 out_dims: Tuple[int],
                 n_layers: List[int] = [2],
                 n_hidden: List[int] = [32],
                 n_latent: int = 2,
                 skip: bool = False,
                 softplus: bool = False,
                 block_configs: List[dict] = [{}],
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block_configs = block_configs
        self.n_blocks = len(block_configs)
        self.out_dims = out_dims
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.skip = skip
        self.softplus = softplus
        self.decoder = []

    @abstractmethod
    def forward(self, *x: Tensor) -> None:
        pass
    
class LinearDecoder(Decoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.blocks = [LinearBlock(**self.block_configs[i]) for i in range(self.n_blocks)]   
        self.decoder_blocks = nn.Sequential(*self.blocks)

    def forward(self, z: Tensor) -> Tensor:
        output = self.decoder_blocks(z)
        return output

class ConvDecoder(Decoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.blocks = [ConvTransposeBlock(n_dims=len(self.out_dims), **self.block_configs[i]) for i in range(self.n_blocks)]        
        self.decoder_blocks = nn.Sequential(*self.blocks)

    def forward(self, z: Tensor) -> Tensor:
        output = self.decoder_blocks(z)
        return output

class LinearVariationalDecoder(LinearDecoder):
    def __init__(self, n_latent: int = 2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_latent = n_latent
        self.decoder_input = nn.Linear(n_latent, self.block_configs[0]['in_features'])

    def forward(self, z: Tensor) -> Tensor:
        z = self.decoder_input(z)
        reconstruciton = self.decoder_blocks(z)
        return reconstruciton
    
class ConvVariationalDecoder(ConvDecoder):
    def __init__(self, n_latent: int = 2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_latent = n_latent
        self.decoder_input = nn.Linear(n_latent, self.block_configs[0]['in_channels'] * np.prod(self.out_dims[:2]))
        self.unflatten = nn.Unflatten(1, unflattened_size=(self.block_configs[0]['in_channels'], *self.out_dims[:2]))   
    
    def forward(self, z: Tensor) -> Tensor:
        z = self.decoder_input(z)
        z = self.unflatten(z)
        reconstruction = self.decoder_blocks(z)
        return reconstruction
    
#=================================
# Trainer definitions
#=================================
@dataclass
class TrainerConfig:
    max_epochs: int = 10
    batch_size: int = 1
    device: Optional[str] = None  # 'cuda', 'cpu', 'mps', or None to auto-select
    gradient_clip_val: Optional[float] = None
    log_every_n_steps: int = 50
    validate_every_n_epochs: int = 1
    checkpoint_dir: Optional[str] = None
    checkpoint_monitor: Optional[str] = "val_loss"  # metric key to monitor for best checkpoint
    checkpoint_mode: str = "min"  # 'min' or 'max'
    accumulate_grad_batches: int = 1
    detect_anomaly: bool = False
    precision: str = "32"  # "32" or "bf16" (requires Amp on supported GPUs) or "16" (amp)
    seed: int = 1
    # Note: Simplified precision handling. Extend as needed.

class Trainer:
    """
    A simplified trainer inspired by PyTorch Lightning.
    Handles training/validation loops, logging, checkpointing, and device placement.
    """
    def __init__(self, 
                 config: TrainerConfig,
                 optimizer: Optimizer,
                 save_path: str = "model") -> None:
        self.cfg = config
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = None
        self.best_ckpt_path = None
        self.optimizer = optimizer
        self.loss_history = {}
        self.filename = save_path
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.device = resolve_training_device(self.cfg.device)
        make_deterministic(self.cfg.seed)

        if self.cfg.checkpoint_dir:
            os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

    def _move_batch_to_device(self, batch):
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._move_batch_to_device(x) for x in batch)
        if isinstance(batch, dict):
            return {k: self._move_batch_to_device(v) for k, v in batch.items()}
        if torch.is_tensor(batch):
            return batch.to(self.device, non_blocking=True)
        return batch

    def _log(self, msg: str):
        logger.info(msg, flush=True)

    def _save_checkpoint(self, model: nn.Module, optimizer: Optimizer, epoch: int, metrics: Dict[str, Any], filename: str):
        if not self.cfg.checkpoint_dir:
            return
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "config": self.cfg.__dict__,
        }
        path = os.path.join(self.cfg.checkpoint_dir, filename)
        torch.save(ckpt, path)
        return path

    def _update_best_checkpoint(self, metrics: Dict[str, Any], model: nn.Module, optimizer: Optimizer, epoch: int):
        if not self.cfg.checkpoint_dir or not self.cfg.checkpoint_monitor:
            return
        key = self.cfg.checkpoint_monitor
        if key not in metrics:
            return
        current = metrics[key]
        better = False
        if self.best_metric is None:
            better = True
        else:
            if self.cfg.checkpoint_mode == "min":
                better = current < self.best_metric
            else:
                better = current > self.best_metric

        if better:
            self.best_metric = current
            self.best_ckpt_path = self._save_checkpoint(
                model, optimizer, epoch, metrics, filename="best.ckpt"
            )
            self._log(f"[Checkpoint] New best {key}={current:.5f} at epoch {epoch}. Saved to {self.best_ckpt_path}")

    def save_model(self, filepath: str = None):
        if filepath is None:
            filepath = self.filename
        
        torch.save(self.model.state_dict(), filepath+".pt")

    def load_model(self, filepath: str = None):
        if filepath is None:
            filepath = self.filename
        
        weights = torch.load(filepath, weights_only=True, map_location=self.device)
        self.model.load_state_dict(weights)
        # assuming loading for inference, 
        # call eval() to set any dropout/batch norm layers to eval mode
        self.model.eval()

    def train_step(self, model: nn.Module, batch):
        # Clear old gradients
        # modestly improve performance and reduce memory fragmentation by setting gradients to None rather than a zero tensor
        self.optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        output = model(batch)
        loss = model.loss_function(*output)

        # Backward pass (compute gradients)
        if not isinstance(loss, dict) or "loss" not in loss:
            raise ValueError("training_step must return a dict with a 'loss' key.")
        
        loss_for_backward = loss["loss"]
        loss_for_backward.backward()

        # Optimizer step (update parameters)
        self.optimizer.step()

        return loss

    def train_epoch(self, model: nn.Module, train_loader: DataLoader):
        model.train()
        training_history = {}
        training_steps = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            batch = self._move_batch_to_device(batch)
            loss = self.train_step(model, batch)

            # Mid-Training Logging
            for k, v in loss.items():
                if torch.is_tensor(v):
                    v = v.detach().item()

                training_history[k] = training_history.get(k, 0.0) + (v if isinstance(v, (int, float)) else 0.0)
            
            if training_steps % self.cfg.log_every_n_steps == 0:
                log_msg = f"Epoch {self.current_epoch} | step {training_steps} | "
                log_msg += " | ".join([f"{k}: {training_history[k]/max(1, training_steps):.4f}" for k in training_history if isinstance(training_history[k], (int, float))])
                self._log(log_msg)
            
            training_steps += 1

        # End of epoch logging
        epoch_metrics = {k: training_history[k] / max(1, training_steps) for k in training_history if isinstance(training_history[k], (int, float))}
        epoch_time = time.time() - epoch_start
        self._log(f"Epoch {self.current_epoch} done in {epoch_time:.1f}s | " +
                    " | ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()]))
        
        return epoch_metrics

    def val_step(self, model: nn.Module, batch):    
        output = model(batch)
        loss = model.loss_function(*output)
        return loss
    
    def val_epoch(self, model: nn.Module, val_loader: DataLoader):
        model.eval()
        val_metrics = {} 
        val_history = {}
        val_steps = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = self._move_batch_to_device(batch)
                val_loss = self.val_step(model, batch)

                if isinstance(val_loss, dict):
                    for k, v in val_loss.items():
                        if torch.is_tensor(v):
                            v = v.detach().item()
                        if isinstance(v, (int, float)):
                            val_history[k] = val_history.get(k, 0.0) + v

                val_steps += 1

        if val_steps > 0:
            val_metrics = {f'val_{k}': val_history[k] / val_steps for k in val_history}

        self._log("Validation | " + " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()]))

        return val_metrics

    def fit(self, model: nn.Module, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> List[Dict]:
        model = model.to(self.device)
        scheduler = None
        epoch_metrics, val_metrics = {}, {}
        train_history: List[Dict] = []

        # If the model defines configure_optimizers(), use its optimizer for training
        # and extract an optional scheduler. Falls back to the optimizer passed to __init__.
        if hasattr(model, "configure_optimizers"):
            optim_config = model.configure_optimizers()
            # Support multiple return types: optimizer, (optimizer, scheduler), list
            if isinstance(optim_config, (list, tuple)):
                if len(optim_config) == 2 and not isinstance(optim_config[0], (list, tuple)):
                    self.optimizer, scheduler = optim_config
                else:
                    self.optimizer = optim_config[0]
                    if len(optim_config) > 1:
                        scheduler = optim_config[1]
            elif optim_config is not None:
                self.optimizer = optim_config
        optimizer = self.optimizer  # local alias for checkpoint saving

        # Optional anomaly detection
        torch.autograd.set_detect_anomaly(self.cfg.detect_anomaly)

        for epoch in range(self.cfg.max_epochs):
            self.current_epoch = epoch

            # Training
            epoch_metrics = self.train_epoch(model, train_loader)

            # Validation
            if val_loader is not None and ((epoch + 1) % self.cfg.validate_every_n_epochs == 0):
                val_metrics = self.val_epoch(model, val_loader)

                # Save best checkpoint
                self._update_best_checkpoint({**epoch_metrics, **val_metrics}, model, optimizer, epoch)

            # Per-epoch checkpoint (optional)
            if self.cfg.checkpoint_dir:
                self._save_checkpoint(model, optimizer, epoch, {**epoch_metrics, **val_metrics}, filename=f"epoch_{epoch}.ckpt")

            # Step LR scheduler per epoch if provided
            if scheduler is not None:
                try:
                    # Support metrics-aware schedulers like ReduceLROnPlateau
                    if hasattr(scheduler, "step"):
                        if "val_loss" in val_metrics:
                            scheduler.step(val_metrics["val_loss"])
                        else:
                            scheduler.step()
                except Exception as e:
                    self._log(f"[Scheduler] step failed: {e}")

            self.loss_history = {**epoch_metrics, **val_metrics}
            train_history.append({"epoch": epoch, **epoch_metrics, **val_metrics})

        if self.best_ckpt_path:
            self._log(f"Best checkpoint: {self.best_ckpt_path} (metric: {self.best_metric})")

        self.save_model(self.filename)
        self._log("Training complete.")
        return train_history