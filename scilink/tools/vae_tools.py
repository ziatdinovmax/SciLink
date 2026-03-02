from abc import abstractmethod
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from scilink.tools.nn_tools import (
    LinearEncoder,
    LinearDecoder,
    ConvEncoder,
    ConvDecoder,
    LinearVariationalEncoder,
    LinearVariationalDecoder,
    ConvVariationalEncoder,
    ConvVariationalDecoder,
)

#=================================
# Base Autoencoder and Variational Autoencoder
#=================================
class BaseAE(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def reconstruct(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

class BaseVAE(BaseAE):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def reparameterize(self):
        pass

# =============================================================================
# Model 1: Vanilla Linear Autoencoder (no KL divergence)
# =============================================================================
class LinearAE(BaseAE):
    """
    Standard (deterministic) autoencoder with fully-connected encoder and decoder.
    Uses reconstruction loss only — no variational bottleneck or KL term.
    Suitable for compression / reconstruction tasks on 1-D data.
    """

    default_encoder_config = [{
        'n_layers': 2,
        'in_features': 64,
        'out_features': 32,
        'batch_norm': False,
        'dropout_prob': 0.0,
        'pooling_type': None,
        'activation_fn': 'lrelu',
        'leaky_negative_slope': 0.01,
    }]

    default_decoder_config = [{
        'n_layers': 2,
        'in_features': 32,
        'out_features': 64,
        'batch_norm': False,
        'dropout_prob': 0.0,
        'pooling_type': None,
        'activation_fn': 'lrelu',
        'leaky_negative_slope': 0.01,
    }]

    def __init__(self,
                 in_dims: Tuple[int, ...],
                 latent_dims: int = 8,
                 encoder_configs: Optional[List[dict]] = None,
                 decoder_configs: Optional[List[dict]] = None,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_dims = in_dims
        self.latent_dims = latent_dims
        self._lr = lr
        self._weight_decay = weight_decay

        enc_cfg = encoder_configs or LinearAE.default_encoder_config
        dec_cfg = decoder_configs or LinearAE.default_decoder_config

        enc_cfg[0]['in_features'] = in_dims[0]
        dec_cfg[-1]['out_features'] = in_dims[0]

        self.encoder = LinearEncoder(in_dims=in_dims, block_configs=enc_cfg)
        # Bottleneck projection
        self.fc_bottleneck = nn.Linear(enc_cfg[-1]['out_features'], latent_dims)
        self.fc_expand = nn.Linear(latent_dims, dec_cfg[0]['in_features'])
        self.decoder = LinearDecoder(out_dims=in_dims, block_configs=dec_cfg)

    def encode(self, x: Tensor) -> Tensor:
        h = self.encoder(x)
        return self.fc_bottleneck(h)

    def decode(self, z: Tensor) -> Tensor:
        h = self.fc_expand(z)
        return self.decoder(h)

    def forward(self, x: Tensor) -> List[Tensor]:
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction, x

    def sample(self, num_samples: int, current_device) -> Tensor:
        z = torch.randn(num_samples, self.latent_dims).to(current_device)
        return self.decode(z)

    def reconstruct(self, x: Tensor) -> Tensor:
        return self.forward(x)[0]

    def loss_function(self, recons: Tensor, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        recons_loss = F.mse_loss(recons, x)
        return {'loss': recons_loss, 'Reconstruction_Loss': recons_loss.detach()}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)


# =============================================================================
# Model 2: Vanilla Convolutional Autoencoder (no KL divergence)
# =============================================================================
class ConvAE(BaseAE):
    """
    Standard (deterministic) autoencoder with convolutional encoder and decoder.
    Uses reconstruction loss only — no variational bottleneck or KL term.
    Suitable for 1-D spectra or 2-D image compression/reconstruction.
    """

    default_encoder_config = [{
        'n_layers': 2,
        'in_channels': 1,
        'out_channels': 16,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'batch_norm': True,
        'dropout_prob': 0.0,
        'pooling_type': None,
        'activation_fn': 'lrelu',
        'leaky_negative_slope': 0.01,
    }]

    default_decoder_config = [{
        'n_layers': 2,
        'in_channels': 16,
        'out_channels': 1,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'batch_norm': True,
        'dropout_prob': 0.0,
        'activation_fn': 'lrelu',
        'leaky_negative_slope': 0.01,
    }]

    def __init__(self,
                 in_dims: Tuple[int, ...],
                 latent_dims: int = 16,
                 encoder_configs: Optional[List[dict]] = None,
                 decoder_configs: Optional[List[dict]] = None,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_dims = in_dims
        self.latent_dims = latent_dims
        self._lr = lr
        self._weight_decay = weight_decay

        enc_cfg = encoder_configs or ConvAE.default_encoder_config
        dec_cfg = decoder_configs or ConvAE.default_decoder_config

        self.encoder = ConvEncoder(in_dims=in_dims, block_configs=enc_cfg)

        # Compute flattened encoder output size for bottleneck FC layers
        n_dims = len(in_dims)
        out_spatial = self.encoder.calc_output_size()
        out_ch = enc_cfg[-1]['out_channels']
        self._flat_size = out_ch * (out_spatial ** n_dims)

        self.fc_bottleneck = nn.Linear(self._flat_size, latent_dims)
        self.fc_expand = nn.Linear(latent_dims, self._flat_size)
        self._unflatten_shape = (out_ch,) + (out_spatial,) * n_dims

        self.decoder = ConvDecoder(out_dims=in_dims, block_configs=dec_cfg)

    def encode(self, x: Tensor) -> Tensor:
        h = self.encoder(x)
        h = h.flatten(1)
        return self.fc_bottleneck(h)

    def decode(self, z: Tensor) -> Tensor:
        h = self.fc_expand(z)
        h = h.view(-1, *self._unflatten_shape)
        return self.decoder(h)

    def forward(self, x: Tensor) -> List[Tensor]:
        z = self.encode(x)
        reconstruction = self.decode(z)
        # ConvDecoder outputs (B, 1, H, W); squeeze the channel dim so that
        # reconstruction matches the (B, H, W) input for the loss function.
        if reconstruction.ndim == x.ndim + 1:
            reconstruction = reconstruction.squeeze(1)
        return reconstruction, x

    def sample(self, num_samples: int, current_device) -> Tensor:
        z = torch.randn(num_samples, self.latent_dims).to(current_device)
        return self.decode(z).squeeze(1)

    def reconstruct(self, x: Tensor) -> Tensor:
        return self.forward(x)[0]

    def loss_function(self, recons: Tensor, x: Tensor, **kwargs) -> Dict[str, Tensor]:
        recons_loss = F.mse_loss(recons, x)
        return {'loss': recons_loss, 'Reconstruction_Loss': recons_loss.detach()}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)


# =============================================================================
# Model 3: Linear Variational Autoencoder
# =============================================================================
class LinearVAE(BaseVAE):
    """
    Variational autoencoder with fully-connected encoder and decoder.
    Uses the reparameterization trick and an ELBO loss
    (reconstruction MSE + weighted KL divergence).
    Suitable for 1-D data where latent space exploration is the goal.
    """

    default_encoder_config = [{
        'n_layers': 2,
        'in_features': 64,
        'out_features': 32,
        'batch_norm': False,
        'dropout_prob': 0.0,
        'pooling_type': None,
        'activation_fn': 'lrelu',
        'leaky_negative_slope': 0.01,
    }]

    default_decoder_config = [{
        'n_layers': 2,
        'in_features': 32,
        'out_features': 64,
        'batch_norm': False,
        'dropout_prob': 0.0,
        'pooling_type': None,
        'activation_fn': 'lrelu',
        'leaky_negative_slope': 0.01,
    }]

    def __init__(self,
                 in_dims: Tuple[int, ...],
                 latent_dims: int = 8,
                 encoder_configs: Optional[List[dict]] = None,
                 decoder_configs: Optional[List[dict]] = None,
                 kld_weight: float = 1.0,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_dims = in_dims
        self.latent_dims = latent_dims
        self.kld_weight = kld_weight
        self._lr = lr
        self._weight_decay = weight_decay

        enc_cfg = encoder_configs or LinearVAE.default_encoder_config
        dec_cfg = decoder_configs or LinearVAE.default_decoder_config

        enc_cfg[0]['in_features'] = in_dims[0]
        dec_cfg[-1]['out_features'] = in_dims[0]

        self.encoder = LinearVariationalEncoder(
            in_dims=in_dims, n_latent=latent_dims, block_configs=enc_cfg
        )
        self.decoder = LinearVariationalDecoder(
            out_dims=in_dims, n_latent=latent_dims, block_configs=dec_cfg
        )

    def encode(self, x: Tensor) -> List[Tensor]:
        return self.encoder(x)  # (mu, log_var)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> List[Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, x, mu, log_var

    def sample(self, num_samples: int, current_device) -> Tensor:
        z = torch.randn(num_samples, self.latent_dims).to(current_device)
        return self.decode(z)

    def reconstruct(self, x: Tensor) -> Tensor:
        return self.forward(x)[0]

    def loss_function(self, recons: Tensor, x: Tensor,
                      mu: Tensor, log_var: Tensor, **kwargs) -> Dict[str, Tensor]:
        recons_loss = F.mse_loss(recons, x)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
        )
        loss = recons_loss + self.kld_weight * kld_loss
        return {
            'loss': loss,
            'Reconstruction_Loss': recons_loss.detach(),
            'KLD': kld_loss.detach(),
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)


# =============================================================================
# Model 4: Convolutional Variational Autoencoder
# =============================================================================
class ConvVAE(BaseVAE):
    """
    Variational autoencoder with convolutional encoder and transposed-conv decoder.
    Uses the reparameterization trick and an ELBO loss
    (reconstruction MSE + weighted KL divergence).
    Suitable for 1-D spectra or 2-D microscopy images where latent space
    exploration and generation are the goals.
    """

    default_encoder_config = [{
        'n_layers': 2,
        'in_channels': 1,
        'out_channels': 16,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'batch_norm': True,
        'dropout_prob': 0.0,
        'pooling_type': 'avg',
        'activation_fn': 'lrelu',
        'leaky_negative_slope': 0.01,
    }]

    default_decoder_config = [{
        'n_layers': 2,
        'in_channels': 16,
        'out_channels': 1,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1,
        'batch_norm': True,
        'dropout_prob': 0.0,
        'activation_fn': 'lrelu',
        'leaky_negative_slope': 0.01,
    }]

    def __init__(self,
                 in_dims: Tuple[int, ...],
                 latent_dims: int = 8,
                 encoder_configs: Optional[List[dict]] = None,
                 decoder_configs: Optional[List[dict]] = None,
                 kld_weight: float = 1.0,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_dims = in_dims
        self.latent_dims = latent_dims
        self.kld_weight = kld_weight
        self._lr = lr
        self._weight_decay = weight_decay

        enc_cfg = encoder_configs or ConvVAE.default_encoder_config
        dec_cfg = decoder_configs or ConvVAE.default_decoder_config

        self.encoder = ConvVariationalEncoder(
            in_dims=in_dims, n_latent=latent_dims, block_configs=enc_cfg
        )
        self.decoder = ConvVariationalDecoder(
            out_dims=in_dims, n_latent=latent_dims, block_configs=dec_cfg
        )

    def encode(self, x: Tensor) -> List[Tensor]:
        return self.encoder(x)  # (mu, log_var)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> List[Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        # ConvVariationalDecoder outputs (B, 1, H, W); squeeze the channel dim
        # so that reconstruction matches the (B, H, W) input for the loss function.
        if reconstruction.ndim == x.ndim + 1:
            reconstruction = reconstruction.squeeze(1)
        return reconstruction, x, mu, log_var

    def sample(self, num_samples: int, current_device) -> Tensor:
        z = torch.randn(num_samples, self.latent_dims).to(current_device)
        return self.decode(z).squeeze(1)

    def reconstruct(self, x: Tensor) -> Tensor:
        return self.forward(x)[0]

    def loss_function(self, recons: Tensor, x: Tensor,
                      mu: Tensor, log_var: Tensor, **kwargs) -> Dict[str, Tensor]:
        recons_loss = F.mse_loss(recons, x)
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
        )
        loss = recons_loss + self.kld_weight * kld_loss
        return {
            'loss': loss,
            'Reconstruction_Loss': recons_loss.detach(),
            'KLD': kld_loss.detach(),
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr, weight_decay=self._weight_decay)


# =============================================================================
# Legacy VAE class (kept for backward compatibility)
# =============================================================================
class VAE(BaseVAE):
    """
    Legacy combined VAE class. Prefer LinearVAE or ConvVAE for new code.
    Selects encoder/decoder type via encoder_type and decoder_type arguments.
    """

    default_encoder = {
        'linear': [{
            'n_layers': 1,
            'out_features': 64,
            'batch_norm': False,
            'dropout_prob': 0.0,
            'pooling_type': None,
            'activation_fn': 'lrelu',
            'leaky_negative_slope': 0.001,
        }],
        'conv': [{
            'n_layers': 1,
            'in_channels': 1,
            'out_channels': 16,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'batch_norm': False,
            'dropout_prob': 0.0,
            'pooling_type': 'avg',
            'activation_fn': 'lrelu',
            'leaky_negative_slope': 0.001,
        }],
    }

    default_decoder = {
        'linear': [{
            'n_layers': 1,
            'in_features': 64,
            'batch_norm': False,
            'dropout_prob': 0.0,
            'pooling_type': None,
            'activation_fn': 'lrelu',
            'leaky_negative_slope': 0.001,
        }],
        'conv': [{
            'n_layers': 1,
            'in_channels': 16,
            'out_channels': 1,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'batch_norm': False,
            'dropout_prob': 0.0,
            'activation_fn': 'lrelu',
            'leaky_negative_slope': 0.001,
        }],
    }

    def __init__(self,
                 in_dims: Tuple[int, ...],
                 latent_dims: int = 2,
                 encoder_type: str = 'conv',
                 decoder_type: str = 'conv',
                 encoder_configs: Optional[List[dict]] = None,
                 decoder_configs: Optional[List[dict]] = None,
                 lr: float = 1e-3,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.latent_dims = latent_dims
        self.in_dims = in_dims
        self._lr = lr

        if encoder_configs is None:
            encoder_configs = VAE.default_encoder[encoder_type]

        if encoder_type == 'conv':
            encoder = ConvVariationalEncoder
        else:
            encoder_configs[0]['in_features'] = in_dims[0]
            encoder = LinearVariationalEncoder

        self.encoder = encoder(in_dims=in_dims, n_latent=latent_dims, block_configs=encoder_configs)

        if decoder_configs is None:
            decoder_configs = VAE.default_decoder[decoder_type]

        if decoder_type == 'conv':
            decoder = ConvVariationalDecoder
        else:
            decoder_configs[-1]['out_features'] = in_dims[0]
            decoder = LinearVariationalDecoder

        self.decoder = decoder(out_dims=in_dims, n_latent=latent_dims, block_configs=decoder_configs)

    def encode(self, x: Tensor) -> List[Tensor]:
        return self.encoder(x)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> List[Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, x, mu, log_var

    def sample(self, num_samples: int, current_device) -> Tensor:
        z = torch.randn(num_samples, self.latent_dims).to(current_device)
        return self.decode(z)

    def reconstruct(self, x: Tensor) -> Tensor:
        return self.forward(x)[0]

    def loss_function(self, recons, input, mu, log_var) -> dict:
        kld_weight = 0.5
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': kld_loss.detach()}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr)


# =============================================================================
# Latent space analysis utilities
# =============================================================================
def get_latent_stats(model: BaseVAE, data_loader: DataLoader,
                     device: torch.device) -> Dict[str, Any]:
    """
    Compute per-dimension statistics of the learned latent space.

    For variational models (LinearVAE, ConvVAE, VAE) this returns the
    analytical KLD per latent dimension and the fraction of 'active' dims.
    For vanilla AEs this returns activation statistics of the bottleneck.

    Args:
        model:       A trained BaseVAE subclass.
        data_loader: DataLoader over the evaluation dataset.
        device:      Torch device the model lives on.

    Returns:
        dict with keys:
            mu_mean      – per-dim mean of encoder means (variational) or
                           bottleneck activations (vanilla)
            mu_std       – per-dim std of encoder means / bottleneck activations
            kld_per_dim  – per-dim analytic KLD (variational only; else None)
            active_dims  – number of dims with std > activity_threshold
    """
    model.eval()
    activity_threshold = 0.1

    all_mu = []
    all_log_var = []
    is_variational = hasattr(model, 'reparameterize')

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)

            if is_variational:
                mu, log_var = model.encode(x)
                all_mu.append(mu.cpu())
                all_log_var.append(log_var.cpu())
            else:
                z = model.encode(x)
                all_mu.append(z.cpu())

    mu_tensor = torch.cat(all_mu, dim=0)          # (N, latent_dims)
    mu_mean = mu_tensor.mean(dim=0).tolist()
    mu_std = mu_tensor.std(dim=0).tolist()
    active_dims = int((torch.tensor(mu_std) > activity_threshold).sum().item())

    kld_per_dim = None
    if is_variational and all_log_var:
        lv_tensor = torch.cat(all_log_var, dim=0)  # (N, latent_dims)
        # Analytic KLD per dim, averaged over the dataset
        kld_per_dim = (
            -0.5 * (1 + lv_tensor - mu_tensor ** 2 - lv_tensor.exp())
        ).mean(dim=0).tolist()

    return {
        'mu_mean': mu_mean,
        'mu_std': mu_std,
        'kld_per_dim': kld_per_dim,
        'active_dims': active_dims,
    }


# =============================================================================
# Model registry — used by VAEAgent for model selection
# =============================================================================
VAE_REGISTRY: Dict[str, type] = {
    'linear_ae': LinearAE,
    'conv_ae': ConvAE,
    'linear_vae': LinearVAE,
    'conv_vae': ConvVAE,
}

VAE_REGISTRY_DESC: Dict[str, str] = {
    'linear_ae': 'Plain linear autoencoder (FC layers, no KL divergence). Use for 1-D data where the goal is compression or denoising, NOT latent space exploration.',
    'conv_ae': 'Plain convolutional autoencoder (no KL divergence). Use for 2-D (or 1-D) data where the goal is spatial reconstruction / denoising.',
    'linear_vae': 'Variational AE with FC encoder/decoder. Use for 1-D data where learning a structured, explorable latent space is the goal.',
    'conv_vae': 'Variational AE with conv encoder/decoder. Use for 2-D (or 1-D) data where structured latent-space exploration of spatial features is the goal.',
}