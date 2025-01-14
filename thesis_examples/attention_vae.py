
import torch
from torch import nn
from torch.nn import functional as F
from typing import *
from torch import Tensor

from thesis_examples.mlp import MLP

class AttentionVAE(nn.Module):

    def __init__(self,
                 sequence_length: int,
                 num_agents: int,
                 latent_dim: int,
                 embedding_dim: int,
                 **kwargs) -> None:
        super(AttentionVAE, self).__init__()

        out_dim = sequence_length*num_agents*3

        self.embedding_dim = embedding_dim
        self.out_dim = out_dim
        self.num_agents = num_agents
        self.sequence_length = sequence_length

        self.mu_dim = latent_dim
        self.sigma_dim = latent_dim ** 2
        self.latent_dim = latent_dim

        self.input_encoder = MLP(input_size=num_agents*3, output_size=self.embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=12, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=12, batch_first=True)
        self.fc_mu = MLP(self.embedding_dim, self.mu_dim)
        self.fc_sigma = MLP(self.embedding_dim, self.sigma_dim)

        self.decoder_layer = MLP(self.latent_dim, self.out_dim)


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B, A, T, D]
        :return: (Tensor) List of latent codes
        """
        batch_size, num_agents, timesteps, traj_dim = input.shape
        input = input.permute(0, 2, 1, 3).reshape(batch_size, timesteps, -1)
        input = self.input_encoder(input)
        result = self.encoder(input)

        # max pool
        result = torch.max(result, dim=1).values

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_sigma(result).reshape(-1, self.latent_dim, self.latent_dim)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_layer(z)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu).unsqueeze(-1)
        return torch.matmul(std, eps).squeeze(-1) + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        result = self.decode(z)
        result = result.reshape(-1, self.num_agents, self.sequence_length, 3)
        return [result, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]