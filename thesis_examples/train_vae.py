
from torch.optim import Adam
from torch import nn
import torch
from torch.utils.data import DataLoader
from attention_vae import AttentionVAE
from pandas_dataset import CustomDataset

import numpy as np

from tqdm import tqdm
import pandas as pd


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD  = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD

def vae_loss(x, x_hat, mean, log_var):
    reproduction_loss = l2_loss(x, x_hat)
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD

def l2_loss(x, x_hat):
    sq_error = (x-x_hat)**2
    sq_error = sq_error.sum(-1).sum(-1).sum(-1)
    return sq_error



def train():

    # Model Hyperparameters

    dataset_path = '~/datasets'

    cuda = True
    DEVICE = torch.device("cuda" if cuda else "cpu")

    #todo: overwrite dims
    batch_size = 1
    x_dim = 784
    hidden_dim = 540
    latent_dim = 2

    num_csvs = 2
    df_list = []
    for i in range(num_csvs):
        df = pd.read_csv(f"csv_files/episodeEpisode{i}.csv")
        df_list.append(df)
    train_dataset = CustomDataset(df_list)
    test_dataset = CustomDataset(df_list)

    model = AttentionVAE(sequence_length=train_dataset.sequence_length,
                         num_agents=train_dataset.num_agents,
                         latent_dim=latent_dim,
                         embedding_dim=hidden_dim)

    lr = 1e-3

    epochs = 30

    kwargs = {'num_workers': 1, 'pin_memory': True}

    # TODO


    # TRAJECTORY format should be [batch_size, num_agents, timesteps, spatial_dim (=2)]
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    BCE_loss = nn.BCELoss()

    optimizer = Adam(model.parameters(), lr=lr)

    print("Start training VAE...")
    model.train()

    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, x in enumerate(train_loader):

            optimizer.zero_grad()

            x_hat, mean, log_var, z = model(x)
            loss = vae_loss(x[..., :2], x_hat, mean, log_var).mean()

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        batch_idx += 1

        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx * batch_size))

    print("Finish!!")


if __name__ == "__main__":
    train()