from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch

class CustomDataset(Dataset):
    def __init__(self, dataframe_list):

        # Convert pandas dataframe to tensor
        episode_tensor_list = []
        for df in dataframe_list:
            columns_of_interest = [x for x in df.columns if "pos" in x]
            num_agents = len(columns_of_interest) // 3
            values = df[columns_of_interest].values
            values_as_tensor = torch.Tensor(values).reshape(-1, num_agents, 3)
            episode_tensor_list.append(values_as_tensor)

        # Truncate episodes
        min_episode_length = min([len(x) for x in episode_tensor_list])
        episode_tensor_list = [x[:min_episode_length] for x in episode_tensor_list]

        self.data = torch.stack(episode_tensor_list).permute(0, 2, 1, 3)
        self.size, self.num_agents, self.sequence_length, self.traj_dim = self.data.shape
        print(episode_tensor_list)

    def __getitem__(self, index):
        traj = self.data[index]
        return traj

    def __len__(self):
        return self.size

