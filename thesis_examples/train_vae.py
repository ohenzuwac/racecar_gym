
from torch.optim import Adam
from torch import nn
import torch
from torch.utils.data import DataLoader
from attention_vae import AttentionVAE
from pandas_dataset import CustomDataset
from mpl_toolkits import mplot3d
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


import numpy as np

from tqdm import tqdm
import pandas as pd


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD  = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD

def vae_loss_old(x, x_hat, mean, log_var):
    reproduction_loss = l2_loss(x, x_hat)
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD

def l2_loss(x, x_hat):
    sq_error = (x-x_hat)**2
    sq_error = sq_error.sum(-1).sum(-1).sum(-1)
    return sq_error

def vae_loss(x, x_hat, mean, log_var):
    reproduction_loss = l2_loss(x, x_hat)
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    
    #changed this from 0.1 KLD and was previously 0.01 KLD
    return reproduction_loss + KLD



def train():

    # Model Hyperparameters

    dataset_path = '~/datasets'

    cuda = True
    DEVICE = torch.device("cuda" if cuda else "cpu")
    torch.set_default_device('cpu')

    chkpoint = 50

    #todo: overwrite dims
    batch_size = 1
    x_dim = 784
    hidden_dim = 540
    latent_dim = 2

    num_csvs = 10
    df_list = []
    
    for i in range(num_csvs):
        df = pd.read_csv(f"/home/jupyter/racecar_gym/thesis_examples/csv_files/0425/Episode{i}")
        df = df.head(4000)
        
        #subsampling, every 5th row
        df = df.iloc[::4,:]
    
        #hard coding this for now        
        start = 1
        end  = 4
        for i in range(6):
            df_agent = pd.concat([df.iloc[:,start:end],df.iloc[:,-1]], axis = 1)
            start = start+3
            end = end +3
            df_list.append(df_agent)
        
        
    train_dataset = CustomDataset(df_list)
    test_dataset = CustomDataset(df_list)

    model = AttentionVAE(sequence_length=train_dataset.sequence_length,
                         num_agents=train_dataset.num_agents,
                         latent_dim=latent_dim,
                         embedding_dim=hidden_dim)

    #lr = 1e-3
    
    #good for single csv
    #lr = 1e-4
    lr = 1e-6

    epochs = 1001

    kwargs = {'num_workers': 1, 'pin_memory': True}

    # TODO


    # TRAJECTORY format should be [batch_size, num_agents, timesteps, spatial_dim (=2)]
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    BCE_loss = nn.BCELoss()

    optimizer = Adam(model.parameters(), lr=lr)

    print("Start training VAE...")
    model.train()
    loss_plot = []

    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, x in enumerate(train_loader):

            #print("x ", x.size(), x)

            optimizer.zero_grad()
            
            #verify model is getting the correct inputs --> I want to input x and y observed positions for all agents
            #print("x",x[..., :2],x[..., :2].shape)
            
            x_hat, mean, log_var, z = model(x[..., :2])
            
            loss = vae_loss(x[..., :2], x_hat, mean, log_var).mean()
            #loss = l2_loss(x[..., :2], x_hat)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        batch_idx += 1

        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx * batch_size))
        
        loss_plot.append(overall_loss / (batch_idx * batch_size))
        
        #saving models
        if epoch % chkpoint == 0 or epoch == 999:
             if epoch > 0:
                plt.figure()
                plt.plot(np.arange(epoch+1),loss_plot,color = 'k')
                plt.xlabel("Epoch")
                plt.ylabel("Average Reconstruction Loss")
                plt.title("Average Reconstruction Loss for a Variational Autoencoder")
                plt.savefig("/home/jupyter/racecar_gym/thesis_examples/VAE_losses/Loss{}.png".format(epoch))
        
        if epoch % chkpoint == 0 or epoch == 999:
            torch.save(model,"/home/jupyter/VAEmodels/04_30_models/model{}.pt".format(epoch))
            sample = model.sample(1,torch.device('cpu'))
            #sample = sample.reshape(-1, model.num_agents, model.sequence_length, 3)
            
            #represents a single agent's trajectory
            sample = sample.reshape(1, 1, model.sequence_length, 2)
            
            #sampled values
            plot_single_agent(sample,model,epoch,0)
            
            #true reconstructions
            plot_single_agent(x_hat,model,epoch,1)
            
            #also plot x to check the GT representation
            plot_single_agent(x,model,epoch,2)
            
    print("Finish!!")
    #print(loss_plot)
    

    return model, test_loader


def plot_single_agent(x, model, epoch, slctr):
      #make this a scatter plot
    #color code the agents by type in some way --> Green, Yellow (orange), Red
    
    titles = ["Sampled VAE Trajectories for Agents After {} Epoch(s)".format(epoch), "VAE Reconstructed Trajectories for Agents After {} Epoch(s)".format(epoch), "Ground Truth Trajectories for Agents After {} Epoch(s)".format(epoch)]
    saving = ['VAE_sampled_traj{}.png'.format(epoch), "VAE_recon_traj{}.png".format(epoch), "GT{}.png".format(epoch)]
    x_vals = []
    y_vals = []

    for i in range(1):
        #print("this is a matrix!")
        #(print(x[0,0])) #--> gets the first matrix
        #print(x[0,0,1]) #--> gets the first row
        #print(x[0,0,:,0]) # --> gets the all rows in the first col

        x_vals.append(x[0,i,:,0].detach().numpy()) #append x and y trajectories for each agent
        y_vals.append(x[0,i,:,1].detach().numpy())

        #print(x_vals[0])

    #print(x_vals[0].numpy().size)
    #x_vals[0] = x_vals[0].numpy()
    #y_vals[0] = y_vals[0].numpy()
    plt.figure()
    plt.scatter(x_vals[0], y_vals[0])
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(titles[slctr])
    plt.legend(["Agent"], loc = "upper left")
    plt.show()
    plt.savefig(saving[slctr])

    return x_vals,y_vals

    

# x is the sample trajectories or original trajectories in the format generated by tensor's dataloader
def plot_traj(x,model,epoch,slctr):
    
    #make this a scatter plot
    #color code the agents by type in some way --> Green, Yellow (orange), Red
    
    titles = ["Sampled VAE Trajectories for Agents After {} Epoch(s)".format(epoch+1), "VAE Reconstructed Trajectories for Agents After {} Epoch(s)".format(epoch+1)]
    x_vals = []
    y_vals = []

    for i in range(model.num_agents):
        #print("this is a matrix!")
        #(print(x[0,0])) #--> gets the first matrix
        #print(x[0,0,1]) #--> gets the first row
        #print(x[0,0,:,0]) # --> gets the all rows in the first col

        x_vals.append(x[0,i,:,0].detach().numpy()) #append x and y trajectories for each agent
        y_vals.append(x[0,i,:,1].detach().numpy())

        #print(x_vals[0])

    #print(x_vals[0].numpy().size)
    #x_vals[0] = x_vals[0].numpy()
    #y_vals[0] = y_vals[0].numpy()
    plt.figure()
    plt.plot(x_vals[0], y_vals[0],x_vals[1],y_vals[1],x_vals[2],y_vals[2],x_vals[3],y_vals[3],
            x_vals[4],y_vals[4],x_vals[5],y_vals[5])
    plt.xlabel("x Position")
    plt.ylabel("Y Position")
    plt.title(titles[slctr])
    plt.legend(["Agent A", "Agent B", "Agent C", "Agent D","Agent E","Agent F"], loc = "upper left")
    plt.show()
    plt.savefig('VAE_gen_traj{}.png'.format(epoch))

    return x_vals,y_vals



#returns a test set for reconstructions
def get_test():
    
    #todo: overwrite dims
    batch_size = 1
    x_dim = 784
    hidden_dim = 540
    latent_dim = 2
    
    kwargs = {'num_workers': 1, 'pin_memory': True}
    
    num_csvs = 10
    df_list = []
    #hard coded
    for i in range(13,27):
        df = pd.read_csv(f"/home/jupyter/racecar_gym/thesis_examples/csv_files/0425_testset/Episode{i}")
        df = df.head(4000)
        df = df.iloc[::4,:]
            
    
        #hard coding this for now        
        start = 1
        end  = 4
        for i in range(6):
            df_agent = pd.concat([df.iloc[:,start:end],df.iloc[:,-1]], axis = 1)
            start = start+3
            end = end +3
            df_list.append(df_agent)
            
    test_dataset = CustomDataset(df_list)
    
    # TRAJECTORY format should be [batch_size, num_agents, timesteps, spatial_dim (=2)]
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    BCE_loss = nn.BCELoss()
    
    return test_loader


#assumes data comes as a list, with each element representing an agent's trajectory (x and y) information for a particular episode
def plot_traj_singles(x,model,epoch,slctr):
    
    titles = ["Sampled VAE Trajectories for Agents After {} Epoch(s)".format(epoch), "VAE Reconstructed Trajectories for Agents After {} Epoch(s)".format(epoch), "Ground Truth Trajectories for Agents After {} Epoch(s)".format(epoch)]
    saving = ['VAE_sampled_traj{}.png'.format(epoch), "VAE_recon_traj{}.png".format(epoch), "GT{}.png".format(epoch)]
    
    x_vals = []
    y_vals = []
 

    #hard coded
    for i in range(6):
        x_vals.append(x[i][0,0,:,0].detach().numpy())
        y_vals.append(x[i][0,0,:,1].detach().numpy())       
       
    plt.figure()
    plt.scatter(x_vals[0], y_vals[0], c = 'darkgreen')
    plt.scatter(x_vals[1],y_vals[1], c = 'lightgreen')
    plt.scatter(x_vals[2],y_vals[2], c = 'yellow')
    plt.scatter(x_vals[3],y_vals[3], c = 'gold')
    plt.scatter(x_vals[4],y_vals[4], c = 'tomato')
    plt.scatter( x_vals[5],y_vals[5] ,c = 'darkred')
    
    
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(titles[slctr])
    plt.legend(["Agent A", "Agent B", "Agent C", "Agent D","Agent E","Agent F"], loc = "upper left")
    plt.show()
    plt.savefig(titles[slctr])

    return x_vals,y_vals



def latent_space_plotter(z_values_x,z_values_y,z_values_z,z_values):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    model = KMeans(n_clusters = 3)
    #print(z_values)
    model.fit(z_values)
    lbls = model.labels_
    #print(lbls)
    #print(np.array([z_values_x, z_values_y, z_values_z]))
    
    ax.scatter3D(z_values_x, z_values_y, z_values_z, c=lbls.astype(float), cmap = 'Set3')
    ax.figure.savefig("example_latentspace.png")


    


if __name__ == "__main__":
    #model, test_loader = train()
    model = torch.load("/home/jupyter/VAEmodels/2_dim_10csv_single_agent_input/model200.pt")
    #sample = model.sample(1,torch.device('cpu'))
    #for batch_idx, x in enumerate(test_loader):
        #result, mu, log_var = model(x)
    test_loader = get_test()
    
    #for batch in test_loader:
    #    break
    
    
    #plot_single_agent(recon,model,99)
    #plot_traj(batch,model,1000000)
    
    #sample = model.sample(1,torch.device('cpu'))
    #sample = sample.reshape(-1, model.num_agents, model.sequence_length, 3)

    #plot original trajectories
    #plot_traj(sample,model,500000)
    
    #New plotting code
    
    #gets first 6 trajectories in the test_loader
    test_traj = []
    VAE_gen = []
    z_values_x = []
    z_values_y = []
    z_values_z = []
    z_values = []
    
    for x in test_loader:
        test_traj.append(x)
        if len(test_traj) == 1:
            break
    #plot_traj_singles(test_traj[11:18],model,1000,2)
    
    for agent in test_traj[11:18]:
        VAE_gen.append(model.generate(agent[...,:2]))
    
    #plot_traj_singles(VAE_gen,model,1000,1)
    
    for x in test_loader:
        result, mu, log_var, z = model(x[...,:2])
        z_values_x.append(z.detach().numpy()[0][0])
        z_values_y.append(z.detach().numpy()[0][1])
        z_values_z.append(z.detach().numpy()[0][2])
        z_values.append(z.detach().numpy()[0])
    
    latent_space_plotter(z_values_x,z_values_y,z_values_z,z_values)
    
    
    
      
    
    