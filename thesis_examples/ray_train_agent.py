#trying to get a minimal example of training working with rllib

import numpy as np
import pprint
import random
import matplotlib.pyplot as plt
import cv2
import yaml
import wandb
import ray
import argparse
import pandas as pd
import os

from ray.rllib.agents.ppo import PPOTrainer
from ray_wrapper import RayWrapper
import gymnasium
from racecar_gym.envs.gym_api import MultiAgentRaceEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.rllib.policy.policy import PolicySpec
from dictionary_space_utility import flatten_obs
from ray.rllib.algorithms.algorithm import Algorithm
from time import sleep
from collections import defaultdict
from agent_traits import assign_traits




def env_creator(env_config):
    env = gymnasium.make(
        id='MultiAgentRaceEnv-v0',
        scenario='../scenarios/8agent_austria_het.yml',
        render_mode="rgb_array_follow"
    )

    return RayWrapper(env)

#maps policies to agents...I don't fully understand this feature

def policy_mapping_fn(agent_id,episode,worker,**kwargs):
    pol_id = agent_id
    return pol_id


#function for simulating agents with a trained model, also collect a dictionary of trajectories for the agent
def simulate(algo,eps):
    env = gymnasium.make(
        id='MultiAgentRaceEnv-v0',
        scenario='../scenarios/8agent_austria_het.yml',
        render_mode="human",
        # render_options=dict(width=320, height=240, agent='A')
    )
    ray_env = RayWrapper(env)

    policy_agent_mapping = algo.config['multiagent']['policy_mapping_fn']
    video_array = []
    trajectories = {}


    for episode in range(eps):
        trajectories['Episode{}'.format(episode)] = {}
        print('Episode: {}'.format(episode))
        obs, _ = ray_env.reset()
        #print(obs)
        done = {agent: False for agent in obs.keys()}

        img_array = []
        timesteps = []
        timestep = 0
        traj = {}
        for agent_id in obs.keys():
            traj['x_pos_{}'.format(agent_id)] = []
            traj['y_pos_{}'.format(agent_id)] = []
            traj['yaw_pos_{}'.format(agent_id)] = []

        while True: # Run until the episode ends
            # Get actions from policies
            joint_action = {}
            for agent_id, agent_obs in obs.items():
                #print(done[agent_id],agent_id)
                #if done[agent_id]: continue # Don't get actions for done agents
                policy_id = policy_agent_mapping(agent_id,episode,None)
                action = algo.compute_single_action(agent_obs, policy_id=policy_id)
                joint_action[agent_id] = action



            # Step the simulation
            obs, reward, done, truncated, info = ray_env.step(joint_action)

            for agent_id in ray_env.agents:
                #need to have a catch here for done agents --> append the information when the agent last active
                if agent_id not in obs.keys():
                    traj['x_pos_{}'.format(agent_id)].append(traj['x_pos_{}'.format(agent_id)][-1])
                    traj['y_pos_{}'.format(agent_id)].append(traj['y_pos_{}'.format(agent_id)][-1])
                    traj['yaw_pos_{}'.format(agent_id)].append(traj['yaw_pos_{}'.format(agent_id)][-1])
                else:
                    #append the observed pose information for each agent
                    traj['x_pos_{}'.format(agent_id)].append(info[agent_id]['observations']['pose'][0])
                    traj['y_pos_{}'.format(agent_id)].append(info[agent_id]['observations']['pose'][1])
                    traj['yaw_pos_{}'.format(agent_id)].append(info[agent_id]['observations']['pose'][5])

            timesteps.append(timestep)
            timestep = timestep + 1

            #rgb_array = ray_env.render()
            #ray_env.render()
            #transfer to BGR for openCV

            #img_array.append(cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))

            #rgb_array = ray_env.render()
            #sleep(0.01)

            if done['__all__']:
                #video_array.append(img_array)

                print("done!!")

                trajectories['Episode{}'.format(episode)] = traj
                trajectories['Episode{}'.format(episode)]['timesteps'] = timesteps
                break
    return trajectories


#assume the input is a set of images representing one video --> not working
def createvideo(vid):
    episode = 1
    filename = "trained_vid.mp4"
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (320,240))
    for pic in vid:
        out.write(pic)

    out.release()


#takes a dictionary containing trajedctories and save them to csv file using pandas
def save_trajs(trajectories,filepath = "/home/christine/thesis_trajectories"):

    #loop through the trajs
    for episode in trajectories.keys():
        filename = "episode{}".format(episode)
        file = os.path.join(filepath,filename)
        df = pd.DataFrame(trajectories[episode])
        df.to_csv(file)


def train(algo,checkpoint_num,epochs):
    #randomize max velocity and max steering angles
    assign_traits()
    for epoch in range(epochs):
        # Train the model for (1?) epoch with a pre-specified rollout fragment length (how many rollouts?).
        results = algo.train()
        #print(pretty_print(results))

        # Log the results
        log_dict = {}  # reset the log dict
        results_top_level_stats = {"stats/" + k:v for (k,v) in results.items() if k != "info" and type(v) is not dict}
        results_info_stats = {"info/" + k:v for (k,v) in results["info"].items() if k != "learner" and type(v) is not dict}
        for d in [results_info_stats, results_top_level_stats]:
            log_dict.update(d)
        for agent_prefix, agent_dict in results["info"]["learner"].items():
            learner_stats = {"learner_agent_" + agent_prefix + "/" + k:v for (k,v) in agent_dict["learner_stats"].items()}
            log_dict.update(learner_stats)
        if epoch % checkpoint_num == 0:
            checkpoint = algo.save("/home/christine/trained_models/0422")
            log_dict["checkpoint"] = checkpoint
        wandb.log(log_dict)


def run():

    register_env("my_env",env_creator)

    #working on parser

    parser = argparse.ArgumentParser(description= "simulating and training MARL environment")
    parser.add_argument("--rollout_fragment_length", default = 'auto')
    parser.add_argument("--epochs", type = int, default = 1000)
    parser.add_argument("--train", type = bool, default = True, help = "if true, train model")
    parser.add_argument("--checkpoint", type = int, default = 10, help = "number of timsteps to wait before saving model while training")
    parser.add_argument("--simulate", type = bool, default = False, help = "if true, simulate agents actions")
    parser.add_argument("--saved_model", type = str, default = None, help = "filepath of saved model to simulate agent actions from")

    #also add render_mode, save_video, video_fp
    #fp --> filepath


    epochs = 100000
    rollout_fragment_length = 'auto'
    params = {"epochs": epochs,
              "rollout_fragment_length": rollout_fragment_length} #TODO(christine.ohenzuwa): add command line args using python argparse
    # https://docs.python.org/3/library/argparse.html


    wandb.init(
        # Set the project where this run will be logged
        project="population-learning",
        # Track hyperparameters and run metadata
        config=params)

    #creating different policies for different agents...not sure if this is necessary since all agents have the same
    #rewards and tasks
    #also not very programatic
    policies = {

        "A": PolicySpec(policy_class = None, observation_space = None, action_space = None, config = None),
        "B": PolicySpec(policy_class = None, observation_space = None, action_space = None, config = None),
        "C":PolicySpec(policy_class = None, observation_space = None, action_space = None, config = None),
        "D":PolicySpec(policy_class = None, observation_space = None, action_space = None, config = None),
        "E": PolicySpec(policy_class=None, observation_space=None, action_space=None, config=None),
        "F": PolicySpec(policy_class=None, observation_space=None, action_space=None, config=None),
        "G": PolicySpec(policy_class=None, observation_space=None, action_space=None, config=None),
        "H": PolicySpec(policy_class=None, observation_space=None, action_space=None, config=None),
        "I": PolicySpec(policy_class=None, observation_space=None, action_space=None, config=None)

    }
    config = PPOConfig().framework("torch").rollouts(rollout_fragment_length=params["rollout_fragment_length"])
    config = config.environment(env = "my_env")
    config = config.multi_agent(policies = policies,
                                policy_mapping_fn = policy_mapping_fn)
    #adding more workers for parallelization
    config = config.rollouts(num_rollout_workers = 2)



    #algo = config.build()
    #checkpoint_num = 10
    #checkpoint_path = "/home/christine/trained_models/0421/checkpoint_000381"
   # algo = Algorithm.from_checkpoint(checkpoint_path)

    #all the stuff above happens on every run of this file

    # param true or false check would go here
    #train(algo,checkpoint_num,params["epochs"])

    #another boolean check would go here --> for simulate I guess

    checkpoint_path = "/home/christine/trained_models/0422/checkpoint_000211"
    algo = Algorithm.from_checkpoint(checkpoint_path)
    trajectories = simulate(algo,2)
    #save_trajs(trajectories)

    #more checks for writing trajectories and rendering videos go here


if __name__ == "__main__":
    run()
