import math
import random
from dataclasses import dataclass
from typing import Dict, Any, List

import gym
import numpy as np
import pybullet as p

from racecar_gym.bullet.configs import MapConfig
from racecar_gym.core import world
from racecar_gym.core.agent import Agent
from racecar_gym.core.definitions import Pose


class World(world.World):
    FLOOR_ID = 0
    WALLS_ID = 1
    FINISH_ID = 2

    @dataclass
    class Config:
        sdf: str
        map_config: MapConfig
        rendering: bool
        time_step: float
        gravity: float
        start_positions: str

    def __init__(self, config: Config, agents: List[Agent]):
        self._config = config
        self._map_id = None
        self._time = 0.0
        self._agents = agents
        self._state = dict([(a.id, {}) for a in agents])
        self._objects = {}
        self._progress_map = np.load(config.map_config.maps)['norm_distance_from_start']
        self._drivable_area = np.load(config.map_config.maps)['drivable_area']
        self._distance_to_obstacle = np.load(config.map_config.maps)['norm_distance_to_obstacle']
        self._starting_grid = np.load(config.map_config.starting_grid)['data']

    def init(self) -> None:
        if self._config.rendering:
            id = -1  # p.connect(p.SHARED_MEMORY)
            if id < 0:
                p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self._load_scene(self._config.sdf)
        p.setTimeStep(self._config.time_step)
        p.setGravity(0, 0, self._config.gravity)

    def reset(self):
        p.setTimeStep(self._config.time_step)
        p.setGravity(0, 0, self._config.gravity)
        p.stepSimulation()
        self._time = 0.0
        self._state = dict([(a.id, {}) for a in self._agents])



    def _load_scene(self, sdf_file: str):
        ids = p.loadSDF(sdf_file)
        objects = dict([(p.getBodyInfo(i)[1].decode('ascii'), i) for i in ids])
        self._objects = objects

    def get_starting_position(self, agent: Agent) -> Pose:
        def to_meter(px, py):
            resolution = self._config.map_config.resolution
            height = self._distance_to_obstacle.shape[0]
            origin_x = self._config.map_config.origin[0]
            origin_y = self._config.map_config.origin[1]
            y = origin_y - (px - height) * resolution
            x = py * resolution + origin_x
            return x, y

        if self._config.start_positions == 'index':
            position = list(map(lambda agent: agent.id, self._agents)).index(agent.id)
            pose = self._starting_grid[position]
            return tuple(pose[:3]), tuple(pose[3:])
        if self._config.start_positions == 'random':
            center_corridor = np.argwhere(self._distance_to_obstacle > 0.4)
            position = random.choice(center_corridor)

            progress = self._progress_map[position[0], position[1]]
            checkpoints = 1.0 / float(self._config.map_config.checkpoints)
            checkpoint = int(progress / checkpoints)
            next_checkpoint = (checkpoint + 1) % self._config.map_config.checkpoints
            progress_area = next_checkpoint * checkpoints
            checkpoint_area = np.argwhere(np.logical_and(
                self._progress_map > progress_area,
                self._progress_map <= progress_area + checkpoints
            ))
            next_position = random.choice(checkpoint_area)
            px = position[0]
            py = position[1]
            x, y = to_meter(px, py)
            next_position = to_meter(next_position[0], next_position[1])


            diff = np.array(next_position) - np.array([x, y])
            angle = np.arctan2(diff[1], diff[0])
            #angle = np.random.normal(loc=angle, scale=0.15)
            return (x, y, 0.05), (0, 0, angle)
        raise NotImplementedError(self._config.start_positions)

    def update(self):
        p.stepSimulation()
        self._time += self._config.time_step

    def state(self) -> Dict[str, Any]:
        for agent in self._agents:
            self._update_race_info(agent=agent)
        return self._state

    def space(self) -> gym.Space:
        return gym.spaces.Dict({
            'time': gym.spaces.Box(low=0, high=math.inf, shape=(1,))
        })

    def _update_race_info(self, agent):
        contact_points = set([c[2] for c in p.getContactPoints(agent.vehicle_id)])

        collision = False
        for contact in contact_points:
            if self._objects['walls'] == contact:
                collision = True
            elif contact != self._objects['floor']:
                collision = True
        self._state[agent.id]['collision'] = collision

        position, orientation = p.getBasePositionAndOrientation(agent.vehicle_id)
        orientation = p.getEulerFromQuaternion(orientation)
        self._state[agent.id]['pose'] = np.array(position + orientation)

        resolution = self._config.map_config.resolution
        x, y = position[0], position[1]
        origin_x, origin_y = self._config.map_config.origin[0], self._config.map_config.origin[1]
        width, height = self._progress_map.shape[1], self._progress_map.shape[0]

        px = int(height - (y - origin_y) / resolution)
        py = int((x - origin_x) / resolution)

        self._state[agent.id]['progress'] = self._progress_map[px, py]
        self._state[agent.id]['time'] = self._time

        progress = self._state[agent.id]['progress']
        checkpoints = 1.0 / float(self._config.map_config.checkpoints)

        checkpoint = int(progress / checkpoints)

        if 'checkpoint' in self._state[agent.id]:
            last_checkpoint = self._state[agent.id]['checkpoint']
            if last_checkpoint + 1 == checkpoint:
                self._state[agent.id]['checkpoint'] = checkpoint
            elif last_checkpoint == self._config.map_config.checkpoints and checkpoint == 0:
                self._state[agent.id]['lap'] += 1
                self._state[agent.id]['checkpoint'] = checkpoint

        else:
            self._state[agent.id]['checkpoint'] = checkpoint
            self._state[agent.id]['lap'] = 0

