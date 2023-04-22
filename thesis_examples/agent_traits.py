
#Script for randomizing agent traits --> max velocity and max steering angle

import sys
import ruamel.yaml
import os
import numpy as np
import scipy as sp
from scipy import stats

#assume there are 3 populations whose max velocity and max steering angled are sampled from a Gaussian
#mean = __, std = 1

def assign_traits():

    yaml = ruamel.yaml.YAML()
    # yaml.preserve_quotes = True

    vehicles = ['racecar/racecar.yml','racecar_slower/racecar_slower.yml','racecar_slowest/racecar_slowest.yml']
    filepath = '/home/christine/racecar_gym/models/vehicles'

    for ind,vehicle in enumerate(vehicles):
        file = os.path.join(filepath,vehicle)

        if ind == 0: #fastest car, smallest steering angle
            loc_vel = 10
            loc_angle = 0.2
            vel = np.random.normal(loc = loc_vel)
            #angle = stats.truncnorm((0-loc_angle)/0.5,(0.99-loc_angle)/0.5, loc = loc_angle)
            #angle = angle.rvs()
            #angle = np.random.normal(loc = loc_angle, scale = 0.1)
        if ind == 1:
            loc_vel = 5
            loc_angle = 0.4
            vel = np.random.normal(loc = loc_vel)
            #angle = np.random.normal(loc = loc_angle,scale = 0.1)

        if ind == 2: #slowest car, largest steering angle
            loc_vel = 2.5
            loc_angle = 0.6
            vel = np.random.normal(loc=loc_vel)
            #angle = np.random.normal(loc=loc_angle, scale = 0.1)

        with open(file) as fp:
            data = yaml.load(fp)
            #velocity values should be floats
            #motor
            data['actuators'][1]['params']['max_velocity'] = float(vel)
            #data['actuators']
            #data['actuators'][0]['params']['max_steering_angle'] = float(angle)

        with open(file,'wb') as fp:
            #velocity values should be floats
            #data['actuators']
            yaml.dump(data, fp)

        #yaml.dump(data, sys.stdout)

#assign_traits()
