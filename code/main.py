import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from itertools import count

import math
import random
from time import time
from keyboard import is_pressed

from Graphics import Graphics
from Car import Car
from Vector import Vector
from Matrix import Matrix
from Environment import Environment

WINDOW_SIZE = (1000, 1000)

CAR_SIZE = (10, 20)
CAR_STARTING_POSITION = (570, 920)

def convert_controls_to_action(acceleration, steering):
    action = 14
    if acceleration == 1 and steering == 1:
        action = 2
    elif acceleration == 1 and steering == -1:
        action = 3
    elif acceleration == -1 and steering == 1:
        action = 4
    elif acceleration == -1 and steering == -1:
        action = 5
    elif acceleration == 0 and steering == 1:
        action = 10
    elif acceleration == 0 and steering == -1:
        action = 11
    elif acceleration == 1:
        action = 0
    elif acceleration == -1:
        action = 1

    return action

def train():
    car = Car(max_speed=300, acceleration=50, brake_force=120, turning_speed=160, drag=40, starting_position=CAR_STARTING_POSITION, steering_to_speed_ratio=0.5)
    environment = Environment(path_to_track="race_track_1", window_dimensions=WINDOW_SIZE, car=car, car_size=CAR_SIZE)
    
    delta_time = 0
    
    while True:
        start_time = time()

        acceleration = 0
        steering = 0
        if is_pressed("w"):
            acceleration = 1
        elif is_pressed("s"):
            acceleration = -1

        if is_pressed("a"):
            steering = -1
        if is_pressed("d"):
            steering = 1

        action = convert_controls_to_action(acceleration, steering)

        raycasts, reward, speed = environment.step(delta_time, action)

        delta_time = time() - start_time

def main():
    train()

if __name__== "__main__":
    main()


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

