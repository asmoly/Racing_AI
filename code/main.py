import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from itertools import count
from collections import deque

import numpy as np
from time import time
from keyboard import is_pressed

from Car import Car
from Environment import Environment

WINDOW_SIZE = (1000, 1000)

CAR_SIZE = (10, 20)
CAR_STARTING_POSITION = (520, 920)

RENDER = True

GAMMA = 0.5

car = Car(max_speed=300, acceleration=50, brake_force=120, turning_speed=160, drag=40, starting_position=CAR_STARTING_POSITION, steering_to_speed_ratio=0.5)
environment = Environment(path_to_track="race_track_1", window_dimensions=WINDOW_SIZE, car=car, car_size=CAR_SIZE)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.layer1 = nn.Linear(27, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, len(environment.actions_to_command))

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        action_scores = self.layer3(x)
        
        return F.softmax(action_scores, dim=1)

def load_model(path):
    model = Policy()
    model.load_state_dict(torch.load(path))
    model.eval()

    return model

#policy = Policy()
policy = load_model("models/race_net_official.pt")    

optimizer = optim.AdamW(policy.parameters(), lr=1e-5)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)

    try:
        m = Categorical(probs=probs)
        action = m.sample()
        policy.saved_log_probs.append(m.log_prob(action))

        return action.item()
    except:
        print("Probs:", probs)
        print("State:", state)
        raise Exception("Categorical is broken!!!!!!!!!!!!!!!")
        return 0

def finish_episode():
    R = 0

    policy_loss = []
    returns = deque()

    for r in policy.rewards[::-1]:
        R = r + GAMMA * R
        returns.appendleft(R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)

    #print(policy_loss)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()  # can be avg() to make it sampled expectation
    #print(f"        Loss: {policy_loss.item()}")

    policy_loss.backward()
    optimizer.step()

    reset_reward_and_probs()

def reset_reward_and_probs():
    del policy.rewards[:]
    del policy.saved_log_probs[:]

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
    running_reward = 0

    for i_episode in count(1):
        delta_time = 0
        
        state = environment.reset()
        episode_reward = 0

        done = False
        sub_episode_counter = 0

        while done == False:
            sub_episode_length = 0
            for i in range (0, 2000):
                start_time = time()

                action = select_action(state) 
                state, reward, done = environment.step(delta_time, action)

                if RENDER == True:
                    environment.render(draw_raycasts=False)

                policy.rewards.append(reward)
                episode_reward += reward

                print(f"\rSubepisode: {sub_episode_counter}, Loop idx: {i} | Action: {action}", end="")

                sub_episode_length = i

                if done == True:
                    break

                delta_time = time() - start_time

            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            
            if sub_episode_length > 10:
                finish_episode()
            else:
                reset_reward_and_probs()

            sub_episode_counter += 1

            if sub_episode_counter >= 20:
                done = True

        print(f"\rRunning Reward: {running_reward}, Episode Reward: {episode_reward}, Episode #: {i_episode}")

        if i_episode % 10 == 0:
            torch.save(policy.state_dict(), "models/race_net.pt")

def main():
    train()

if __name__== "__main__":
    main()

