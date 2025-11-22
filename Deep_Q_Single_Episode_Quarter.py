#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep_Q_Single_Episode_Quarter.py
Modified version of Deep_Q_Single_Episode_MacOS_FIXED.py
Now supports quarterly training using start_time and stop_time.
"""

import os
import sys
import json
import random
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
import shutil

# ---------------- Configuration ----------------
EPISODES = 40
BUFFER_LIMIT = 50000
BATCH_SIZE = 100
GAMMA = 0.99
LR = 1e-3
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.05
TARGET_UPDATE_FREQUENCY = 25
NETWORK_SAVE_FREQUENCY = 10
# ------------------------------------------------


class Neural_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(3, 200)
        self.layer_2 = nn.Linear(200, 200)
        self.layer_3 = nn.Linear(200, 11)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)


class ReplayBuffer:
    def __init__(self, capacity, path):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.path = path

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
            print(f"Loaded replay buffer with {len(self.buffer)} transitions")


class environment:
    def __init__(self, FMU_PATH):
        self.model_description = read_model_description(FMU_PATH)
        self.FMU_PATH = FMU_PATH
        self.action_dict = dict(enumerate([-5000, -4000, -3000, -2000, -1000, 0, 1000, 2000, 3000, 4000, 5000]))

    def instantiate(self):
        self.vrs = {v.name: v.valueReference for v in self.model_description.modelVariables}
        self.unzipdir = extract(self.FMU_PATH)
        self.fmu = FMU2Slave(
            guid=self.model_description.guid,
            unzipDirectory=self.unzipdir,
            modelIdentifier=self.model_description.coSimulation.modelIdentifier,
            instanceName="instance1",
        )

    # 🆕 Modified: use dynamic start and stop times
    def reset_env(self, start_time, stop_time):
        self.fmu.instantiate()
        self.fmu.reset()
        self.fmu.setupExperiment(startTime=start_time, stopTime=stop_time)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()

    def terminate_env(self):
        self.fmu.terminate()
        self.fmu.freeInstance()
        shutil.rmtree(self.unzipdir, ignore_errors=True)

    def get_state(self):
        Tin = round(self.fmu.getReal([self.vrs["Tin"]])[0], 1)
        Tout = round(self.fmu.getReal([self.vrs["Tout"]])[0], 1)
        return Tin, Tout

    def do_step(self, current_time, action, step_size):
        self.fmu.setReal([self.vrs["Q"]], [float(action)])
        self.fmu.doStep(currentCommunicationPoint=current_time, communicationStepSize=step_size)
        return current_time + step_size

    def reward(self, temp_factor, energy_factor, temp_next, next_setpoint, action):
        upper = next_setpoint + 1
        lower = next_setpoint - 1
        if lower <= temp_next <= upper:
            temp_reward = 10
        elif temp_next < lower:
            temp_reward = temp_next - (lower - 1)
        else:
            temp_reward = -temp_next + (upper + 1)
        return temp_factor * temp_reward + energy_factor * action


def train_model(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma):
    if len(replay_buffer.buffer) < batch_size:
        return 0
    transitions = replay_buffer.sample(batch_size)
    state, action, reward, next_state, done = zip(*transitions)
    state = torch.FloatTensor(state)
    action = torch.LongTensor(action).unsqueeze(1)
    reward = torch.FloatTensor(reward).unsqueeze(1)
    next_state = torch.FloatTensor(next_state)
    done = torch.FloatTensor(done).unsqueeze(1)

    q_values = policy_net(state).gather(1, action)
    next_q_values = target_net(next_state).max(1)[0].detach().unsqueeze(1)
    expected_q_values = reward + (1 - done) * gamma * next_q_values

    loss = F.mse_loss(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    # 🆕 Accept start and stop times as arguments
    episode = int(sys.argv[1])
    title = sys.argv[2]
    work_dir_path = Path(sys.argv[3])
    episodes = int(sys.argv[4])
    start_time = float(sys.argv[5])
    stop_time = float(sys.argv[6])

    print(f"\n=== Running Episode {episode} for {title} ===")
    print(f"Training from {start_time} to {stop_time} seconds\n")

    FMU_PATH = f"/Users/luisrubio/Desktop/Energy/_fmu_export_schedule_V940.fmu"
    env_obj = environment(FMU_PATH)
    env_obj.instantiate()

    policy_net = Neural_Network()
    target_net = Neural_Network()
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)

    buffer_path = work_dir_path / "replay_buffer.pkl"
    replay_buffer = ReplayBuffer(BUFFER_LIMIT, buffer_path)
    replay_buffer.load()

    epsilon = max(MIN_EPSILON, 1.0 * (EPSILON_DECAY ** episode))

    # 🆕 Call reset_env with dynamic quarter times
    env_obj.reset_env(start_time, stop_time)

    Tin, Tout = env_obj.get_state()
    setpoint = 20
    state = np.array([Tin, Tout, setpoint], dtype=np.float32)

    total_reward = 0
    total_loss = 0
    steps = 960
    step_size = 900
    temp_factor = 1
    energy_factor = -0.001

    for t in range(steps):
        if random.random() < epsilon:
            action_idx = random.choice(list(env_obj.action_dict.keys()))
        else:
            with torch.no_grad():
                q_values = policy_net(torch.FloatTensor(state).unsqueeze(0))
                action_idx = torch.argmax(q_values).item()

        action = env_obj.action_dict[action_idx]
        current_time = start_time + t * step_size
        current_time = env_obj.do_step(current_time, action, step_size)

        Tin_next, Tout_next = env_obj.get_state()
        next_setpoint = 20
        next_state = np.array([Tin_next, Tout_next, next_setpoint], dtype=np.float32)

        reward = env_obj.reward(temp_factor, energy_factor, Tin_next, next_setpoint, action)
        done = 0
        replay_buffer.push((state, action_idx, reward, next_state, done))

        loss = train_model(policy_net, target_net, optimizer, replay_buffer, BATCH_SIZE, GAMMA)
        total_loss += loss
        total_reward += reward

        state = next_state

        if t % TARGET_UPDATE_FREQUENCY == 0:
            target_net.load_state_dict(policy_net.state_dict())

    env_obj.terminate_env()
    replay_buffer.save()

    save_path = work_dir_path / f"{title}_episode_{episode}_weights.pth"
    torch.save(policy_net.state_dict(), save_path)

    results = {
        "Episode": episode,
        "Start time": start_time,
        "Stop time": stop_time,
        "Rewards": total_reward,
        "T_in": Tin_next,
        "T_out": Tout_next,
        "Saved path": str(save_path),
    }

    with open(work_dir_path / f"{title}_episode_{episode}_training_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Episode {episode} complete. Reward: {total_reward:.2f}, Loss: {total_loss:.4f}")


if __name__ == "__main__":
    main()