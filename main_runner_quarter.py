#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 2, 2025

Fixed merged version for Luis Rubio.
Compatible with Rosetta (x86_64) and FMU _fmu_export_schedule_V940.fmu
"""

import os
import subprocess
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# Force Rosetta/x86 Python path
PYTHON_X86 = sys.executable


# Add EnergyPlus directory to PATH
os.environ["PATH"] = "/Applications/EnergyPlus-9-4-0:" + os.environ.get("PATH", "")

# Environment variables
env = os.environ.copy()
env["FMUSOCKETHOSTNAME"] = "127.0.0.1"


class environment:
    def __init__(self, FMU_PATH):
        self.model_description = read_model_description(FMU_PATH)
        self.FMU_PATH = FMU_PATH
        act_disc_states = [-5000, -4000, -3000, -2000, -1000, 0, 1000, 2000, 3000, 4000, 5000]
        self.action_dict = dict(enumerate(act_disc_states))

    def instantiate(self):
        self.vrs = {v.name: v.valueReference for v in self.model_description.modelVariables}
        self.unzipdir = extract(self.FMU_PATH)
        self.fmu = FMU2Slave(
            guid=self.model_description.guid,
            unzipDirectory=self.unzipdir,
            modelIdentifier=self.model_description.coSimulation.modelIdentifier,
            instanceName="instance1",
        )
        print("Environment Instantiated")

    def reset_env(self, start_time, stop_time):
        self.fmu.instantiate()
        self.fmu.reset()
        self.fmu.setupExperiment(startTime=start_time, stopTime=stop_time)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()
        print(f"Environment Reset (Start={start_time}, Stop={stop_time})")

    def terminate_env(self):
        self.fmu.terminate()
        self.fmu.freeInstance()
        shutil.rmtree(self.unzipdir, ignore_errors=True)
        print("Environment Terminated\n")

    def get_state(self):
        Tin = round(self.fmu.getReal([self.vrs["Tin"]])[0], 1)
        Tout = round(self.fmu.getReal([self.vrs["Tout"]])[0], 1)
        return Tin, Tout

    def do_step(self, current_time, action, main_step_size, step_size):
        self.fmu.setReal([self.vrs["Q"]], [float(action)])
        self.fmu.doStep(currentCommunicationPoint=current_time, communicationStepSize=step_size)
        return time.time(), current_time + step_size

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


class Neural_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(3, 200)
        self.layer_2 = nn.Linear(200, 200)
        self.layer_3 = nn.Linear(200, 11)

    def forward(self, x, device):
        x = torch.Tensor(x).to(device)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)


#%% ⭐ UPDATED — Train Agent with quarterly start/stop times and organized output
class DQN_Train():
    def __init__(self, episodes, title, work_dir_path, start_time, stop_time):
        self.episodes = episodes
        self.title = title
        self.work_dir_path = work_dir_path
        self.start_time = start_time
        self.stop_time = stop_time

        # ⭐ Create a results folder
        self.results_folder = os.path.join(self.work_dir_path, f"{self.title}_results")
        os.makedirs(self.results_folder, exist_ok=True)
        print(f"Created results folder: {self.results_folder}")

    def train_agent(self):
        for episode in range(self.episodes):
            print(f'Running Episode: {episode}')
            episode_folder = os.path.join(self.results_folder, f"episode_{episode}")
            os.makedirs(episode_folder, exist_ok=True)

            quarter_script_path = os.path.join(self.work_dir_path, "Deep_Q_Single_Episode_Quarter.py")

            subprocess.run([
                sys.executable,
                quarter_script_path,
                str(episode),
                self.title,
                episode_folder,
                str(self.episodes),
                str(self.start_time),
                str(self.stop_time)
            ], env=env, cwd=self.work_dir_path)


    def get_training_results(self):
        set_working_directory(self.results_folder)
        all_rewards, all_Tin, all_Tout, all_paths = [], [], [], []

        for i in range(self.episodes):
            # 🔹 NEW: look inside each episode subfolder
            episode_folder = os.path.join(self.results_folder, f"episode_{i}")
            filename = os.path.join(episode_folder, f"{self.title}_episode_{i}_training_results.json")

            if os.path.exists(filename):
                with open(filename, "r") as f:
                    data = json.load(f)
                    all_rewards.append(data['Rewards'])
                    all_Tin.append(data['T_in'])
                    all_Tout.append(data['T_out'])
                    all_paths.append(data['Saved path'])
                    print(f"Loaded {filename}")
            else:
                print(f"Warning: Missing file {filename}")

        return {"T_in": all_Tin, "T_out": all_Tout, "Rewards": all_rewards, "Saved paths": all_paths}
  

    def plot_training_results(self, training_results):
        fig, ax = plt.subplots()
        ax.plot(training_results['Rewards'])
        ax.set_title(f'{self.title} - Training Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel("Reward")
        ax.grid(True)
        plt.savefig(os.path.join(self.results_folder, f"{self.title}_training_rewards.png"))
        print(f"Saved training plot to {self.results_folder}")




#%% Test Agent Class

class DQN_Test:
    def __init__(self, test_config):
        self.test_config = test_config
        self.title = test_config.get('title')
        self.env = test_config.get("test_env")
        self.test_start_time = test_config.get("test_start_time")
        self.test_stop_time = test_config.get("test_stop_time", 31104000)
        self.test_number_of_steps = test_config.get("test_number_of_steps")
        self.main_step_size = test_config.get("main_step_size")
        self.step_size = test_config.get("step_size")
        self.temp_factor = test_config.get("temp_factor")
        self.energy_factor = test_config.get("energy_factor")
        self.occupied_setpoint = test_config.get('occupied_setpoint')
        self.unoccupied_setpoint = test_config.get('unoccupied_setpoint')
        
        self.device, self.dev = self.get_device()
        self.device = torch.device(self.device)
        self.setpoints = self.setpoints()
        
    def get_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU is available and being used")
            dev = 'GPU'
        else:
            device = torch.device("cpu")
            print("GPU is not available, using CPU instead")
            dev = 'CPU'
        return device, dev
    
                
    def setpoints(self):
        # One constant setpoint for entire test
        constant_setpoint = self.occupied_setpoint   # or choose unoccupied_setpoint if you prefer
        setpoints = [constant_setpoint for _ in range(self.test_number_of_steps)]

        # Optional plot
        fig, ax = plt.subplots()
        ax.plot(setpoints)
        ax.set_title('Constant Setpoint')
        ax.set_xlabel('TimeStep')
        ax.set_ylabel('Setpoint (C)')
        ax.grid(True)

        return setpoints

    def test_agent(self, test_path):
        print('Testing Agent\n')
        
        # Save metadata
        test_config_string = {k:str(v) for k,v in self.test_config.items()}
        test_meta = test_config_string
        test_meta['setpoints'] = self.setpoints
        test_meta['FMU_path'] = self.env.FMU_PATH
        
        # Load test network
        test_network = Neural_Network().to(self.device)
        test_network.load_state_dict(torch.load(test_path))
        test_network.eval()
    
        test_time_start = time.time()
      
        # Initialise environment
        self.env.instantiate()
        
        # Reset environment 
        self.env.reset_env(self.test_start_time, self.test_stop_time)
      
        # Reset episode parameters
        test_sum_reward = 0
        current_time = self.test_start_time
        
        # Episode data collectors
        test_actions = []
        test_indoor_temps = []
        test_outdoor_temps = []
        test_rewards = []
        
        # Get initial state
        Tin, Tout = self.env.get_state()
        current_state=np.array([Tin, Tout, self.setpoints[0]], dtype=np.float32)
        test_indoor_temps.append(Tin)
        test_outdoor_temps.append(Tout)
            
        for step in range(self.test_number_of_steps):              
            
            # Select action
            est_q_vals = test_network.forward(current_state.reshape((1,) + current_state.shape), self.device)
            discrete_action = torch.argmax(est_q_vals, dim=1).tolist()[0]
            action = self.env.action_dict[discrete_action]
            test_actions.append(float(action)) 
        
            # Do step
            do_step_time, current_time = self.env.do_step(current_time, action, self.main_step_size, self.step_size)
            
            # Get next state
            Tin, Tout = self.env.get_state()
            test_indoor_temps.append(Tin)
            test_outdoor_temps.append(Tout)
            if step==self.test_number_of_steps-1:
                next_setpoint=self.setpoints[0]
            else:
                next_setpoint=self.setpoints[step+1]
            next_state=np.array([Tin, Tout, next_setpoint], dtype=np.float32)
            
            # Calculate reward
            step_reward = self.env.reward(self.temp_factor, self.energy_factor, next_state[0], next_setpoint, action)
            test_sum_reward += step_reward
            test_rewards.append(step_reward)
        
            # update current state to new state
            current_state = np.copy(next_state)
            
        # Terminate environment
        self.env.terminate_env()
          
        # Get run runtime
        test_time_end = time.time()
        test_time = test_time_end - test_time_start
        print(f"Test Reward: {test_sum_reward:.3f}")
        print(f"Test runtime: {test_time:.3f} seconds\n")
        
        test_results = {'Meta':test_meta,
                        'Rewards':test_rewards,
                        'T_in':test_indoor_temps,
                        'T_out':test_outdoor_temps,
                        'Actions':test_actions}
      
        return test_results


    def plot_test_results(self, test_data):
        # Create output folder if missing
        os.makedirs("Test_Plots", exist_ok=True)

        # Plot 1️⃣: Rewards over the full year
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.cumsum(test_data['Rewards']))
        ax.set_title(f'{self.title} — Reward over Full Year')
        ax.set_xlabel('Months')
        ax.set_ylabel('Reward')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"Test_Plots/{self.title}_Rewards_Over_Year.png", dpi=300)
        plt.close()
        print(f"✅ Saved reward plot: Test_Plots/{self.title}_Rewards_Over_Year.png")


        # Plot 2️⃣: Indoor & Outdoor Temperatures with Setpoints
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(test_data['T_in'], label='Indoor Temp (T_in)', linewidth=1.2)
        ax.plot(test_data['T_out'], color='green', label='Outdoor Temp (T_out)', linewidth=1.2, alpha = 0.4)
        ax.plot(self.setpoints, label='Setpoint', linestyle='--', color='orange', linewidth=1.2)
        #ax.fill_between(range(len(self.setpoints)), lower_bound, upper_bound, color='orange', alpha=0.1, label='Comfort Band (±1°C)')

        # ----- MONTH AXIS -----

        # Steps per month (approx; assuming 30-day months)
        steps_per_month = int((30 * 24 * 3600) / 900)   # = 2880

        # Tick positions
        month_ticks = [i * steps_per_month for i in range(13)]  # 0 to 12 months

        # Labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', '']

        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels)

        ax.set_title(f'{self.title} — Indoor/Outdoor Temperature over Full Year')
        ax.set_xlabel('Timestep (900 s)')
        ax.set_ylabel('Temperature (°C)')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"Test_Plots/{self.title}_Temperature_Over_Year.png", dpi=300)
        plt.close()
        print(f"✅ Saved temperature plot: Test_Plots/{self.title}_Temperature_Over_Year.png")

        
    # Test all paths
    def test_all_paths(self, data):
        all_paths_test_results = {}
        all_test_rewards = []
        all_training_rewards = []
        all_test_episodes = []
        for episode, path in zip(range(len(data['Saved paths'])), data['Saved paths']):
            print(f'Testing episode {episode}')
            test_results = self.test_agent(path)
            test_reward_sum = sum(test_results['Rewards'])
            all_test_rewards.append(test_reward_sum)
            all_paths_test_results[f'{episode}'] = test_results
            training_reward = data['Rewards'][episode]
            all_training_rewards.append(training_reward)
            all_test_episodes.append(episode)
        
        best_reward = max(all_test_rewards)
        best_index = all_test_rewards.index(best_reward)
        best_episode = all_test_episodes[best_index]
        
        print(f"Best Test Episode: {best_episode:.3f}")
        print(f"Best Reward: {best_reward:.3f}")
        
        fig, ax = plt.subplots()
        ax.scatter(all_training_rewards, all_test_rewards)
        ax.set_title(f'{self.title} - Training vs Testing Rewards')
        ax.set_xlabel('Training Reward')
        ax.set_ylabel("Testing reward")
        ax.grid()
        
        return all_paths_test_results, best_episode



#%% Set working directory helper
def set_working_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    os.chdir(path)
    print(f"Working directory changed to:\n{os.getcwd()}\n")

def save_results(data, title):
    with open(f"{title}.json", "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    EPISODES = 150
    TITLE = "Deep_Q_EP_Toy_Problem"
    WORK_DIR = "/Users/luisrubio/Desktop/Energy"
    FMU_PATH = f"{WORK_DIR}/_fmu_export_schedule_V940.fmu"

    # ==========================
    #  TRAIN EACH QUARTER
    # ==========================
    quarter_times = {
        "Q1": (0, 7776000),          # Jan–Mar
        "Q2": (7776000, 15552000),   # Apr–Jun
        "Q3": (15552000, 23328000),  # Jul–Sep
        "Q4": (23328000, 31104000),  # Oct–Dec
    }

    quarterly_training_results = {}

    for quarter, (start_t, stop_t) in quarter_times.items():
        title_q = f"{TITLE}_{quarter}"
        print(f"\n=== TRAINING {quarter}: {start_t}–{stop_t} seconds ===")

        RL_Train = DQN_Train(EPISODES, title_q, WORK_DIR, start_t, stop_t)
        RL_Train.train_agent()
        training_results = RL_Train.get_training_results()
        RL_Train.plot_training_results(training_results)

        quarterly_training_results[quarter] = training_results
        print(f"Finished training for {quarter}\n")

    # ==========================
    #  TEST EACH QUARTERLY MODEL OVER FULL YEAR
    # ==========================


    # Whole year time window (0 → 31,104,000 seconds)
    TEST_START = 0
    TEST_STEPS = int((31104000) / 900)  # 900 s per step → 34,560 steps for 1 year

    for quarter, training_results in quarterly_training_results.items():
        print(f"\n=== TESTING {quarter} model over full year ===")

        test_env = environment(FMU_PATH)
        test_config = {
            "title": f"{TITLE}_{quarter}_FullYearTest",
            "test_env": test_env,
            "test_start_time": TEST_START,
            "test_stop_time": 31104000,
            "test_number_of_steps": TEST_STEPS,
            "main_step_size": 1,
            "step_size": 900,
            "temp_factor": 1,
            "energy_factor": -0.001,
            "occupied_setpoint": 20,
            "unoccupied_setpoint": 15,
        }

        RL_Test = DQN_Test(test_config)
        all_paths_test_results, best_episode = RL_Test.test_all_paths(training_results)
        RL_Test.plot_test_results(all_paths_test_results[str(best_episode)])
        save_results(all_paths_test_results, f"{TITLE}_{quarter}_FullYear_results")

        print(f"Completed full-year test for {quarter}\n")
