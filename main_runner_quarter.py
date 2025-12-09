#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

        # Create discrete action space (heating/cooling power levels)
        act_disc_states = [-5000, -4000, -3000, -2000, -1000, 0, 1000, 2000, 3000, 4000, 5000]
        self.action_dict = dict(enumerate(act_disc_states))

    def instantiate(self):
        # Map variable names to value references
        self.vrs = {v.name: v.valueReference for v in self.model_description.modelVariables}

        # Extract FMU
        self.unzipdir = extract(self.FMU_PATH)

        # Instantiate FMU as co-simulation slave
        self.fmu = FMU2Slave(
            guid=self.model_description.guid,
            unzipDirectory=self.unzipdir,
            modelIdentifier=self.model_description.coSimulation.modelIdentifier,
            instanceName="instance1",
        )
        print("Environment Instantiated")

    def reset_env(self, start_time, stop_time):
        # Reset and initialize FMU for a given simulation window
        self.fmu.instantiate()
        self.fmu.reset()
        self.fmu.setupExperiment(startTime=start_time, stopTime=stop_time)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()
        print(f"Environment Reset (Start={start_time}, Stop={stop_time})")

    def terminate_env(self):
        # Terminate FMU and clean temporary directory
        self.fmu.terminate()
        self.fmu.freeInstance()
        shutil.rmtree(self.unzipdir, ignore_errors=True)
        print("Environment Terminated\n")

    def get_state(self):
        # Get indoor and outdoor temperature
        Tin = round(self.fmu.getReal([self.vrs["Tin"]])[0], 1)
        Tout = round(self.fmu.getReal([self.vrs["Tout"]])[0], 1)
        return Tin, Tout

    def do_step(self, current_time, action, main_step_size, step_size):
        # Apply action (Q) and advance FMU one step in time
        self.fmu.setReal([self.vrs["Q"]], [float(action)])
        self.fmu.doStep(currentCommunicationPoint=current_time, communicationStepSize=step_size)
        return time.time(), current_time + step_size

    def reward(self, temp_factor, energy_factor, temp_next, next_setpoint, action):
        # Reward based on comfort band and HVAC energy use
        upper = next_setpoint + 1
        lower = next_setpoint - 1

        if lower <= temp_next <= upper:
            temp_reward = 10
        elif temp_next < lower:
            temp_reward = temp_next - (lower - 1)
        else:
            temp_reward = -temp_next + (upper + 1)

        # Note: energy_factor should generally be negative so larger |action| is penalized
        return temp_factor * temp_reward + energy_factor * abs(action)


class Neural_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(3, 200)
        self.layer_2 = nn.Linear(200, 200)
        self.layer_3 = nn.Linear(200, 11)

    def forward(self, x, device):
        # Forward pass of DQN: state -> Q-values for each discrete action
        x = torch.Tensor(x).to(device)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)


# Train Agent with quarterly start/stop times and organized output
class DQN_Train():
    def __init__(self, episodes, title, work_dir_path, start_time, stop_time):
        self.episodes = episodes
        self.title = title
        self.work_dir_path = work_dir_path
        self.start_time = start_time
        self.stop_time = stop_time

        # Create a results folder for this training run
        self.results_folder = os.path.join(self.work_dir_path, f"{self.title}_results")
        os.makedirs(self.results_folder, exist_ok=True)
        print(f"Created results folder: {self.results_folder}")

    def train_agent(self):
        # Launch quarterly single-episode script for each episode
        for episode in range(self.episodes):
            print(f"Running Episode: {episode}")
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
        # Aggregate results from all episode subfolders
        set_working_directory(self.results_folder)
        all_rewards, all_Tin, all_Tout, all_paths = [], [], [], []

        for i in range(self.episodes):
            # Look inside each episode subfolder
            episode_folder = os.path.join(self.results_folder, f"episode_{i}")
            filename = os.path.join(episode_folder, f"{self.title}_episode_{i}_training_results.json")

            if os.path.exists(filename):
                with open(filename, "r") as f:
                    data = json.load(f)
                    all_rewards.append(data["Rewards"])
                    all_Tin.append(data["T_in"])
                    all_Tout.append(data["T_out"])
                    all_paths.append(data["Saved path"])
                    print(f"Loaded {filename}")
            else:
                print(f"Warning: Missing file {filename}")

        return {"T_in": all_Tin, "T_out": all_Tout, "Rewards": all_rewards, "Saved paths": all_paths}

    def plot_training_results(self, training_results, y_min=None, y_max=None):
        # Plot total reward per episode for this training run
        fig, ax = plt.subplots()

        rewards = training_results["Rewards"]
        episode_totals = [float(r) for r in rewards]   # each r is a scalar total reward
        ax.plot(episode_totals)

        # Apply global limits if provided
        if (y_min is not None) and (y_max is not None):
            ax.set_ylim(y_min, y_max)

        ax.set_xlim(0, len(rewards) - 1)
        ax.set_title(f"{self.title} - Training Rewards")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True)

        # Save plot to this training run's results folder
        plt.savefig(os.path.join(self.results_folder, f"{self.title}_training_rewards.png"))
        plt.close()


# Test Agent Class
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
        self.setpoints = self.generate_setpoints()
        
    def get_device(self):
        # Choose GPU if available, otherwise CPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("GPU is available and being used")
            dev = 'GPU'
        else:
            device = torch.device("cpu")
            print("GPU is not available, using CPU instead")
            dev = 'CPU'
        return device, dev
    
    def generate_setpoints(self):
        # One constant setpoint for entire test (can change to unoccupied_setpoint if desired)
        constant_setpoint = self.occupied_setpoint
        setpoints = [constant_setpoint for _ in range(self.test_number_of_steps)]

        # Optional preview plot of setpoints
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
        test_config_string = {k: str(v) for k, v in self.test_config.items()}
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
        current_state = np.array([Tin, Tout, self.setpoints[0]], dtype=np.float32)
        test_indoor_temps.append(Tin)
        test_outdoor_temps.append(Tout)
            
        for step in range(self.test_number_of_steps):              
            
            # Select action (greedy) from DQN
            est_q_vals = test_network.forward(current_state.reshape((1,) + current_state.shape), self.device)
            discrete_action = torch.argmax(est_q_vals, dim=1).tolist()[0]
            action = self.env.action_dict[discrete_action]
            test_actions.append(float(action)) 
        
            # Do step in FMU
            do_step_time, current_time = self.env.do_step(current_time, action, self.main_step_size, self.step_size)
            
            # Get next state
            Tin, Tout = self.env.get_state()
            test_indoor_temps.append(Tin)
            test_outdoor_temps.append(Tout)

            # Advance setpoint index
            if step == self.test_number_of_steps - 1:
                next_setpoint = self.setpoints[0]
            else:
                next_setpoint = self.setpoints[step + 1]

            next_state = np.array([Tin, Tout, next_setpoint], dtype=np.float32)
            
            # Calculate reward
            step_reward = self.env.reward(self.temp_factor, self.energy_factor, next_state[0], next_setpoint, action)
            test_sum_reward += step_reward
            test_rewards.append(step_reward)
        
            # Update current state to new state
            current_state = np.copy(next_state)
            
        # Terminate environment
        self.env.terminate_env()
          
        # Get run runtime
        test_time_end = time.time()
        test_time = test_time_end - test_time_start
        print(f"Test Reward: {test_sum_reward:.3f}")
        print(f"Test runtime: {test_time:.3f} seconds\n")
        
        test_results = {
            'Meta': test_meta,
            'Rewards': test_rewards,
            'T_in': test_indoor_temps,
            'T_out': test_outdoor_temps,
            'Actions': test_actions
        }
      
        return test_results
    
    def plot_test_results(self, test_data):
        # Create output folder for test plots
        os.makedirs("Test_Plots", exist_ok=True)

        # Extract test arrays
        T_in_raw = test_data['T_in']
        T_out_raw = test_data['T_out']
        actions = np.array(test_data['Actions'])
        T_set = np.array(self.setpoints)

        # Fix T_in length mismatch (extra first element)
        if len(T_in_raw) == len(T_set) + 1:
            T_in_raw = T_in_raw[1:]

        T_in = np.array(T_in_raw)
        T_out = np.array(T_out_raw)
        deviation = T_in - T_set

        dt = 900  # seconds per timestep (15 min)
        nsteps = len(T_in)

        # Plot 1: Cumulative Reward
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.cumsum(test_data['Rewards']))
        ax.set_title(f'{self.title} — Reward over Full Year')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Cumulative Reward')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"Test_Plots/{self.title}_Rewards_Over_Year.png", dpi=300)
        plt.close()

        print("Saved reward plot")

        # Plot 2: Indoor and Outdoor Temperature vs Setpoint
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(T_in, label='Indoor Temp', linewidth=1.2)
        ax.plot(T_out, label='Outdoor Temp', color='green', alpha=0.4)
        ax.plot(T_set, label='Setpoint', linestyle='--', color='orange', linewidth=1.2)

        # Month ticks (approximate)
        steps_per_month = int((30 * 24 * 3600) / dt)  # = 2880
        month_ticks = [i * steps_per_month for i in range(13)]
        month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec','']
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels)

        ax.set_title(f'{self.title} — Temperature Over Full Year')
        ax.set_ylabel("°C")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"Test_Plots/{self.title}_Temperature_Over_Year.png", dpi=300)
        plt.close()

        print("Saved temperature plot")

        # Detect excursions outside comfort band (> 1 hour)
        outside = np.abs(deviation) > 1.0
        excursions = []
        current_start = None
        count = 0

        for i, out in enumerate(outside):
            if out:
                if current_start is None:
                    current_start = i
                count += 1
            else:
                if current_start is not None and count >= 4:  # 4 steps = 1 hr (4 x 15 min)
                    excursions.append((current_start, i - 1))
                current_start = None
                count = 0

        if current_start is not None and count >= 4:
            excursions.append((current_start, len(outside) - 1))

        print(f"\nFound {len(excursions)} excursions > 1 hour:")
        for start, end in excursions:
            mins = (end - start + 1) * (dt / 60)
            print(f"  - Steps {start} → {end}  ({mins:.0f} minutes)")

        # Excursion timeline plot
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.plot([0, nsteps], [0, 0], color='black', linewidth=1)

        for start, end in excursions:
            ax.axvspan(start, end, color='red', alpha=0.4)

        ax.set_title(f"{self.title} — Excursion Timeline (>1 hour)")
        ax.set_xlabel("Timestep (900 s)")
        ax.set_yticks([])
        ax.set_xlim(0, nsteps)
        ax.grid(True, axis='x', alpha=0.2)

        plt.tight_layout()
        fpath = f"Test_Plots/{self.title}_ExcursionTimeline.png"
        plt.savefig(fpath, dpi=300)
        plt.close()

        print("Saved excursion timeline plot")

        # Comfort band compliance summary
        within_band = np.abs(deviation) <= 1.0
        percent_within = 100 * np.sum(within_band) / nsteps

        print(f"\nComfort Band Compliance: {percent_within:.2f}% within ±1°C")

        # Deviation stem plot
        fig, ax = plt.subplots(figsize=(10, 4))
        markerline, stemlines, baseline = ax.stem(deviation)
        markerline.set_marker('')        # remove circles
        baseline.set_color('none')       # remove baseline
        plt.setp(stemlines, 'color', 'blue')

        ax.axhline(0, color='black', linewidth=1)
        ax.axhline(1, color='red', linestyle='--', alpha=0.6)
        ax.axhline(-1, color='red', linestyle='--', alpha=0.6)

        ax.set_title(f"{self.title} — Deviation From Setpoint")
        ax.set_ylabel("Tin - Tset (°C)")
        ax.set_xlabel("Timestep (900 s)")
        ax.grid(True)
        plt.tight_layout()

        fpath = f"Test_Plots/{self.title}_Deviation_StemPlot.png"
        plt.savefig(fpath, dpi=300)
        plt.close()

        print("Saved deviation stem plot")

        # Energy metrics: total kWh and kWh per comfort hour
        E_J = np.sum(np.abs(actions) * dt)
        E_kWh = E_J / (3600 * 1000)

        print(f"\nTotal HVAC Energy: {E_kWh:.2f} kWh")

        comfort_hours = np.sum(within_band) * dt / 3600.0
        kWh_per_comfort_hr = E_kWh / comfort_hours if comfort_hours > 0 else float('inf')

        print(f"kWh per comfort-hour: {kWh_per_comfort_hr:.4f}\n")
    
    
    def test_all_paths(self, data):
        # Test all saved models from training and compare rewards
        all_paths_test_results = {}
        all_test_rewards = []
        all_training_rewards = []
        all_test_episodes = []

        num_episodes = len(data['Saved paths'])

        for episode in range(num_episodes):
            path = data['Saved paths'][episode]
            print(f'Testing episode {episode}')

            # Test reward (sum of per-step rewards in test_results)
            test_results = self.test_agent(path)
            test_reward_sum = sum(test_results['Rewards'])
            all_test_rewards.append(test_reward_sum)

            # Training reward (scalar from JSON)
            episode_reward_data = data['Rewards'][episode]
            training_reward = float(episode_reward_data)
            all_training_rewards.append(training_reward)

            all_paths_test_results[str(episode)] = test_results
            all_test_episodes.append(episode)

        best_reward = max(all_test_rewards)
        best_index = all_test_rewards.index(best_reward)
        best_episode = all_test_episodes[best_index]

        print(f"\nBest Test Episode: {best_episode}")
        print(f"Best Reward: {best_reward}")

        fig, ax = plt.subplots()
        ax.scatter(all_training_rewards, all_test_rewards)
        ax.set_title(f'{self.title} - Training vs Testing Rewards')
        ax.set_xlabel('Training Reward (total)')
        ax.set_ylabel('Testing Reward (total)')
        ax.grid()

        return all_paths_test_results, best_episode


# Set working directory helper
def set_working_directory(path):
    # Create the directory if it does not exist and change into it
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    os.chdir(path)
    print(f"Working directory changed to:\n{os.getcwd()}\n")


def save_results(data, title):
    # Save dictionary data to JSON
    with open(f"{title}.json", "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    import time
    script_start = time.time()
    print("\nStarting full pipeline...\n")

    EPISODES = 400
    TITLE = "Deep_Q_EP_Toy_Problem"
    WORK_DIR = "/Users/luisrubio/Desktop/Energy"
    FMU_PATH = f"{WORK_DIR}/_fmu_export_schedule_V940.fmu"

    # Train each quarter (Q1–Q4) on its own time window
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
        # Plotting will be handled after all quarters are trained
        quarterly_training_results[quarter] = training_results
        print(f"Finished training for {quarter}\n")

    # After all training: compute global y-limits for reward plots
    all_rewards_global = []

    for quarter in quarterly_training_results:
        all_rewards_global.extend(quarterly_training_results[quarter]["Rewards"])

    global_ymin = min(all_rewards_global)
    global_ymax = max(all_rewards_global)

    print("\nGlobal Y-axis limits for reward plots:")
    print("Min:", global_ymin, "Max:", global_ymax)

    # Plot all quarter training curves with unified axes
    for quarter, training_results in quarterly_training_results.items():
        title_q = f"{TITLE}_{quarter}"

        # Create a temporary train object just for plotting
        RL_Train_Plot = DQN_Train(EPISODES, title_q, WORK_DIR, 0, 0)

        RL_Train_Plot.plot_training_results(
            training_results,
            y_min=global_ymin,
            y_max=global_ymax
        )

    # Test each quarterly model over full year

    # Whole year time window (0 to 31,104,000 seconds)
    TEST_START = 0
    TEST_STEPS = int(31104000 / 900)  # 900 s per step → 34,560 steps for 1 year

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
            "energy_factor": -0.01,
            "occupied_setpoint": 20,
            "unoccupied_setpoint": 15,
        }

        RL_Test = DQN_Test(test_config)
        all_paths_test_results, best_episode = RL_Test.test_all_paths(training_results)
        RL_Test.plot_test_results(all_paths_test_results[str(best_episode)])
        save_results(all_paths_test_results, f"{TITLE}_{quarter}_FullYear_results")

        print(f"Completed full-year test for {quarter}\n")
    
    script_end = time.time()
    total_seconds = script_end - script_start

    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60

    print("\nFull pipeline complete")
    print(f"Total runtime: {hours} hr {minutes} min {seconds:.1f} sec\n")
