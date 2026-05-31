# BERL Research: Deep Q-Learning HVAC Controller

This repository contains reinforcement learning code developed for research at Columbia University's Building Energy Research Lab. The project uses a Deep Q-Network (DQN) to control HVAC heating/cooling actions in a simulated building environment.

The goal is to train an agent that can keep indoor temperature near a target setpoint while reducing unnecessary HVAC energy use.

## Project Overview

The system connects a reinforcement learning agent to a building simulation exported as an FMU. At each timestep, the agent observes the current building state and chooses a discrete HVAC action. The FMU then updates the building temperature based on that action.

The agent is trained to balance two objectives:

1. Maintain indoor comfort near the temperature setpoint.
2. Penalize high HVAC power usage.

## How It Works

The DQN receives a state vector containing:

* Indoor temperature
* Outdoor temperature
* Temperature setpoint

It chooses from 11 discrete HVAC actions:

```text
[-5000, -4000, -3000, -2000, -1000, 0, 1000, 2000, 3000, 4000, 5000]
```

These actions represent different heating/cooling power levels applied to the building simulation.

The reward function gives positive reward when indoor temperature stays within ±1°C of the setpoint and penalizes larger HVAC energy usage.

## Training Approach

The model uses:

* Deep Q-Learning
* PyTorch neural network policy
* Experience replay buffer
* Target network updates
* Epsilon-greedy exploration
* Quarterly training windows

The training pipeline splits the year into four quarters:

* Q1: January–March
* Q2: April–June
* Q3: July–September
* Q4: October–December

Each quarterly model is trained separately, then tested over a full simulated year to evaluate generalization across seasons.

## Repository Structure

```text
BERL-Research/
├── Deep_Q_Single_Episode_Quarter.py   # Runs one DQN training episode for a selected quarter
├── main_runner_quarter.py             # Main training/testing pipeline for quarterly experiments
└── README.md
```

## Main Files

### `Deep_Q_Single_Episode_Quarter.py`

Runs a single training episode for a specific time window. It:

* Loads the FMU simulation environment
* Initializes the DQN policy and target networks
* Selects HVAC actions using epsilon-greedy exploration
* Stores transitions in a replay buffer
* Updates the policy network
* Saves trained weights and episode results

### `main_runner_quarter.py`

Coordinates the full experiment pipeline. It:

* Trains DQN agents on quarterly time windows
* Aggregates training rewards
* Tests saved models over a full-year simulation
* Generates plots for reward, temperature tracking, comfort excursions, and energy usage

## Outputs

The pipeline generates:

* Saved PyTorch model weights
* JSON training results
* Training reward plots
* Full-year temperature tracking plots
* Comfort band compliance analysis
* HVAC energy usage estimates
* Excursion timelines showing when temperature leaves the comfort band

## Technologies Used

* Python
* PyTorch
* NumPy
* Matplotlib
* FMPy
* EnergyPlus FMU co-simulation

## Research Context

This project was developed as part of research into using reinforcement learning for building energy optimization. HVAC systems are a major source of building energy demand, and better control policies can reduce energy usage while maintaining occupant comfort.

## Note

This repository contains research code and depends on a local FMU file exported from EnergyPlus. The current version is not packaged as a standalone application.
