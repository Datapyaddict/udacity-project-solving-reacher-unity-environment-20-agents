# Unity / Reacher environment / Continuous Control and Distributed Training 
### (Udacity's Deep Reinforcement Learning project)
***

## Table of Contents
[1. Project Requirements](#project_requirement) 
- [1.1 Unity environments](#unity_environment) 
- [1.2 Reacher environment](#reacher_environment)
- [1.3 Solving the "Reacher" environment](#solving_environment)

[2. Required Python libraries](#libraries)  
[3. Solution implemented](#solution)  
[4. Script: `shared_ddpg.py`](#shared_ddpg)  
- [4.1 Overview](#shared_ddpg_overview)  
- [4.2 CLI arguments](#shared_ddpg_arguments)  
- [4.3 Logging](#shared_ddpg_logging)  
- [4.4 Plots](#plots)  
- [4.5 CSV summaries](#csv_summaries) 
- [4.6 JSON files](#json_files) 
- [4.7 Model checkpoints](#model_checkpoints) 
- [4.8 2-step runs (commands)](#shared_ddpg_runs)

[5. Script: `extract_weights_n_load_into_policy.py`](#extract_weights_n_load_into_policy)  
> [5.1 Overview](#extract_weights_n_load_into_policy_overview)   
> [5.2 CLI arguments ](#extract_weights_n_load_into_policy_arguments)  
> [5.3 runs](#extract_weights_n_load_into_policy_run)  

[6. Repository structure (key items)](#repo_structure)  


***

<a id = 'project_requirement'></a>
# 1. Project Requirements

<a id = 'unity_environment'></a>
## 1.1 Unity environments

Unity Machine Learning Agents (ML-Agents) is an open-source Unity plugin that enables games and simulations to serve as environments for training intelligent agents.
For game developers, these trained agents can be used for multiple purposes, including controlling NPC behaviour (in a variety of settings such as multi-agent and adversarial), automated testing of game builds and evaluating different game design decisions pre-release.

<a id = 'reacher_environment'></a>
## 1.2. Reacher environment

This project aims to solve the Unity environment called **"Reacher"** provided by Udacity. It can be downloaded from one of the links below:
- [Linux]( https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- [Mac OSX]( https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- [Windows (32-bit)]( https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- [Windows (64-bit)]( https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Note: this repository already includes a Windows build at `./20_Agents_Reacher_Windows_x86_64/`.


<a id = 'solving_environment'></a>
## 1.3 Solving the "Reacher" environment

For this project, we will be working with the Reacher environment.

In this environment, a double-jointed arm can move to target locations. A reward of `+0.1` is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of `33 variables` corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with `four continuous numbers`, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between `-1` and `1`.

The current experiment aims to solve the version of the environment that contains `20 identical agents`, `each with its own copy of the environment`. The agents must get an average score of `+30 (over 100 consecutive episodes, and over all agents)`. Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields `20 (potentially different) scores`. We then take `the average of these 20 scores`.
- This yields an `average score for each episode` (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

It has been shown that having multiple copies of the same agent [sharing experience can accelerate learning](https://ai.googleblog.com/2016/10/how-robots-can-acquire-new-skills-from.html)

<img src="pictures/reacher.gif" width="500"/>

As an indication, in the solution implemented by Udacity, the average score over 100 episodes has reached the threshold of `+30` at episode `163`.
At one point, it was accomplished an average score (over 100 episodes) of around +39!

<img src="pictures/reacher_udacity_performance.png" width="500"/>

Our current solution aims to outperform Udacity's performance by reaching the threshold earlier than episode `163` during the training phase. 

<a id = 'libraries'></a>
# 2.Required Python libraries


The requirements used for the experiment are:

- Matplotlib==3.5.3
- Numpy==1.18.5
- Torch==1.13.1+cu116
- Unityagents==0.4.0
- Python==3.7.6

Important note: `unityagents` is the legacy Python API used here to talk to a Unity-built environment `Reacher.exe`. In that older API, the “brain” concept is specific to the older ML-Agents toolchain.

Do **not** use the "Reacher" version provided by Unity on the [Unity ML-Agents GitHub page]( https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md).

<a id = 'solution'></a>
# 3. Solution implemented

The solution uses a **MADDPG (Multi Agent Deep Deterministic Policy Gradient)** agent, which relies on:

- an online **shared Actor** (policy) network and an online **Central Critic** (producing 2 Q-values) network.
- a shaped team reward based on the environment rewards.
- plus **target** networks (actor and central critic).

See [report.pdf](https://github.com/Datapyaddict/udacity-project-solving-multi-agent-tennis-unity-environment/blob/main/report.pdf) for the methodology and results.

Two scripts are used:

- `shared_actor_maddpg_with_team_reward.py`
- `extract_weights_n_load_into_policy.py`

<a id = 'shared_actor_maddpg_with_team_reward'></a>
# 4. Script: `shared_ddpg.py`

<a id = 'shared_actor_maddpg_with_team_reward_overview'></a>
## 4.1 Overview

This script is an end-to-end training runner for a DDPG agent on the Unity Reacher environment, handling a batch of 20 agents.

It can run:

- one fixed set of hyperparameters (the default configuration), or
- a randomly sampled hyperparameter search over a predefined grid,

and it saves checkpoints, logs, plots, and CSV/JSON summaries under a timestamped results folder:

`results/collab_compet/run_YYYYMMDD_HHMMSS/`

In the repository, 2 runs are referenced:

- Hyperparameter search: `run_HP_SEARCH`
- Training across seeds: `run_TRAINING_PER_SEED`

The following features are elaborated in [report.pdf](https://github.com/Datapyaddict/udacity-project-solving-multi-agent-tennis-unity-environment/blob/main/report.pdf):

- Shared Actor (`FCDP`) and Critic (`FCQV`) per trial/seed.
- Gaussian noise exploration for action selection.
- Replay buffer storing online transitions.
- Training with based-exploration strategy to build the replay buffer.
- Offline optimization via minibatch sampling from the replay buffer.
- Target networks.
- Early stopping (goal performance, max time, max episodes).
- 1-episode evaluation after each training episode using a greedy (no-noise) strategy.
- Final evaluation using 100 episodes with a greedy strategy.

The script also resolves `Reacher.exe` via:

- `--env-exe` CLI argument, or
- `UNITY_REACHER_EXE` environment variable, or
- common local paths / filesystem search

It launches Unity in a separate process (fault-tolerant): training interacts with Unity through `UnityVectorEnv`, which communicates via a multiprocessing pipe and can respawn the worker with a new `worker_id` if Unity crashes or there are port conflicts.

Built-in defaults:

- `gamma=0.99`
- `max_minutes=180`
- `max_episodes=200`
- `goal_mean_100_reward=30`
- `final_eval_episodes=100`

<a id = 'shared_actor_maddpg_with_team_reward_arguments'></a>
## 4.2 CLI arguments

- `--search`: enables hyperparameter search over the search grid. If not provided, the script runs a single default configuration.

Search grid:

    - `hidden_dims`: `[(64, 64), (128,128), (256,256)]`
    - `optimizer`: `['Adam']`
    - `lr`: `[1e-4, 3e-4, 5e-4]`
    - `buffer_size`: `[100000]`
    - `batch_size`: `[64, 128, 256, 500]`
    - `exploration_noise_ratio`: `[0.05, 0.1, 0.2]`
    - `n_warmup_batches`: `[5, 10]`
    - `target_update_every`: `[2, 5, 10]`
    - `tau`: `[0.001, 0.005, 0.01]`

- `--max-trials <int>` (default: 8): number of randomly sampled trials to run when `--search` is enabled.
- `--seeds <comma-separated-ints>` (default: `12,34,56`): seeds to run.
- `--env-exe <path>`: path to `Reacher.exe`.
- `--cpu`: forces PyTorch CPU.

<a id = 'shared_actor_maddpg_with_team_reward_logging'></a>
## 4.3 Logging (`run.log`)

Path:

- `results/collab_compet/run_.../run.log`

Contents:

- Same `INFO` lines as the console.
- One line per episode during training:
    - `elapsed_time` (HH:MM:SS)
    - `episode` (0-based in the log line)
    - `episode_env_steps`
    - `ep_max_sum_rewards_per_agent (r0, r1)`
    - `ep_max_sum_rewards` (max across the two agents)
    - `mean_reward_over_100_episodes_from_training (mean ± std)` (last up to 100)
    - `mean_exploration_ratio_over_100_episodes_from_training (mean ± std)` (last up to 100)
    - `mean_reward_over_100_episodes_from_eval (mean ± std)` (last up to 100)
- Info about reaching the goal, stopping condition, final evaluation score, and bootstrap CI across seeds.

<a id = 'plots'></a>
## 4.4 Plots

Plots are generated from evaluation rewards (not raw training rewards).

Per trial/seed evaluation plot:

- Directory: `results/collab_compet/run_.../plots/`
- Filename: `evaluation_mean100_{trial_id}_seed_{seed}.png`
- Curves:
    - Eval (per-episode): one greedy 1-episode evaluation score per training episode
    - Eval (100-episode MA): moving average over the last 100 evaluation scores
- A “goal window start” vertical line indicates the first episode of the 100-episode window where the moving average meets or exceeds the goal.

Best-trial plot (only when `--search` is enabled):

- Directory: `results/collab_compet/run_.../plots/`
- Filename: `{trial_id}_evaluation_mean100.png`
- One curve: Eval (100-episode MA) averaged across seeds

<a id = 'csv_summaries'></a>
## 4.5 CSV summaries

`plots_summary.csv`:

- Path: `results/collab_compet/run_.../plots_summary.csv`
- One row per trial/seed about early stopping cause, episode count, and (if solved) moving average at stop.
- Columns:
    - `trial_id`
    - `seed`
    - `cause` (`goal`, `time`, or `episodes`)
    - `episode`
    - `mean_100_eval_at_stop` (only if `cause == 'goal'`)
    - `plot_path`

`hparam_search.csv` (only when `--search` is enabled):

- Path: `results/collab_compet/run_.../hparam_search.csv`
- One row per hyperparameter trial.
- Columns:
    - `trial_id`
    - `avg` (mean of final scores across seeds)
    - `std` (std across seeds; `ddof=1` when multiple seeds; otherwise `0.0`)
    - `best_seed_score`
    - `seeds`
    - all hyperparameters (`hidden_dims`, `optimizer`, `lr`, `buffer_size`, `batch_size`, `exploration_noise_ratio`, `n_warmup_batches`, `target_update_every`, `tau`)

<a id = 'json_files'></a>
## 4.6 JSON files

`best_hparams.json` (only when `--search` is enabled):

- Path: `results/collab_compet/run_.../best_hparams.json`
- The file is a single json object saved after hyperparameter search finishes. It stores the best trial (by highest avg_score)'s performance and hyper-parameters .
- Contents:
    - best trial_id
    - avg_score : the mean of the per-seed final_eval_score values for this trial.
    - std score : the standard deviation across seeds of the per-seed final scores.
    - seeds : the exact seed list used, e.g. [12, 34, 56]
    - Hyperparameters:
        - `hidden_dims` : e.g. [256, 256]
        - `optimizer` (string): e.g. "Adam"
        - `lr` 
        - `buffer_size 
        - `batch_size` 
        - `exploration_noise_ratio` 
        - `n_warmup_batches` 
        - `target_update_every`
        - `tau` 
    - `scores_per_seed`

`solved_evaluation_metric.json` (only if goal is reached):

- For each trial_id/seed run, when the goal is reached, the file displays the episode when the run first meets the “solved” condition based on the evaluation 100-episode moving average.
- Path: per trial/seed agent run directory , e.g.:
  `results/collab_compet/run_.../trials/trial_001/seed_12/solved_evaluation_metric.json`
- Contents:
    - `Episode` : the 1-based training episode index at which the agent is first considered solved (i.e., the first episode where the conditions below are met).
    - `mean_100_eval` : the 100-episodes moving average of the 1-episode evaluation score 

<a id = 'model_checkpoints'></a>
## 4.7 Model checkpoints

- Only the online policy model is checkpointed (not critic, not optimizers).
- Path pattern:
  `results/collab_compet/run.../trials/trial_{trial}/seed_{seed}/checkpoints/online_policy_model.{episode}.tar`. e.g.: online_policy_model.99.tar

For simplicity, only the first and last episode checkpoints are kept in the repository.

<a id = 'shared_actor_maddpg_with_team_reward_runs'></a>
## 4.8 2-step runs (commands)

Our approach to solving the environment is to run the script with a --search mode and a single seed common across 10 trials in order to identify the best hyper-parameters. 
```bash
python shared_ddpg.py --cpu --search --max-trials 10 --seeds 29
```

Then, we run the script with the best hyperparameters as default hyper parameters across 10 seeds:
```bash
python shared_ddpg.py --cpu --seeds 33,41,66,39,8,77,21,20,44,22
```
<a id = 'extract_weights_n_load_into_policy'></a>
# 5. Script: `extract_weights_n_load_into_policy.py`

<a id = 'extract_weights_n_load_into_policy_overview'></a>
## 5.1 Overview
This is a checkpoint utility used to:
- Finds the checkpoint folder for a given training run (run_dir), trial (trial_id), and seed (seed).
- Selects a specific policy checkpoint file (online_policy_model.<episode>.tar) either:
    - the exact --episode requested, or
    - the latest one (highest episode number) if --episode is omitted.
- Loads the policy network weights (state_dict) from that checkpoint.
- Exports the weights to:
    - a PyTorch *.weights.pt file (a state_dict), and/or
    - a NumPy *.weights.npz file (arrays per parameter).

Optionally validates the weights by reconstructing the FCDP policy network with inferred dimensions and running a dummy forward pass.

<a id = 'extract_weights_n_load_into_policy_arguments'></a>
## 5.2 CLI arguments

Required CLI argument:
- --run-dir : path to a specific run folder, 
e.g.:
...\p3_collab-compet\results\collab_compet\run_YYYYMMDD_HHMMSS.

Optional CLI arguments:
- --trial-id (default: "default") : selects subfolder: trials/<trial_id>/... (e.g. trial_003).
- --seed (default: 12) : selects subfolder: seed_<seed>.
- --episode (default: None). If provided, loads exactly:
    online_policy_model.<episode>.tar. If omitted, loads the latest checkpoint (by episode number).
- --out-dir (default: None). Where to write exported files. If omitted, writes into the checkpoint directory itself.
- --no-pt : Don’t write the .weights.pt export.
- --no-npz : Don’t write the .weights.npz export.
- --no-validate : Skip rebuilding FCDP and the dummy forward-pass check.

<a id = 'extract_weights_n_load_into_policy_run'></a>
## 5.3 runs

In this experiment, the script was run twice to extract the weights of the policy models generated by seeds 41 and 22. 
```bash
python extract_weights_n_load_into_policy.py --run-dir "path/to/results/collab_compet/run_20251228_195414_TRAINING_PER_SEED" --seed 41

python extract_weights_n_load_into_policy.py --run-dir "path/to/results/collab_compet/run_20251228_195414_TRAINING_PER_SEED" --seed 22
```

<a id = 'repo_structure'></a>
# 6. Repository structure (key items)

- `shared_actor_maddpg_with_team_reward.py`
- `extract_weights_n_load_into_policy.py`
- `report.pdf`
- `./20_Agents_Reacher_Windows_x86_64/` (includes `Reacher.exe`)
- `./results/collab_compet/run_.../` (logs, plots, summaries, trials)
- `./best_model_and_weights_and_plot/` (best model, weights, plot)
- `./pictures/`



