import warnings ; warnings.filterwarnings('ignore')
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['OMP_NUM_THREADS'] = '1'

import argparse
import json
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
from itertools import count, product
import random
import time
import gc
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment
import multiprocessing as mp

# Logging is adapted to print and write one line per episode (no timed gating), matching navigation_v3.py.
LEAVE_PRINT_EVERY_N_SECS = 60  # unused; kept for backward compatibility
ERASE_LINE = '\x1b[2K'         # unused; kept for backward compatibility
EPS = 1e-6
# Gradient clipping is opt-in via CLI; None disables clipping.
MAX_GRAD_NORM: Optional[float] = None
RESULTS_DIR = os.path.join('results')

np.set_printoptions(suppress=True)


def _resolve_env_exe(arg_path: Optional[str]) -> str:
    """
    Resolve the path to the Unity Reacher executable (20-agent version).

    Strategy:
    - If --env-exe provided and exists, use it.
    - Else, check UNITY_REACHER_EXE env var.
    - Else, search common local paths adjacent to this script.
    - Else, walk up to depth 3 and locate Reacher.exe.

    Raises:
        FileNotFoundError if no executable is found.

    Returns:
        str: Absolute path to Reacher.exe.
    """
    if arg_path and os.path.isfile(arg_path):
        return os.path.abspath(arg_path)
    env_path = os.getenv('UNITY_REACHER_EXE')
    if env_path and os.path.isfile(env_path):
        return os.path.abspath(env_path)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, '20_Agents_Reacher_Windows_x86_64', 'Reacher.exe'),
        os.path.join(here, 'Reacher_Windows_x86_64', 'Reacher.exe'),
        os.path.join(os.path.dirname(here), 'p2_continuous-control', '20_Agents_Reacher_Windows_x86_64', 'Reacher.exe'),
        os.path.join(os.path.dirname(here), '20_Agents_Reacher_Windows_x86_64', 'Reacher.exe'),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    base = os.path.dirname(here)
    for root, dirs, files in os.walk(base):
        depth = root.replace(base, "").count(os.sep)
        if depth > 3:
            dirs[:] = []
            continue
        if 'Reacher.exe' in files:
            return os.path.abspath(os.path.join(root, 'Reacher.exe'))
    raise FileNotFoundError(
        "Unity Reacher executable not found. Pass --env-exe, set UNITY_REACHER_EXE, "
        "or place Reacher.exe under 20_Agents_Reacher_Windows_x86_64 adjacent to this script."
    )


def _setup_logging(run_dir: str) -> str:
    """
    Initialize root logger to write both to console and to run.log.

    Prevent duplicate log lines by clearing existing handlers first.

    Behavior:
    - Log one line per episode to console and run.log (no timed gating), matching navigation_v3.py.
    - Uses logging.FileHandler (Windows-friendly) instead of WatchedFileHandler.
    """
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, 'run.log')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Clear existing handlers (avoid duplicates)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)

    # Avoid propagating to ancestor loggers that might add handlers
    logger.propagate = False

    logging.info(f'Logging initialized. Log file: {log_path}')
    return log_path


def _format_hparams(hp: Dict[str, Any]) -> str:
    """
    Compact JSON string for hyperparameters (sorted keys).
    """
    return json.dumps(hp, separators=(',', ':'), sort_keys=True)


def _default_env_settings() -> Dict[str, Any]:
    """
    Default environment/training stop conditions for Reacher (20 agents).

    Returns:
        dict with keys:
            gamma (float): Discount factor.
            max_minutes (int): Wall-clock budget (minutes).
            max_episodes (int): Episode cap.
            goal_mean_100_reward (float): Early stop threshold on eval mean_100.
            final_eval_episodes (int): Episodes for final evaluation at the end.
    """
    return {
        'gamma': 0.99,
        'max_minutes': 180,
        'max_episodes': 200,
        'goal_mean_100_reward': 30,
        'final_eval_episodes': 100,
    }


def _build_search_space() -> List[Dict[str, Any]]:
    """
    Build a grid of hyperparameter configurations for DDPG search.
    """
    grid = {
        'hidden_dims': [(64, 64), (128,128), (256,256)],
        'optimizer': ['Adam'],
        'lr': [1e-4, 3e-4, 5e-4],
        'buffer_size': [100000],
        'batch_size': [64, 128, 256, 500],
        'exploration_noise_ratio': [0.05, 0.1, 0.2],
        'n_warmup_batches': [5, 10],
        'target_update_every': [2, 5, 10],
        'tau': [0.001, 0.005, 0.01],
    }
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    cfgs = [{k: v for k, v in zip(keys, combo)} for combo in product(*vals)]
    return cfgs


def _optimizer_from_name(torch_optim, name: str):
    """
    Map optimizer name to torch optimizer class.

    Supported:
    - 'adam' -> torch.optim.Adam
    - 'rmsprop' -> torch.optim.RMSprop
    """
    name = name.lower()
    if name == 'adam': return torch_optim.Adam
    if name == 'rmsprop': return torch_optim.RMSprop
    raise ValueError(f'Unsupported optimizer: {name}')


# ---- Hypothesis tests (bootstrap CI for final scores) ----
def _bootstrap_ci(data: List[float], n_boot: int = 5000, ci: float = 0.95, rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for the mean.

    Args:
        data: List of final evaluation scores.
        n_boot: Number of bootstrap resamples.
        ci: Confidence level for the interval.
        rng: Optional numpy Generator for reproducibility.

    Returns:
        (lo, hi) confidence interval bounds for the mean.
    """
    if rng is None:
        rng = np.random.default_rng()
    data = np.asarray(data, dtype=float)
    if len(data) == 0:
        return float('nan'), float('nan')
    n = len(data)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        samp = rng.choice(data, size=n, replace=True)
        boots[i] = float(np.mean(samp))
    lo = float(np.quantile(boots, (1-ci)/2))
    hi = float(np.quantile(boots, 1 - (1-ci)/2))
    return lo, hi


def _log_hypothesis_tests(final_scores: List[float], goal_mean_100_reward: float, header_prefix: str = "") -> None:
    """
    Log bootstrap confidence interval summaries against the goal threshold.

    Prints:
      - Scores, mean, std.
      - 95% bootstrap CI for the mean.
      - Whether CI lower bound exceeds the goal (mean > goal at 95%).
    """
    scores = np.asarray(final_scores, dtype=float)
    mean = float(np.mean(scores)) if len(scores) else float('nan')
    std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    n = len(scores)
    ci_lo, ci_hi = _bootstrap_ci(final_scores, n_boot=5000, ci=0.95) if n >= 1 else (float('nan'), float('nan'))

    logging.info(f'{header_prefix}Hypothesis tests over {n} seeds:')
    logging.info(f'{header_prefix}Final scores: {np.round(scores, 3).tolist()}')
    logging.info(f'{header_prefix}Mean={mean:.3f}, Std={std:.3f}, Goal={goal_mean_100_reward:.3f}')
    logging.info(f'{header_prefix}Bootstrap 95% CI for mean: [{ci_lo:.3f}, {ci_hi:.3f}]')
    significant = (not any(np.isnan([ci_lo, ci_hi]))) and (ci_lo > goal_mean_100_reward)
    logging.info(f'{header_prefix}Significant (mean > goal @95%)? {"Yes" if significant else "No"}')


# ---- Unity vectorized env (20 agents) in a separate process to avoid hangs ----
def _unity_worker(worker_end, env_exe_path: str, seed: int, worker_id: int):
    """
    Worker process hosting the UnityEnvironment to isolate the native runtime.

    Commands:
    - 'reset' with kwargs {'train_mode': bool}
    - 'step' with kwargs {'actions': np.ndarray}
    - 'close'

    Shapes:
    - reset(train_mode): returns observations of shape (20, 33)
    - step(actions): expects actions of shape (20, 4)
      returns obs (20,33), rewards (20,1) float32, dones (20,1) bool

    Note:
    - Worker logs go to console from the child process; they are not written to run.log in the parent.
    """
    try:
        logging.info(f'[Worker {worker_id}] Starting worker (seed={seed}). Opening Unity env: {env_exe_path}')
    except Exception:
        pass

    # Let exceptions during initialization terminate the worker so the parent can detect and respawn.
    env = UnityEnvironment(file_name=env_exe_path, seed=seed, no_graphics=True, worker_id=worker_id)
    brain_name = env.brain_names[0]

    try:
        logging.info(f'[Worker {worker_id}] Unity env opened. brain_name="{brain_name}"')
    except Exception:
        pass

    def _reset(train_mode: bool = True):
        info = env.reset(train_mode=train_mode)[brain_name]
        return info.vector_observations  # (20, 33)

    def _step(actions: np.ndarray):
        info = env.step(actions)[brain_name]
        obs = info.vector_observations          # (20, 33)
        rewards = np.asarray(info.rewards, dtype=np.float32)  # (20,)
        dones = np.asarray(info.local_done, dtype=np.bool_)   # (20,)
        return obs, rewards[:, None], dones[:, None]          # (20,33), (20,1), (20,1)

    try:
        while True:
            cmd, kwargs = worker_end.recv()
            if cmd == 'reset':
                worker_end.send(_reset(train_mode=kwargs.get('train_mode', True)))
            elif cmd == 'step':
                worker_end.send(_step(kwargs.get('actions')))
            elif cmd == 'close':
                try:
                    logging.info(f'[Worker {worker_id}] Closing Unity env.')
                except Exception:
                    pass
                env.close(); worker_end.close(); break
            else:
                try:
                    logging.info(f'[Worker {worker_id}] Unknown command "{cmd}". Closing Unity env.')
                except Exception:
                    pass
                env.close(); worker_end.close(); break
    finally:
        try:
            env.close()
            logging.info(f'[Worker {worker_id}] Unity env closed. Worker exiting.')
        except Exception:
            try:
                logging.info(f'[Worker {worker_id}] Worker exiting (env close raised).')
            except Exception:
                pass


class UnityVectorEnv:
    """
    Lightweight vectorized environment wrapper that runs a UnityEnvironment in a separate
    Python process and communicates via a multiprocessing Pipe.

    Why a separate process
    - The Unity ML-Agents binary (Reacher.exe) can hang or crash the main process. Hosting it
      in a worker process isolates failures and keeps the training process responsive.
    - Avoids GPU/graphics context contention and allows clean shutdowns.

    Design
    - A worker process executes _unity_worker(), which:
      * Opens UnityEnvironment(file_name=env_exe_path, seed=seed, worker_id=base_worker_id, no_graphics=True).
      * Handles 3 commands received over the Pipe:
          'reset' -> returns observations (20, 33)
          'step'  -> expects actions (20, 4), returns (obs, rewards, dones)
          'close' -> closes the Unity env and exits
    - This class is the main-process handle:
      * __init__ spawns the worker and stores the parent Pipe end.
      * reset()/step() send a command and wait for the reply.
      * close() requests shutdown and joins the worker.

    Fault-tolerance (new):
    - If the worker fails to bind its gRPC port (e.g., "Only one usage of each socket address ..."),
      the child will die and the pipe raises EOFError/BrokenPipeError. On such errors, this wrapper
      will respawn a new worker with a different worker_id and retry the operation automatically.
    - Also handles cases where a previous Unity process is still bound to the port range.
    """

    def __init__(self, env_exe_path: str, seed: int, base_worker_id: int = 0):
        self.env_exe_path = env_exe_path
        self.seed = seed
        self.base_worker_id = base_worker_id

        # Retry/respawn settings
        self.max_spawn_retries = 6
        self._current_worker_id = base_worker_id

        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
        self._ctx = mp.get_context('spawn')
        self.pipe = None
        self.worker = None

        logging.info(f'[Main] Spawning worker {self.base_worker_id} for Unity env (seed={self.seed}). Executable: {self.env_exe_path}')
        self._start_worker(self._current_worker_id)

    def _start_worker(self, worker_id: int):
        # Clean previous if any
        self._cleanup_worker(silent=True)

        parent_end, child_end = self._ctx.Pipe()
        self.pipe = parent_end
        self.worker = self._ctx.Process(
            target=_unity_worker,
            args=(child_end, self.env_exe_path, self.seed, worker_id),
            daemon=True
        )
        self.worker.start()
        logging.info(f'[Main] Worker {worker_id} started (pid={self.worker.pid}).')
        # Give the child a moment to initialize; parent will still robustly handle failures.
        time.sleep(0.25)

    def _cleanup_worker(self, silent: bool = False):
        if self.pipe is not None:
            try:
                self.pipe.close()
            except Exception:
                pass
            self.pipe = None
        if self.worker is not None:
            try:
                if self.worker.is_alive():
                    if not silent:
                        logging.info(f'[Main] Terminating worker {self._current_worker_id} (pid={self.worker.pid})')
                    self.worker.terminate()
                    self.worker.join(timeout=1.5)
            except Exception:
                pass
            self.worker = None

    def _choose_alternate_worker_id(self, attempt: int) -> int:
        """
        Choose an alternate worker id to avoid port conflicts.
        Ports are assigned as 5004 + worker_id in ML-Agents; keep result under 65535.
        """
        rnd = random.randint(0, 999)
        candidate = int((self.base_worker_id or 0) + 1000 * (attempt + 1) + rnd)
        candidate = max(1, min(candidate, 59000))  # 5004 + 59000 = 64004 < 65535
        return candidate

    def _respawn_with_retry(self, reason: str):
        last_err = None
        for attempt in range(self.max_spawn_retries):
            new_id = self._choose_alternate_worker_id(attempt)
            logging.warning(f'[Main] Worker {self._current_worker_id} failed ({reason}). Respawning with worker_id={new_id} (attempt {attempt+1}/{self.max_spawn_retries})')
            self._current_worker_id = new_id
            try:
                self._start_worker(new_id)
                # Let caller retry the operation (reset/step) after return.
                return
            except Exception as e:
                last_err = e
                self._cleanup_worker(silent=True)
                time.sleep(0.3)
        raise RuntimeError(f'Unity environment failed to spawn after {self.max_spawn_retries} retries.') from last_err

    def reset(self, **kwargs) -> np.ndarray:
        if self.pipe is None or self.worker is None or not self.worker.is_alive():
            self._respawn_with_retry('dead worker before reset')
        try:
            self.pipe.send(('reset', kwargs))
            return self.pipe.recv()  # (20, 33)
        except (EOFError, BrokenPipeError, OSError) as e:
            # Worker likely failed to bind port or crashed; try alternate worker_id.
            self._respawn_with_retry(type(e).__name__)
            # Retry once after respawn
            self.pipe.send(('reset', kwargs))
            return self.pipe.recv()

    def step(self, actions: np.ndarray):
        if self.pipe is None or self.worker is None or not self.worker.is_alive():
            self._respawn_with_retry('dead worker before step')
        try:
            self.pipe.send(('step', {'actions': actions}))  # actions (20,4)
            obs, rewards, dones = self.pipe.recv()          # (20,33), (20,1), (20,1)
            infos = [{}] * obs.shape[0]
            return obs, rewards, dones, infos
        except (EOFError, BrokenPipeError, OSError) as e:
            # Attempt to recover mid-episode: respawn and reset caller’s episode.
            logging.warning(f'[Main] Worker {self._current_worker_id} failed during step ({type(e).__name__}). Respawning and forcing reset.')
            self._respawn_with_retry(type(e).__name__)
            self.pipe.send(('reset', {'train_mode': True}))
            obs = self.pipe.recv()
            rewards = np.zeros((obs.shape[0], 1), dtype=np.float32)
            dones = np.zeros((obs.shape[0], 1), dtype=np.bool_)
            infos = [{}] * obs.shape[0]
            return obs, rewards, dones, infos

    def close(self) -> None:
        logging.info(f'[Main] Requesting worker {self._current_worker_id} to close Unity env.')
        try:
            if self.pipe is not None:
                self.pipe.send(('close', {}))
        except Exception:
            pass
        try:
            if self.worker is not None:
                self.worker.join(timeout=2.0)
                logging.info(f'[Main] Worker {self._current_worker_id} joined. Unity env fully closed.')
        except Exception:
            logging.info(f'[Main] Worker {self._current_worker_id} join timed out or failed.')
        finally:
            self._cleanup_worker(silent=True)


# ---- Models and RL components (module level) ----
import torch
import torch.nn as nn
import torch.nn.functional as F

GLOBAL_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _set_global_device(force_cpu: bool = False) -> torch.device:
    global GLOBAL_DEVICE
    GLOBAL_DEVICE = torch.device("cpu") if force_cpu else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return GLOBAL_DEVICE

class FCQV(nn.Module):
    """
    Critic network for DDPG: approximates the state–action value Q(s, a) -> ℝ.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int, int] = (256,256), activation_fc=F.relu, device: Optional[torch.device] = None):
        super().__init__()
        self.activation_fc = activation_fc
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            in_dim = hidden_dims[i]
            if i == 0:
                in_dim += output_dim  # concat action at first hidden
            self.hidden_layers.append(nn.Linear(in_dim, hidden_dims[i+1]))
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.device = GLOBAL_DEVICE if device is None else device
        self.to(self.device)

    def _format(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32).unsqueeze(0)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, device=self.device, dtype=torch.float32).unsqueeze(0)
        return x, u

    def forward(self, state, action):
        x, u = self._format(state, action)
        x = self.activation_fc(self.input_layer(x))
        for i, hl in enumerate(self.hidden_layers):
            if i == 0:
                x = torch.cat((x, u), dim=1)
            x = self.activation_fc(hl(x))
        return self.output_layer(x)

    def load(self, experiences):
        states, actions, rewards, new_states, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, rewards, new_states, is_terminals


class FCDP(nn.Module):
    """
    Deterministic actor (policy) network for DDPG: maps states to continuous actions a = π(s).
    """
    def __init__(self, input_dim: int, action_bounds: Tuple[np.ndarray, np.ndarray], 
                 hidden_dims: Tuple[int,int] = (256,256), 
                 activation_fc=F.relu, out_activation_fc=F.tanh,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.activation_fc = activation_fc
        self.out_activation_fc = out_activation_fc
        env_min, env_max = action_bounds
        self.env_min_np = np.array(env_min, dtype=np.float32)
        self.env_max_np = np.array(env_max, dtype=np.float32)
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims)-1)])
        self.output_layer = nn.Linear(hidden_dims[-1], len(self.env_max_np))
        self.device = GLOBAL_DEVICE if device is None else device
        self.to(self.device)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32).unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hl in self.hidden_layers:
            x = self.activation_fc(hl(x))
        x = self.output_layer(x)
        x = self.out_activation_fc(x)  # in [-1,1]
        return x


class ReplayBuffer:
    """
    Simple experience replay buffer for off-policy training.
    """
    def __init__(self, max_size: int = 100000, batch_size: int = 500):
        self.ss_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.as_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.rs_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.ps_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.ds_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.max_size = max_size
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0
    def store(self, sample: Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]) -> None:
        s, a, r, p, d = sample
        self.ss_mem[self._idx] = s
        self.as_mem[self._idx] = a
        self.rs_mem[self._idx] = np.array([r], dtype=np.float32)
        self.ps_mem[self._idx] = p
        self.ds_mem[self._idx] = np.array([d], dtype=np.float32)
        self._idx = (self._idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    def sample(self, batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if batch_size is None:
            batch_size = self.batch_size
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return (np.vstack(self.ss_mem[idxs]),
                np.vstack(self.as_mem[idxs]),
                np.vstack(self.rs_mem[idxs]),
                np.vstack(self.ps_mem[idxs]),
                np.vstack(self.ds_mem[idxs]))
    def __len__(self) -> int:
        return self.size


class GreedyStrategy:
    """
    Evaluation strategy: use the deterministic policy output and clip it to the environment bounds.
    """
    def __init__(self, bounds: Tuple[np.ndarray, np.ndarray]):
        self.low = np.array(bounds[0], dtype=np.float32)
        self.high = np.array(bounds[1], dtype=np.float32)
        self.ratio_noise_injected = 0.0
    def select_action(self, model: FCDP, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            act = model(state).cpu().detach().data.numpy().squeeze()
        act = np.clip(act, self.low, self.high)
        return np.reshape(act, self.high.shape)


class NormalNoiseStrategy:
    """
    Training strategy: add Gaussian (normal) noise to deterministic policy actions.
    """
    def __init__(self, bounds: Tuple[np.ndarray, np.ndarray], exploration_noise_ratio: float = 0.1):
        self.low = np.array(bounds[0], dtype=np.float32)
        self.high = np.array(bounds[1], dtype=np.float32)
        self.exploration_noise_ratio = float(exploration_noise_ratio)
        self.ratio_noise_injected = 0.0
    def select_action(self, model: FCDP, state: np.ndarray, max_exploration: bool = False) -> np.ndarray:
        noise_scale = self.high if max_exploration else self.exploration_noise_ratio * self.high
        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()
        noise = np.random.normal(loc=0, scale=noise_scale, size=len(self.high))
        action = np.clip(greedy_action + noise, self.low, self.high)
        self.ratio_noise_injected = np.mean(abs((greedy_action - action)/(self.high - self.low + 1e-12)))
        return action


class DDPG:
    """
    Deep Deterministic Policy Gradient (DDPG) agent.

    Updated behavior:
    - Early stopping summary is recorded in self.early_stop_info:
      {'cause': 'goal'|'time'|'episodes', 'episode': int, 'mean_100_eval_at_stop': float or None}
      The mean_100_eval_at_stop is only populated when cause == 'goal'.
    - No aggregated plot across seeds; plotting is per-seed only.
    - Logging: one line per episode to console and to run.log (no timed gating), like navigation_v3.py.
    """
    def __init__(self,
                 replay_buffer_fn,
                 policy_model_fn,
                 policy_optimizer_fn,
                 policy_optimizer_lr,
                 value_model_fn,
                 value_optimizer_fn,
                 value_optimizer_lr,
                 training_strategy_fn,
                 evaluation_strategy_fn,
                 n_warmup_batches,
                 update_target_every_steps,
                 tau):
        self.replay_buffer_fn = replay_buffer_fn
        self.policy_model_fn = policy_model_fn
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.n_warmup_batches = n_warmup_batches
        self.update_target_every_steps = update_target_every_steps
        self.tau = tau
        self.trial_id = None
        self.hparams = None
        self.log_prefix = ""
        self.run_dir = None
        self._logged_train_agents = False
        self._logged_eval_agents = False
        self.early_stop_info: Dict[str, Any] = {}

    def _update_targets(self, tau: Optional[float] = None) -> None:
        tau = self.tau if tau is None else tau
        for target, online in zip(self.target_value_model.parameters(), self.online_value_model.parameters()):
            target.data.copy_((1.0 - tau) * target.data + tau * online.data)
        for target, online in zip(self.target_policy_model.parameters(), self.online_policy_model.parameters()):
            target.data.copy_((1.0 - tau) * target.data + tau * online.data)

    def _optimize(self, experiences) -> None:
        states, actions, rewards, next_states, is_terminals = experiences
        argmax_a_q_sp = self.target_policy_model(next_states)
        max_a_q_sp = self.target_value_model(next_states, argmax_a_q_sp)
        target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)

        q_sa = self.online_value_model(states, actions)
        td_error = q_sa - target_q_sa.detach()
        value_loss = td_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        if MAX_GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(self.online_value_model.parameters(), float(MAX_GRAD_NORM))
        self.value_optimizer.step()

        argmax_a_q_s = self.online_policy_model(states)
        max_a_q_s = self.online_value_model(states, argmax_a_q_s)
        policy_loss = -max_a_q_s.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        if MAX_GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(self.online_policy_model.parameters(), float(MAX_GRAD_NORM))
        self.policy_optimizer.step()

    def save_checkpoint(self, episode_idx: int, model) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(self.checkpoint_dir, f'online_policy_model.{episode_idx}.tar'))

    def train(self,
              env_exe_path: str,
              seed: int,
              gamma: float,
              max_minutes: int,
              max_episodes: int,
              goal_mean_100_reward: float,
              final_eval_episodes: int,
              base_worker_id: int):
        """
        Train the DDPG agent on Unity Reacher (20 agents).

        Returns
        - result: np.ndarray (max_episodes, 4) with columns:
            [0] mean_100_reward_from_training
            [1] mean_100_reward_from_eval
            [2] cumulative_training_time_seconds
            [3] cumulative_wallclock_time_seconds
        - final_eval_score: Mean reward over final_eval_episodes (greedy policy).
        - training_time: Total time spent stepping the env and training (sum of episode times).
        - wallclock_time: Total time from train() start to finish.

        Also sets:
        - self.early_stop_info dict with keys 'cause', 'episode', 'mean_100_eval_at_stop'
          (the last one only when cause == 'goal').

        Logging:
        - One line per episode is printed to console and written to run.log (no timed gating),
          matching the behavior of navigation_v3.py.
        """
        training_start = time.time()
        self.checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        prefix = f"[{self.trial_id} | seed {seed}] " if self.trial_id is not None else f"[seed {seed}] "
        self.log_prefix = prefix
        logging.info(f"{prefix}=== Starting run for seed {seed} ===")
        if self.hparams is not None:
            logging.info(f"{prefix}Hyperparameters: {_format_hparams(self.hparams)}")

        torch.manual_seed(seed) ; np.random.seed(seed) ; random.seed(seed)

        logging.info(f'{self.log_prefix}Launching UnityVectorEnv with worker_id={base_worker_id}, seed={seed}')
        envs = UnityVectorEnv(env_exe_path, seed, base_worker_id=base_worker_id)

        nS, nA = 33, 4
        action_bounds = np.array([-1,-1,-1,-1]), np.array([1,1,1,1])

        self.gamma = gamma
        self.final_eval_episodes = final_eval_episodes

        self.episode_timestep, self.episode_reward = [], []
        self.episode_seconds, self.evaluation_scores = [], []
        self.episode_exploration = []
        self.solved_episode_eval_100: Optional[int] = None
        self.early_stop_info = {}

        self.target_value_model = self.value_model_fn(nS, nA)
        self.online_value_model = self.value_model_fn(nS, nA)
        self.target_policy_model = self.policy_model_fn(nS, action_bounds)
        self.online_policy_model = self.policy_model_fn(nS, action_bounds)

        self._update_targets(tau=1.0)

        self.value_optimizer = self.value_optimizer_fn(self.online_value_model, self.value_optimizer_lr)
        self.policy_optimizer = self.policy_optimizer_fn(self.online_policy_model, self.policy_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn(action_bounds)
        self.evaluation_strategy = self.evaluation_strategy_fn(action_bounds)

        result = np.empty((max_episodes, 4), dtype=np.float32); result[:] = np.nan
        training_time = 0.0
        total_steps_accum = 0

        states = envs.reset()  # (20,33)
        if not self._logged_train_agents:
            try:
                n_agents = int(states.shape[0])
                logging.info(f"{self.log_prefix}Environment batched agents (train): {n_agents}")
            except Exception:
                logging.info(f"{self.log_prefix}Environment batched agents (train): unknown")
            self._logged_train_agents = True

        # ---- Episode loop ----
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()
            ep_reward_sum = 0.0
            ep_env_steps = 0
            ep_steps_sum = 0
            ep_exploration_sum = 0.0
            ep_sum_max_reward_per_step = 0.0
            ep_sum_max_abs_reward_per_step = 0.0
            agent_states = states  # current states for all 20 agents

            for _ in count():
                actions = np.vstack([self.training_strategy.select_action(self.online_policy_model, s) for s in agent_states])  # (20,4)
                obs, rewards, dones, _ = envs.step(actions)
                next_states = obs

                ep_env_steps += 1
                ep_sum_max_reward_per_step += float(np.max(rewards))
                ep_sum_max_abs_reward_per_step += float(np.max(np.abs(rewards)))

                for a_idx, s in enumerate(agent_states):
                    r = float(rewards[a_idx][0])
                    d = float(dones[a_idx][0])
                    ns = next_states[a_idx]
                    a = actions[a_idx]
                    self.replay_buffer.store((s, a, r, ns, d))
                    ep_reward_sum += r
                    ep_steps_sum += 1
                    ep_exploration_sum += self.training_strategy.ratio_noise_injected
                    total_steps_accum += 1

                agent_states = next_states

                min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
                if len(self.replay_buffer) > min_samples:
                    experiences = self.replay_buffer.sample()
                    experiences = self.online_value_model.load(experiences)
                    self._optimize(experiences)

                if total_steps_accum % self.update_target_every_steps == 0:
                    self._update_targets()

                if np.any(dones):
                    gc.collect()
                    break

            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed

            eval_score, _ = self.evaluate(self.online_policy_model, envs, n_episodes=1)
            self.save_checkpoint(episode-1, self.online_policy_model)

            self.episode_reward.append(ep_reward_sum / 20.0)
            self.episode_timestep.append(ep_steps_sum)
            self.episode_exploration.append(ep_exploration_sum / max(1, ep_steps_sum))
            self.evaluation_scores.append(eval_score)

            mean_100_reward = np.mean(self.episode_reward[-100:]) if len(self.episode_reward) >= 1 else np.nan
            std_100_reward = np.std(self.episode_reward[-100:]) if len(self.episode_reward) >= 1 else np.nan
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:]) if len(self.evaluation_scores) >= 1 else np.nan
            std_100_eval_score = np.std(self.evaluation_scores[-100:]) if len(self.evaluation_scores) >= 1 else np.nan

            if len(self.episode_exploration) >= 1:
                lst_100_exp_rat = np.array(self.episode_exploration[-100:])
                mean_100_exp_rat = np.mean(lst_100_exp_rat)
                std_100_exp_rat = np.std(lst_100_exp_rat)
            else:
                mean_100_exp_rat = np.nan
                std_100_exp_rat = np.nan

            if (
                self.solved_episode_eval_100 is None
                and not np.isnan(mean_100_eval_score)
                and mean_100_eval_score >= goal_mean_100_reward
                and len(self.evaluation_scores) >= 100
            ):
                self.solved_episode_eval_100 = episode
                logging.info(f'{self.log_prefix}Requirement met (evaluation metric): solved at episode {episode} with mean_100_eval={mean_100_eval_score:.2f}')
                try:
                    with open(os.path.join(self.run_dir, 'solved_evaluation_metric.json'), 'w', encoding='utf-8') as f:
                        json.dump({'episode': episode, 'mean_100_eval': float(mean_100_eval_score)}, f, indent=2)
                except Exception:
                    pass

            wallclock_elapsed = time.time() - training_start
            result[episode-1] = mean_100_reward, mean_100_eval_score, training_time, wallclock_elapsed

            # --------- Logging (every episode, console + file) ----------
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = (f'{self.log_prefix}elapsed_time {elapsed_str}, episode {episode-1:04}, '
                             f'episode_env_steps {ep_env_steps:04d}, '
                             f'episode_sum_max_reward_per_step {ep_sum_max_reward_per_step:09.6f}, '
                             f'episode_sum_max_abs_reward_per_step {ep_sum_max_abs_reward_per_step:09.6f}, '
                             f'mean_reward_over_100_episodes_from_training {mean_100_reward:05.2f}±{std_100_reward:05.2f}, '
                             f'mean_exploration_ratio_over_100_episodes_from_training {mean_100_exp_rat:02.4f}±{std_100_exp_rat:02.4f}, '
                             f'mean_reward_over_100_episodes_from_eval {mean_100_eval_score:05.2f}±{std_100_eval_score:05.2f}')
            print(debug_message, flush=True)
            logging.info(debug_message)

            # --------- Stopping conditions (no timed gating) ----------
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = (not np.isnan(mean_100_eval_score)) and (len(self.evaluation_scores) >= 100) and (mean_100_eval_score >= goal_mean_100_reward)
            training_is_over = reached_max_minutes or reached_max_episodes or reached_goal_mean_reward

            if training_is_over:
                cause = 'goal' if reached_goal_mean_reward else ('time' if reached_max_minutes else 'episodes')
                self.early_stop_info = {
                    'cause': cause,
                    'episode': int(episode),
                    'mean_100_eval_at_stop': float(mean_100_eval_score) if cause == 'goal' else None
                }
                if reached_max_minutes: logging.info(f'{self.log_prefix}--> reached_max_minutes x')
                if reached_max_episodes: logging.info(f'{self.log_prefix}--> reached_max_episodes x')
                if reached_goal_mean_reward: logging.info(f'{self.log_prefix}--> reached_goal_mean_reward (eval mean_100) OK')
                break

        final_eval_score, score_std = self.evaluate(self.online_policy_model, envs, n_episodes=self.final_eval_episodes)
        wallclock_time = time.time() - training_start
        logging.info(f'{self.log_prefix}Training complete.')

        if self.solved_episode_eval_100 is not None:
            logging.info(f'{self.log_prefix}Solved (evaluation metric) at episode {self.solved_episode_eval_100}.')
        else:
            logging.info(f'{self.log_prefix}Solved (evaluation metric) episode not reached within the run.')

        logging.info(f'{self.log_prefix}Final evaluation score {final_eval_score:.2f}±{score_std:.2f} in {training_time:.2f}s training, {wallclock_time:.2f}s wall.')
        logging.info(f'{self.log_prefix}Closing UnityVectorEnv (worker_id={base_worker_id}).')
        envs.close()
        return result, float(final_eval_score), float(training_time), float(wallclock_time)

    def evaluate(self, eval_policy_model: FCDP, envs: UnityVectorEnv, n_episodes: int = 1) -> Tuple[float, float]:
        """
        Evaluate the actor using greedy (noise-free) actions over n_episodes.
        """
        rs = []
        for _ in range(n_episodes):
            state = envs.reset(train_mode=True)  # (20,33)
            if not self._logged_eval_agents:
                try:
                    n_agents = int(state.shape[0])
                    logging.info(f"{self.log_prefix}Environment batched agents (eval): {n_agents}")
                except Exception:
                    logging.info(f"{self.log_prefix}Environment batched agents (eval): unknown")
                self._logged_eval_agents = True

            rs.append(0.0)
            for _ in count():
                actions = np.vstack([self.evaluation_strategy.select_action(eval_policy_model, s) for s in state])  # (20,4)
                obs, rewards, dones, _ = envs.step(actions)
                state = obs
                rs[-1] += float(np.mean(rewards))
                if np.any(dones):
                    break
        return np.mean(rs), np.std(rs)


def run_single_training(run_dir: str,
                        seeds: Tuple[int, ...],
                        env_settings: Dict[str, Any],
                        hparams: Dict[str, Any],
                        env_exe_path: str,
                        trial_id: Optional[str] = None,
                        trial_index: int = 1) -> Tuple[List[np.ndarray], List[float], float, List[Dict[str, Any]]]:
    """
    Train the DDPG agent across multiple seeds for a single hyperparameter configuration.

    Changes:
    - Remove aggregated plot across seeds; keep only per-seed plots.
    - Write plots_summary.csv incrementally (like navigation_v3.py), with one row per seed:
        trial_id,seed,cause,episode,mean_100_eval_at_stop,plot_path
    - Per-seed plot shows:
        * Eval (per-episode) rewards from the 1-episode eval run each training episode.
        * Eval (100-episode MA) for the same series.

    Logging:
    - Each episode line is printed and logged in the agent's train() method, so no episodes are skipped.
    """
    results_per_seed = []
    final_scores = []
    plots_summary_rows: List[Dict[str, Any]] = []
    best_eval_score = float('-inf')

    # Prepare CSV (create header once)
    plots_summary_csv = os.path.join(run_dir, 'plots_summary.csv')
    os.makedirs(run_dir, exist_ok=True)
    if not os.path.isfile(plots_summary_csv):
        with open(plots_summary_csv, 'w', encoding='utf-8') as f:
            f.write('trial_id,seed,cause,episode,mean_100_eval_at_stop,plot_path\n')

    policy_model_fn = lambda nS, bounds: FCDP(nS, bounds, hidden_dims=tuple(hparams['hidden_dims']))
    value_model_fn = lambda nS, nA: FCQV(nS, nA, hidden_dims=tuple(hparams['hidden_dims']))

    import torch.optim as optim
    policy_opt_cls = _optimizer_from_name(optim, hparams['optimizer'])
    value_opt_cls = _optimizer_from_name(optim, hparams['optimizer'])
    policy_optimizer_fn = lambda net, lr: policy_opt_cls(net.parameters(), lr=lr)
    value_optimizer_fn = lambda net, lr: value_opt_cls(net.parameters(), lr=lr)
    policy_optimizer_lr = float(hparams['lr'])
    value_optimizer_lr = float(hparams['lr'])

    training_strategy_fn = lambda bounds: NormalNoiseStrategy(bounds, exploration_noise_ratio=float(hparams['exploration_noise_ratio']))
    evaluation_strategy_fn = lambda bounds: GreedyStrategy(bounds)

    replay_buffer_fn = lambda: ReplayBuffer(max_size=int(hparams['buffer_size']), batch_size=int(hparams['batch_size']))

    n_warmup_batches = int(hparams['n_warmup_batches'])
    update_target_every_steps = int(hparams['target_update_every'])
    tau = float(hparams['tau'])

    gamma = env_settings['gamma']
    max_minutes = env_settings['max_minutes']
    max_episodes = env_settings['max_episodes']
    goal_mean_100_reward = env_settings['goal_mean_100_reward']
    final_eval_episodes = env_settings['final_eval_episodes']

    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    for seed in seeds:
        agent_run_dir = os.path.join(run_dir, 'trials', str(trial_id) if trial_id else 'default', f'seed_{seed}')
        os.makedirs(agent_run_dir, exist_ok=True)

        agent = DDPG(replay_buffer_fn,
                     policy_model_fn,
                     policy_optimizer_fn,
                     policy_optimizer_lr,
                     value_model_fn,
                     value_optimizer_fn,
                     value_optimizer_lr,
                     training_strategy_fn,
                     evaluation_strategy_fn,
                     n_warmup_batches,
                     update_target_every_steps,
                     tau)
        agent.run_dir = agent_run_dir
        agent.trial_id = trial_id if trial_id else 'default'
        agent.hparams = hparams

        # Worker id offset: large spacing to reduce chance of port collisions.
        base_worker_id = 1000*int(trial_index) + int(seed)

        logging.info(f"[{agent.trial_id} | seed {seed}] Start training with hparams: {_format_hparams(hparams)}")

        result, final_eval_score, training_time, wallclock_time = agent.train(
            env_exe_path, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward, final_eval_episodes, base_worker_id
        )

        results_per_seed.append(result)
        final_scores.append(final_eval_score)
        logging.info(f"[{agent.trial_id} | seed {seed}] Final eval score: {final_eval_score:.2f}")

        if final_eval_score > best_eval_score:
            best_eval_score = final_eval_score

        # --- Per-seed plot: per-episode eval rewards and 100-episode moving average; dotted vline at goal window start ---
        plot_path = ''
        try:
            # 100-episode moving average (from result[:,1])
            eval_mean_100_curve = result[:, 1].astype(np.float32)  # column 1 is eval mean_100
            mask = ~np.isnan(eval_mean_100_curve)

            # Raw per-episode evaluation rewards (each eval uses n_episodes=1)
            raw_eval = np.array(agent.evaluation_scores, dtype=np.float32)
            xs_raw = np.arange(raw_eval.shape[0])

            fig, ax = plt.subplots(1, 1, figsize=(15, 6), sharex=True)

            # Plot raw per-episode eval first
            if raw_eval.size > 0:
                ax.plot(xs_raw, raw_eval, color='steelblue', alpha=0.6, linewidth=1.25, label='Eval (per-episode)')

            # Overlay 100-episode moving average from result[:,1]
            xs = np.arange(len(eval_mean_100_curve))[mask]
            ys = eval_mean_100_curve[mask]
            if ys.size > 0:
                ax.plot(xs, ys, color='orange', linewidth=2, label='Eval (100-episode MA)')

            ax.set_title(f"Evaluation Performance (seed {seed}, trial='{agent.trial_id}')")
            ax.set_xlabel('Episodes'); ax.set_ylabel('Reward')
            ax.grid(True, linestyle=':', alpha=0.5)

            goal = goal_mean_100_reward
            all_idxs = np.arange(len(eval_mean_100_curve))
            hit_idxs = np.where(mask & (all_idxs >= 99) & (eval_mean_100_curve >= goal))[0]
            if hit_idxs.size > 0:
                hit_idx = int(hit_idxs[0])
                start_idx = max(0, hit_idx - 99)
                ax.axvline(x=start_idx, linestyle=':', color='gray', linewidth=1.5, label='Goal window start')

            if np.any(mask) or raw_eval.size > 0:
                last_idx_ma = int(np.where(mask)[0].max()) if np.any(mask) else -1
                last_idx_raw = int(xs_raw[-1]) if raw_eval.size > 0 else -1
                last_idx = max(last_idx_ma, last_idx_raw)
                ax.tick_params(axis='x', which='both', labelsize=10)
                ax.margins(x=0.02)
                step = max(1, last_idx // 10) if last_idx > 0 else 1
                ax.set_xticks(np.arange(0, last_idx + 1, step))
                ax.set_xlim(0, max(1, last_idx))
            else:
                logging.warning(f'[{agent.trial_id} | seed {seed}] No valid points to plot for this seed.')

            ax.legend(loc='upper left')
            fig.tight_layout(rect=[0, 0.03, 1, 0.98])

            plot_path = os.path.join(plots_dir, f"evaluation_mean100_{agent.trial_id}_seed_{seed}.png")
            fig.savefig(plot_path, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"[{agent.trial_id} | seed {seed}] Per-seed evaluation plot saved to {plot_path}")
        except Exception as e:
            logging.warning(f'[{agent.trial_id} | seed {seed}] Per-seed plotting failed: {e}')

        # --- Append CSV summary row immediately (navigation_v3.py behavior) ---
        cause = agent.early_stop_info.get('cause', '')
        episode = agent.early_stop_info.get('episode', '')
        mean_100_eval_at_stop = agent.early_stop_info.get('mean_100_eval_at_stop', None)
        row = {
            'trial_id': agent.trial_id,
            'seed': int(seed),
            'cause': cause,
            'episode': int(episode) if episode != '' else '',
            'mean_100_eval_at_stop': f"{float(mean_100_eval_at_stop):.6f}" if (cause == 'goal' and mean_100_eval_at_stop is not None) else '',
            'plot_path': plot_path or ''
        }
        plots_summary_rows.append(row)
        try:
            with open(plots_summary_csv, 'a', encoding='utf-8') as f:
                f.write(f"{row['trial_id']},{row['seed']},{row['cause']},{row['episode']},{row['mean_100_eval_at_stop']},{row['plot_path']}\n")
            logging.info(f"[{agent.trial_id} | seed {seed}] Summary appended to {plots_summary_csv}")
        except Exception as e:
            logging.warning(f"[{agent.trial_id} | seed {seed}] Failed writing plots_summary.csv: {e}")

    return results_per_seed, final_scores, best_eval_score, plots_summary_rows


def main():
    """
    Entry point to orchestrate training/evaluation of a DDPG agent on Unity Reacher (20 agents).

    Updated behavior
    - Writes one line per episode to console and run.log using the same logging setup as navigation_v3.py.
    - Writes plots_summary.csv incrementally per seed, with columns:
        trial_id, seed, cause, episode, mean_100_eval_at_stop (only when cause == 'goal'), plot_path.
    - Logs hypothesis tests (bootstrap CI for mean final score vs goal) after search best and after single run.
    - Logging is reset per run to avoid duplicates when running from IDEs or reusing the Python process.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--search', action='store_true', help='Enable hyperparameter search')
    parser.add_argument('--max-trials', type=int, default=8, help='Maximum number of trials to run')
    parser.add_argument('--seeds', type=str, default='12,34,56', help='Comma-separated list of seeds')
    parser.add_argument('--env-exe', type=str, default=None, help='Path to Unity Reacher executable (20 agents).')
    parser.add_argument('--cpu', action='store_true', help='Force CPU (disable CUDA usage in PyTorch).')
    args = parser.parse_args()

    _set_global_device(force_cpu=bool(args.cpu))
    logging.info(f"Torch device: {GLOBAL_DEVICE}")

    seeds = tuple(int(s) for s in args.seeds.split(',') if s.strip() != '')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(RESULTS_DIR, 'continuous_control', f'run_{timestamp}')
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    _setup_logging(run_dir)
    env_exe = _resolve_env_exe(args.env_exe)
    logging.info(f'Results will be saved to: {run_dir}')
    logging.info(f'Seeds: {seeds}')
    logging.info(f'Unity exe: {env_exe}')

    env_settings = _default_env_settings()

    if args.search:
        all_cfgs = _build_search_space()
        rng = np.random.default_rng(1234)
        cfgs = all_cfgs if args.max_trials >= len(all_cfgs) else [all_cfgs[i] for i in rng.choice(len(all_cfgs), size=args.max_trials, replace=False)]

        logging.info(f'Hyperparameter search enabled. Trials to run: {len(cfgs)} (max-trials={args.max_trials})')

        trials_csv = os.path.join(run_dir, 'hparam_search.csv')
        with open(trials_csv, 'w', encoding='utf-8') as f:
            f.write('trial_id,avg,std,best_seed_score,seeds,hidden_dims,optimizer,lr,buffer_size,batch_size,exploration_noise_ratio,'
                    'n_warmup_batches,target_update_every,tau\n')

        best = None
        best_results_arrays = None

        for i, cfg in enumerate(cfgs, start=1):
            trial_id = f"trial_{i:03d}"
            logging.info(f'[{trial_id}] Starting trial with hparams: {_format_hparams(cfg)}')

            results_per_seed, final_scores, best_seed_score, _ = run_single_training(
                run_dir, seeds, env_settings, cfg, env_exe, trial_id=trial_id, trial_index=i
            )

            avg = float(np.mean(final_scores))
            std = float(np.std(final_scores, ddof=1)) if len(final_scores) > 1 else 0.0
            logging.info(f'[{trial_id}] Final scores across seeds: {np.round(final_scores, 3).tolist()}')
            logging.info(f'[{trial_id}] Avg={avg:.3f} ± {std:.3f}, Best seed score={best_seed_score:.3f}')

            with open(trials_csv, 'a', encoding='utf-8') as f:
                f.write(f"{trial_id},{avg:.6f},{std:.6f},{best_seed_score:.6f},\"{list(seeds)}\","
                        f"\"{tuple(cfg['hidden_dims'])}\",{cfg['optimizer']},{cfg['lr']},{cfg['buffer_size']},{cfg['batch_size']},"
                        f"{cfg['exploration_noise_ratio']},{cfg['n_warmup_batches']},{cfg['target_update_every']},{cfg['tau']}\n")

            if best is None or avg > best['avg']:
                best = {'trial_id': trial_id, 'avg': avg, 'std': std, 'cfg': cfg, 'scores': final_scores}
                best_results_arrays = results_per_seed

        if best is not None:
            best_path = os.path.join(run_dir, 'best_hparams.json')
            with open(best_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'trial_id': best['trial_id'],
                    'avg_score': best['avg'],
                    'std_score': best['std'],
                    'seeds': list(seeds),
                    'hyperparameters': best['cfg'],
                    'scores_per_seed': best['scores'],
                }, f, indent=2, sort_keys=True)
            logging.info(f"Best trial: {best['trial_id']} with Avg={best['avg']:.3f} ± {best['std']:.3f}")
            logging.info(f"Best hyperparameters: {_format_hparams(best['cfg'])}")
            logging.info(f'Best hyperparameters saved to: {best_path}')

            # Plot evaluation moving average curve across episodes (averaged across seeds)
            try:
                ddpg_results = np.array(best_results_arrays, dtype=np.float32)  # (n_seeds, max_episodes, 4)
                eval_mean_100_curve = np.nanmean(ddpg_results[:, :, 1], axis=0)  # column 1 is eval mean_100
                mask = ~np.isnan(eval_mean_100_curve)
                fig, ax = plt.subplots(1, 1, figsize=(15,6), sharex=True)
                ax.plot(np.arange(len(eval_mean_100_curve))[mask], eval_mean_100_curve[mask], color='orange', linewidth=2, label='Eval (100-episode MA)')
                ax.set_title('Evaluation Performance (100-episode moving average)')
                ax.set_xlabel('Episodes'); ax.set_ylabel('Reward')
                ax.legend(loc='upper left'); ax.grid(True, linestyle=':', alpha=0.5)
                if np.any(mask):
                    last_idx = int(np.where(mask)[0].max())
                    ax.tick_params(axis='x', which='both', labelsize=10)
                    ax.margins(x=0.02)
                    step = max(1, last_idx // 10)
                    ax.set_xticks(np.arange(0, last_idx + 1, step))
                    ax.set_xlim(0, last_idx)
                else:
                    logging.warning('No valid points to plot for best trial.')
                fig.tight_layout(rect=[0, 0.03, 1, 0.98])
                plot_path = os.path.join(plots_dir, f'{best["trial_id"]}_evaluation_mean100.png')
                fig.savefig(plot_path, bbox_inches='tight')
                logging.info(f'Best trial evaluation plot saved to {plot_path}')
            except Exception as e:
                logging.warning(f'Plotting for best trial failed: {e}')

            # Hypothesis tests vs goal for the best trial
            _log_hypothesis_tests(best['scores'], env_settings['goal_mean_100_reward'], header_prefix=f"[{best['trial_id']}] ")

        logging.info('Hyperparameter search finished.')
        return

    # ---- Single-run baseline path ----
    logging.info('No hyperparameter search requested; running a single configuration.')

    default_hparams = {
        'hidden_dims': (256, 256),
        'optimizer': 'Adam',
        'lr': 5e-4,
        'buffer_size': 100000,
        'batch_size': 500,
        'exploration_noise_ratio': 0.05,
        'n_warmup_batches': 5,
        'target_update_every': 5,
        'tau': 0.01,
    }
    logging.info(f'Using hyperparameters: {_format_hparams(default_hparams)}')

    results_per_seed, final_scores, best_seed_score, _ = run_single_training(
        run_dir, seeds, env_settings, default_hparams, env_exe, trial_id='default', trial_index=0
    )
    logging.info(f'Default config final scores: {np.round(final_scores, 3).tolist()} | best seed score={best_seed_score:.3f}')

    # Hypothesis tests vs goal for the baseline run
    _log_hypothesis_tests(final_scores, env_settings['goal_mean_100_reward'])

    logging.info('Run finished.')


if __name__ == '__main__':
    os.environ['PYTHONUNBUFFERED'] = '1'  # Unbuffered mode so logs/prints flush immediately.
    main()