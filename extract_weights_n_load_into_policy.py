import os
import glob
import argparse
from typing import Optional, Dict, Tuple, Any

import numpy as np
import torch

# Policy class from your Tennis DDPG script
from shared_ddpg import FCDP


def find_checkpoint_dir(run_dir: str, trial_id: str = "default", seed: int = 12) -> str:
    """Return the checkpoints directory for a given run/trial/seed."""
    ckpt_dir = os.path.join(run_dir, "trials", trial_id, f"seed_{seed}", "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoints dir not found: {ckpt_dir}")
    return ckpt_dir


def pick_checkpoint(ckpt_dir: str, episode: Optional[int] = None) -> str:
    """
    Pick a checkpoint file:
      - If episode is provided, select that episode file.
      - Otherwise, pick the latest episode (highest number).
    """
    pattern = os.path.join(ckpt_dir, "online_policy_model.*.tar")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoints found in: {ckpt_dir}")

    if episode is not None:
        target = os.path.join(ckpt_dir, f"online_policy_model.{episode}.tar")
        if not os.path.isfile(target):
            raise FileNotFoundError(f"Checkpoint for episode {episode} not found: {target}")
        return target

    def ep_num(p: str) -> int:
        base = os.path.basename(p)
        # online_policy_model.<num>.tar
        parts = base.split(".")
        if len(parts) < 3:
            return -1
        return int(parts[1])

    return max(files, key=ep_num)


def load_policy_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """
    Load the weights (state_dict) from a policy checkpoint.

    Supports either:
      - raw state_dict (mapping param_name -> Tensor)
      - dict containing a nested 'state_dict'
    """
    obj: Any = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        # If it's already a param dict: values are tensors with .shape
        if obj and all(hasattr(v, "shape") for v in obj.values()):
            return obj  # type: ignore[return-value]
        # Common wrapper
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            sd = obj["state_dict"]
            if sd and all(hasattr(v, "shape") for v in sd.values()):
                return sd  # type: ignore[return-value]
    raise ValueError("Checkpoint does not appear to contain a valid policy state_dict.")


def infer_hidden_dims_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """
    Infer hidden_dims tuple from checkpoint tensor shapes.
    Assumes FCDP keys like:
      - input_layer.weight: [h1, nS]
      - hidden_layers.0.weight (optional): [h2, h1]
    """
    if "input_layer.weight" not in state_dict:
        raise KeyError("Missing key 'input_layer.weight' in state_dict.")

    h1 = int(state_dict["input_layer.weight"].shape[0])
    if "hidden_layers.0.weight" in state_dict:
        h2 = int(state_dict["hidden_layers.0.weight"].shape[0])
    else:
        h2 = h1
    return (h1, h2)


def infer_io_dims_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """
    Infer (state_size, action_size) from checkpoint tensor shapes.
      - input_layer.weight: [hidden, state_size]
      - output_layer.weight: [action_size, last_hidden]
    """
    if "input_layer.weight" not in state_dict:
        raise KeyError("Missing key 'input_layer.weight' in state_dict.")
    if "output_layer.weight" not in state_dict:
        raise KeyError("Missing key 'output_layer.weight' in state_dict.")

    state_size = int(state_dict["input_layer.weight"].shape[1])
    action_size = int(state_dict["output_layer.weight"].shape[0])
    return state_size, action_size


def save_weights_pt(state_dict: Dict[str, torch.Tensor], out_path: str) -> None:
    """Save the state_dict to a .pt file."""
    torch.save(state_dict, out_path)
    print(f"Saved PyTorch weights: {out_path}")


def save_weights_npz(state_dict: Dict[str, torch.Tensor], out_path: str) -> None:
    """Save the state_dict as a NumPy .npz."""
    np_weights = {k: v.detach().cpu().numpy() for k, v in state_dict.items()}
    np.savez(out_path, **np_weights)
    print(f"Saved NumPy weights: {out_path}")


def rebuild_policy_and_load(
    state_dict: Dict[str, torch.Tensor],
    hidden_dims: Optional[Tuple[int, int]] = None,
) -> FCDP:
    """
    Recreate the policy network with dims matching the checkpoint and load weights.
    Deduces:
      - nS from input_layer.weight second dimension
      - nA from output_layer.weight first dimension
    """
    if hidden_dims is None:
        hidden_dims = infer_hidden_dims_from_state_dict(state_dict)

    nS, nA = infer_io_dims_from_state_dict(state_dict)

    # Both Tennis and Reacher use symmetric [-1, 1] action bounds; just infer dimensionality.
    action_low = -np.ones(nA, dtype=np.float32)
    action_high = np.ones(nA, dtype=np.float32)
    action_bounds = (action_low, action_high)

    policy = FCDP(nS, action_bounds, hidden_dims=hidden_dims)
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    policy.to("cpu")
    return policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract policy weights and optionally validate by rebuilding FCDP.")
    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help=r"Path to run dir, e.g. ...\p3_collab-compet\results\continuous_control\run_YYYYMMDD_HHMMSS_... ",
    )
    p.add_argument("--trial-id", type=str, default="default", help="Trial id folder name (e.g. default, trial_003).")
    p.add_argument("--seed", type=int, default=12, help="Seed folder to load (seed_<seed>).")
    p.add_argument("--episode", type=int, default=None, help="Episode index to load. If omitted, loads latest.")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. If omitted, saves next to the checkpoint in checkpoints/.",
    )
    p.add_argument("--no-pt", action="store_true", help="Do not save .weights.pt")
    p.add_argument("--no-npz", action="store_true", help="Do not save .weights.npz")
    p.add_argument("--no-validate", action="store_true", help="Skip rebuilding policy and dummy forward.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_dir = find_checkpoint_dir(args.run_dir, trial_id=args.trial_id, seed=args.seed)
    ckpt_path = pick_checkpoint(ckpt_dir, episode=args.episode)
    print(f"Using checkpoint: {ckpt_path}")

    state_dict = load_policy_state_dict(ckpt_path)

    # Choose output directory
    out_dir = args.out_dir or ckpt_dir
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.basename(ckpt_path)          # online_policy_model.2566.tar
    stem = os.path.splitext(base)[0]            # online_policy_model.2566

    if not args.no_pt:
        out_pt = os.path.join(out_dir, f"{stem}.weights.pt")
        save_weights_pt(state_dict, out_pt)

    if not args.no_npz:
        out_npz = os.path.join(out_dir, f"{stem}.weights.npz")
        save_weights_npz(state_dict, out_npz)

    # Validate load by rebuilding policy with inferred dims
    if not args.no_validate:
        hidden_dims = infer_hidden_dims_from_state_dict(state_dict)
        nS, nA = infer_io_dims_from_state_dict(state_dict)
        print(f"Inferred hidden_dims: {hidden_dims}")
        print(f"Inferred (state_size, action_size): ({nS}, {nA})")

        policy = rebuild_policy_and_load(state_dict, hidden_dims=hidden_dims)

        dummy_state = torch.zeros(2, nS, device="cpu")
        with torch.no_grad():
            action = policy(dummy_state)
        print("Sample action shape:", tuple(action.shape))
        print("Sample action:", action)


if __name__ == "__main__":
    main()