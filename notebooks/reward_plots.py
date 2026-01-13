import os
import sys
import pathlib
import tqdm
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import argparse  # Added for CLI

# Add parent directory to path for imports
directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))

import notebook_utils as nbu
from viper_rl.videogpt.reward_models import LOAD_REWARD_MODEL_DICT


def save_plot_to_png(
    values,
    source_labels=None,
    npz_rewards=None,               # NEW
    filename="output.png",
    title="Rewards",
    xlabel="frame index",
    ylabel="score",
    figsize=(12, 4),
):
    """
    Plots model rewards, optional .npz rewards, and optional source labels (anomalies).
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    # --- Plot 1: Model Rewards (Left Axis) ---
    x_rewards = np.arange(len(values))
    line1 = ax1.plot(
        x_rewards, values, label="Model Reward",
        linewidth=1.5, color="tab:blue"
    )

    lines = list(line1)

    # --- Plot 1b: NPZ Rewards (Left Axis) ---
    if npz_rewards is not None:
        x_npz = np.arange(len(npz_rewards))
        line_npz = ax1.plot(
            x_npz, npz_rewards, label="NPZ rewards",
            linewidth=1.5, color="tab:green", alpha=0.9
        )
        lines += list(line_npz)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel, color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(axis="x", linestyle="--", alpha=0.5, color="gray")

    # --- Plot 2: Source Labels (Right Axis) - Only if provided ---
    if source_labels is not None:
        ax2 = ax1.twinx()

        if np.ndim(source_labels) == 0:
            plot_labels = np.full(len(values), source_labels)
            x_labels = x_rewards
        else:
            plot_labels = source_labels
            x_labels = np.arange(len(source_labels))

        line2 = ax2.plot(
            x_labels, plot_labels, label="Anomaly (Source Label)",
            linewidth=1.5, color="tab:red", linestyle="-", alpha=0.7
        )
        ax2.set_ylabel("Source Label (0=Normal, 1=Anomaly)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.set_ylim(-0.1, 1.1)

        lines += list(line2)

    # Create combined legend
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    ax1.set_title(title)
    ax1.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)


def _resolve_npz_inputs(npz_inputs):
    if not npz_inputs:
        return []
    out = []
    for p in npz_inputs:
        pp = pathlib.Path(p)
        if pp.is_dir():
            out.extend(sorted(str(x) for x in pp.glob("*.npz")))
        elif pp.is_file() and pp.suffix == ".npz":
            out.append(str(pp))
        else:
            raise FileNotFoundError(f"Invalid --npz_files entry (not .npz file/dir): {p}")
    return sorted(out)


def _map_overlay_to_full_files(overlay_files, sequence_dir):
    """
    For each overlay .npz file, select the full .npz from sequence_dir using the same filename.
    Returns list of (full_path, overlay_path).
    """
    sequence_dir = pathlib.Path(sequence_dir)
    pairs = []
    missing = []

    for overlay in overlay_files:
        overlay_p = pathlib.Path(overlay)
        full_p = sequence_dir / overlay_p.name  # match by filename [web:21]
        if full_p.exists():
            pairs.append((str(full_p), str(overlay_p)))
        else:
            missing.append(str(full_p))

    if missing:
        raise FileNotFoundError(
            "Could not find matching full .npz files in sequence_dir for these expected paths:\n"
            + "\n".join(missing)
        )
    return pairs


def plot_rewards_from_npz_file_pairs(file_pairs, reward_model, task_name, output_dir="notebooks/plots"):
    """
    file_pairs: list of (full_npz_path, overlay_npz_path)
      - full_npz_path contains frames etc. used for model reward computation
      - overlay_npz_path contains only key 'rewards' used as extra plotted line
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing {len(file_pairs)} files...")
    processed_count = 0

    for full_path, overlay_path in tqdm.tqdm(file_pairs):
        # --- load overlay rewards only ---
        with open(overlay_path, "rb") as f:
            overlay_data = np.load(f)
            npz_rewards = np.copy(overlay_data["reward"])  # overlay-only [web:10]

        # --- load full sequence for model eval / labels ---
        with open(full_path, "rb") as f:
            data = np.load(f)

            source_labels = np.copy(data["source_label"]) if "source_label" in data else None
            data_copy = {k: np.copy(v) for k, v in data.items()}

        if "is_first" not in data_copy:
            seq_len = data_copy["image"].shape[0]
            is_first = np.zeros(seq_len, dtype=bool)
            is_first[0] = True
            data_copy["is_first"] = is_first

        seq_len = data_copy["image"].shape[0]
        seq = [{k: v[i] for k, v in data_copy.items()} for i in range(seq_len)]
        result = reward_model(seq)
        rewards = nbu.extract_key_from_seqs([result], "density")
        rewards = jnp.array(rewards)
        f_min, f_max = jnp.min(rewards), jnp.max(rewards)

        rewards = ((rewards - f_min) / (f_max - f_min + 1e-8)).astype(jnp.float32)
        base_name = pathlib.Path(full_path).stem
        save_path = os.path.join(output_dir, f"{base_name}.png")

        # Shorten filename: keep last 25 chars if too long
        short_name = base_name if len(base_name) <= 25 else f"...{base_name[-25:]}"

        save_plot_to_png(
            rewards,
            source_labels=source_labels,
            npz_rewards=npz_rewards,
            filename=save_path,
            title=f"Task: {task_name} | File: {short_name}",
        )
        processed_count += 1

    print(f"Done. Processed {processed_count} files.")


def plot_rewards_from_npz_files(file_list, 
                                reward_model, 
                                task_name,
                                output_dir="notebooks/plots"):
    """
    Loads sequences, evaluates them, and saves plots.
    No error handling included.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(file_list)} files...")

    processed_count = 0

    for fpath in tqdm.tqdm(file_list):
        # --- Loading Logic ---
        with open(fpath, 'rb') as f:
            data = np.load(f)
            
            # --- Handle Source Label (Mixed Quality) ---
            source_labels = None
            if 'source_label' in data:
                source_labels = np.copy(data['source_label'])

            # Create a clean dictionary copy 
            data_copy = {k: np.copy(v) for k, v in data.items()}

        # --- Inject is_first if missing ---
        if 'is_first' not in data_copy:
            seq_len = data_copy['image'].shape[0]
            is_first = np.zeros(seq_len, dtype=bool)
            is_first[0] = True
            data_copy['is_first'] = is_first

        # --- Restructuring Logic ---
        seq_len = data_copy['image'].shape[0]
        seq = []
        
        for i in range(seq_len):
            # Construct frame dict without try/except
            seq.append({k: v[i] for k, v in data_copy.items()})
        
        # --- Evaluation ---
        result = reward_model(seq)
        
        # Extract computed rewards
        rewards = nbu.extract_key_from_seqs([result], 'density')
        
        # --- Saving Plot ---
        base_name = pathlib.Path(fpath).stem 
        save_path = os.path.join(output_dir, f"{base_name}.png")
        
        # Shorten filename: keep last 25 chars if too long
        short_name = base_name if len(base_name) <= 25 else f"...{base_name[-25:]}"

        save_plot_to_png(
            rewards, 
            source_labels=source_labels, 
            filename=save_path, 
            title=f"Task: {task_name} | File: {short_name}"
        )
        processed_count += 1

    print(f"Done. Processed {processed_count} files.")
    
if __name__ == "__main__":

    # Parsing CLI Arguments
    parser = argparse.ArgumentParser(
        description="Evaluate and plot reward model outputs for npz sequences."
    )

    parser.add_argument(
        "--rm_key",
        type=str,
        default="dmc_clen16_fskip4",
        help="Key for the reward model dictionary.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="dmc_cartpole_balance",
        help="DMC task name.",
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="random",
        choices=["mixed", "normal", "random"],
        help="Data quality level.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Path to input (full) .npz directory. If None, uses default path structure.",
    )

    # NEW: overlay-only npz files/dirs; each contains ONLY key 'rewards'
    parser.add_argument(
        "--npz_files",
        nargs="*",
        default=None,
        help="Overlay-only .npz files and/or directories. Each .npz contains only key 'rewards'. "
             "Filenames are used to select matching full .npz from --input_dir / default dir.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="notebooks/plots",
        help="Base directory to save output plots.",
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=5,
        help="Number of files to process. Set to -1 to process all.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES ID.",
    )

    args = parser.parse_args()

    # Environment setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    print(f"Loading reward model: {args.rm_key}")
    reward_model = LOAD_REWARD_MODEL_DICT[args.rm_key](
        task=args.task,
        minibatch_size=2,
        encoding_minibatch_size=32,
        compute_joint=True,
    )

    task_name = args.task.replace("dmc_", "")

    # Determine directory containing the FULL sequences (images, etc.)
    if args.input_dir:
        sequence_dir = args.input_dir
    else:
        sequence_dir = f"/work/MLShare/vadrl_v5/dmc/test/{args.quality}/{task_name}"

    if not os.path.exists(sequence_dir):
        print(f"Directory not found: {sequence_dir}")
        raise SystemExit(1)

    output_subdir = os.path.join(args.output_dir, f"{args.quality}_{task_name}")
    os.makedirs(output_subdir, exist_ok=True)

    # --- Case A: overlay rewards provided; match by filename into sequence_dir ---
    if args.npz_files:
        overlay_files = _resolve_npz_inputs(args.npz_files)

        # Optionally limit count (deterministic because we sort) [web:42]
        if args.num_files != -1:
            overlay_files = overlay_files[:args.num_files]

        # Map overlay basename -> full file in sequence_dir using Path.name [web:21]
        file_pairs = _map_overlay_to_full_files(overlay_files, sequence_dir)

        plot_rewards_from_npz_file_pairs(
            file_pairs,
            reward_model,
            task_name=task_name,
            output_dir=output_subdir,
        )

    # --- Case B: old behavior (no overlay); just process full files from sequence_dir ---
    else:
        all_files = [os.path.join(sequence_dir, f) for f in os.listdir(sequence_dir) if f.endswith(".npz")]
        all_files.sort()  # filesystem order isn't guaranteed, so sort explicitly [web:42]

        if args.num_files != -1:
            all_files = all_files[:args.num_files]

        # Keep your old function for this mode (no overlay line)
        plot_rewards_from_npz_files(
            all_files,
            reward_model,
            task_name=task_name,
            output_dir=output_subdir,
        )
