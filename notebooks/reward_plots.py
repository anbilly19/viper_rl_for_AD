import os
import sys
import pathlib
import tqdm
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
import argparse
import toml 

# Add parent directory to path for imports
directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))

import notebook_utils as nbu
from viper_rl.videogpt.reward_models import LOAD_REWARD_MODEL_DICT

def save_plot_to_png(
    values,
    source_labels=None,
    external_rewards=None,
    filename="output.png",
    title="Rewards",
    xlabel="frame index",
    ylabel="score",
    figsize=(12, 4),
):
    """
    Plots Viper rewards, multiple external model rewards (from TOML), and optional source labels.
    """
    fig, ax1 = plt.subplots(figsize=figsize)

    # --- Plot 1: Main Viper Model Rewards ---
    x_rewards = np.arange(len(values))
    line1 = ax1.plot(
        x_rewards, values, label="Viper (Current)",
        linewidth=2.0, color="tab:blue", zorder=10
    )
    lines = list(line1)

    # --- Plot 1b: External Models from TOML ---
    if external_rewards:
        # Distinct colors for external models
        colors = ["tab:green", "tab:orange", "tab:purple", "tab:brown", "tab:pink", "tab:cyan"]
        
        for i, (model_name, rewards) in enumerate(external_rewards.items()):
            if rewards is None:
                continue
            
            x_ext = np.arange(len(rewards))
            color = colors[i % len(colors)]
            
            line_ext = ax1.plot(
                x_ext, rewards, label=model_name,
                linewidth=1.5, color=color, alpha=0.8, linestyle="--"
            )
            lines += list(line_ext)

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel, color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(axis="x", linestyle="--", alpha=0.5, color="gray")

    # --- Plot 2: Source Labels ---
    if source_labels is not None:
        ax2 = ax1.twinx()
        if np.ndim(source_labels) == 0:
            plot_labels = np.full(len(values), source_labels)
            x_labels = x_rewards
        else:
            plot_labels = source_labels
            x_labels = np.arange(len(source_labels))

        line2 = ax2.plot(
            x_labels, plot_labels, label="Anomaly (GT)",
            linewidth=1.5, color="tab:red", linestyle="-", alpha=0.6
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


def process_and_plot_sequences(
    target_filenames, 
    sequence_dir,
    reward_model, 
    external_model_paths, 
    output_dir="notebooks/plots"
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing {len(target_filenames)} files...")
    
    processed_count = 0

    for fname in tqdm.tqdm(target_filenames):
        full_path = sequence_dir / fname
        stem_name = full_path.stem

        # --- 1. Load Main Sequence Data (Viper Input) ---
        try:
            with open(full_path, "rb") as f:
                data = np.load(f)
                source_labels = np.copy(data["source_label"]) if "source_label" in data else None
                data_copy = {k: np.copy(v) for k, v in data.items()}
        except Exception as e:
            print(f"Skipping {fname}: Failed to load full sequence. {e}")
            continue

        if "is_first" not in data_copy:
            seq_len = data_copy["image"].shape[0]
            is_first = np.zeros(seq_len, dtype=bool)
            is_first[0] = True
            data_copy["is_first"] = is_first

        # --- 2. Compute Viper Rewards ---
        seq_len = data_copy["image"].shape[0]
        seq = [{k: v[i] for k, v in data_copy.items()} for i in range(seq_len)]
        result = reward_model(seq)
        viper_rewards = nbu.extract_key_from_seqs([result], "density")
        viper_rewards = jnp.array(viper_rewards)
        
        # Normalize Viper rewards
        f_min, f_max = jnp.min(viper_rewards), jnp.max(viper_rewards)
        viper_rewards = ((viper_rewards - f_min) / (f_max - f_min + 1e-8)).astype(jnp.float32)

        # --- 3. Load External Rewards from TOML Paths ---
        external_rewards_data = {}
        
        for model_name, model_dir in external_model_paths.items():
            # Check if this specific model has a reward file for this sequence
            ext_file_path = pathlib.Path(model_dir) / fname
            
            if ext_file_path.exists():
                try:
                    with open(ext_file_path, "rb") as f_ext:
                        ext_data = np.load(f_ext)
                        if "reward" in ext_data:
                            external_rewards_data[model_name] = np.copy(ext_data["reward"])
                except Exception:
                    pass 
            # If file doesn't exist for this model, we just don't add it to dict (no line plotted)

        # --- 4. Plotting ---
        save_path = os.path.join(output_dir, f"{stem_name}.png")

        save_plot_to_png(
            viper_rewards,
            source_labels=source_labels,
            external_rewards=external_rewards_data,
            filename=save_path,
            title=f"Rewards Comparison: {stem_name}",
        )
        processed_count += 1

    print(f"Done. Processed {processed_count} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate and plot reward model outputs for npz sequences."
    )
    parser.add_argument("--rm_key", type=str, default="dmc_clen16_fskip4")
    parser.add_argument("--task", type=str, default="dmc_cartpole_balance")
    parser.add_argument("--quality", type=str, default="random", choices=["mixed", "normal", "random"])
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing FULL .npz sequences.")
    
    # TOML is now the primary driver for file selection
    parser.add_argument(
        "--toml_config", 
        type=str, 
        required=True, 
        help="Path to TOML file. Folders inside determine which files are processed."
    )
    
    parser.add_argument("--output_dir", type=str, default="notebooks/plots")
    parser.add_argument("--num_files", type=int, default=5, help="Limit number of files.")
    parser.add_argument("--device", type=str, default="0")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # 1. Load External Config
    print(f"Loading external model config from: {args.toml_config}")
    try:
        external_model_paths = toml.load(args.toml_config)
    except Exception as e:
        print(f"Error reading TOML: {e}")
        raise SystemExit(1)

    # 2. Determine Sequence Directory (Source of Truth for Images)
    task_name = args.task.replace("dmc_", "")
    if args.input_dir:
        sequence_dir = pathlib.Path(args.input_dir)
    else:
        sequence_dir = pathlib.Path(f"/work/MLShare/vadrl_v5/dmc/test/{args.quality}/{task_name}")

    if not sequence_dir.exists():
        print(f"Sequence directory not found: {sequence_dir}")
        raise SystemExit(1)

    # 3. Discover Files to Process (Driven by TOML folders)
    # We collect a Set of all .npz filenames present in ANY of the external model folders
    candidate_files = set()
    
    print("Scanning TOML folders for candidate files...")
    for model_name, path_str in external_model_paths["models"].items():
        p = pathlib.Path(path_str)
        if p.exists() and p.is_dir():
            # Add all .npz files from this folder
            found = [x.name for x in p.glob("*.npz")]
            candidate_files.update(found)
        else:
            print(f"Warning: Folder for '{model_name}' not found: {path_str}")

    # 4. Filter Candidates against Sequence Directory
    # We can only process files that ALSO exist in the full sequence directory (need images for Viper)
    valid_files_to_process = []
    
    for fname in candidate_files:
        full_seq_path = sequence_dir / fname
        if full_seq_path.exists():
            valid_files_to_process.append(fname)
            
    valid_files_to_process.sort()

    # Apply Limit
    if args.num_files != -1:
        valid_files_to_process = valid_files_to_process[:args.num_files]

    if not valid_files_to_process:
        print("No matching files found between TOML folders and sequence directory.")
        sys.exit(0)

    # 5. Load Reward Model
    print(f"Loading Viper reward model: {args.rm_key}")
    reward_model = LOAD_REWARD_MODEL_DICT[args.rm_key](
        task=args.task, minibatch_size=2, encoding_minibatch_size=32, compute_joint=True,
    )
    
    output_subdir = os.path.join(args.output_dir, f"{args.quality}_{task_name}")

    # 6. Run
    process_and_plot_sequences(
        valid_files_to_process,
        sequence_dir,
        reward_model,
        external_model_paths=external_model_paths,
        output_dir=output_subdir,
    )
