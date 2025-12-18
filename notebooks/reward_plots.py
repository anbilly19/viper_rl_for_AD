import os
import sys
import pathlib
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse  # Added for CLI


# Add parent directory to path for imports
directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))

import notebook_utils as nbu
from viper_rl.videogpt.reward_models import LOAD_REWARD_MODEL_DICT


def save_plot_to_png(values, 
                     source_labels=None,
                     filename="output.png", 
                     title="Rewards", 
                     xlabel="frame index", 
                     ylabel="score", 
                     figsize=(12, 4)):
    """
    Plots model rewards and optional source labels (anomalies).
    """
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # --- Plot 1: Model Rewards (Left Axis) ---
    x_rewards = np.arange(len(values))
    line1 = ax1.plot(x_rewards, values, label="Model Reward", linewidth=1.5, color='tab:blue')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(axis='x', linestyle='--', alpha=0.5, color='gray')

    lines = line1
    
    # --- Plot 2: Source Labels (Right Axis) - Only if provided ---
    if source_labels is not None:
        ax2 = ax1.twinx()
        
        # Handle scalar vs array source_labels
        if np.ndim(source_labels) == 0:
            plot_labels = np.full(len(values), source_labels)
            x_labels = x_rewards
        else:
            plot_labels = source_labels
            x_labels = np.arange(len(source_labels))

        line2 = ax2.plot(x_labels, plot_labels, label="Anomaly (Source Label)", 
                         linewidth=1.5, color='tab:red', linestyle='-', alpha=0.7)
        
        ax2.set_ylabel("Source Label (0=Normal, 1=Anomaly)", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_ylim(-0.1, 1.1)
        
        lines = line1 + line2

    # Create combined legend
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")
    
    ax1.set_title(title)
    ax1.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)

def plot_rewards_from_npz_files(file_list, 
                                reward_model, 
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
        
        save_plot_to_png(
            rewards, 
            source_labels=source_labels, 
            filename=save_path, 
            title=f"Rewards: {base_name}"
        )
        processed_count += 1

    print(f"Done. Processed {processed_count} files.")

# --- Main Execution ---

if __name__ == "__main__":
    # Parsing CLI Arguments
    parser = argparse.ArgumentParser(description="Evaluate and plot reward model outputs for npz sequences.")
    
    parser.add_argument("--rm_key", type=str, default='dmc_clen16_fskip4', 
                        help="Key for the reward model dictionary.")
    parser.add_argument("--task", type=str, default='dmc_cartpole_balance', 
                        help="DMC task name.")
    parser.add_argument("--quality", type=str, default='random', 
                        choices=['mixed', 'normal', 'random'], 
                        help="Data quality level.")
    parser.add_argument("--input_dir", type=str, default=None, 
                        help="Path to input .npz directory. If None, uses default path structure.")
    parser.add_argument("--output_dir", type=str, default="notebooks/plots", 
                        help="Base directory to save output plots.")
    parser.add_argument("--num_files", type=int, default=5, 
                        help="Number of files to process. Set to -1 to process all.")
    parser.add_argument("--device", type=str, default='0', 
                        help="CUDA_VISIBLE_DEVICES ID.")

    args = parser.parse_args()

    # Environment setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    print(f"Loading reward model: {args.rm_key}")
    reward_model = LOAD_REWARD_MODEL_DICT[args.rm_key](
        task=args.task, 
        minibatch_size=2, 
        encoding_minibatch_size=32, 
        compute_joint=True
    )

    task_name = args.task.replace('dmc_','')  # Extract task name without 'dmc_'] 
    # Determine Input Directory
    if args.input_dir:
        sequence_dir = args.input_dir
    else:
        # Default path logic from original script
        # Note: 'cartpole_balance' part is hardcoded here matching original logic
        # You may want to make this dynamic based on args.task if needed
        sequence_dir = f'/work/MLShare/vadrl_v5/dmc/test/{args.quality}/{task_name}'
    
    if os.path.exists(sequence_dir):
        all_files = [os.path.join(sequence_dir, f) for f in os.listdir(sequence_dir) if f.endswith('.npz')]
        all_files.sort()
        
        # Filter files based on num_files argument
        if args.num_files != -1:
            files_to_process = all_files[:args.num_files]
        else:
            files_to_process = all_files
        
        output_subdir = os.path.join(args.output_dir, f"{args.quality}_{task_name}")
        
        plot_rewards_from_npz_files(
            files_to_process, 
            reward_model, 
            output_dir=output_subdir
        )
    else:
        print(f"Directory not found: {sequence_dir}")
