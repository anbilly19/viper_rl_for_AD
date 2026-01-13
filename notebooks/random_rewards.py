import argparse
import random
from pathlib import Path
from calc_score import calculate_sum_loss_anomaly_score
import numpy as np


def main(task_name="cheetah_run"):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="Folder containing *.npz clips.")
    ap.add_argument("--output_dir", type=str, required=True, help="Folder to write reward *.npz files into.")
    ap.add_argument("--num", type=int, default=3, required=True, help="How many random *.npz files to process.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling.")

    ap.add_argument("--ckpt", type=str, default='AdoRe/src/adore/ad_rewards/mae/experiments/dmc_all/four_tasks/checkpoint-epoch012.pth')
    ap.add_argument("--cfg_root", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no_norm", action="store_true", help="Return raw fused values (no min-max).")
    ap.add_argument("--no_png_grad", action="store_true", help="Do NOT read dataset gradient PNGs.")
    ap.add_argument("--neighbor_gap", type=int, default=3)
    ap.add_argument("--smooth_range", type=int, default=0)
    ap.add_argument("--smooth_mu", type=int, default=0)

    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # mkdir -p equivalent 
    all_npz = sorted(in_dir.glob("*.npz"))
    if not all_npz:
        raise RuntimeError(f"No .npz files found in: {in_dir}")

    if args.num < 1:
        raise ValueError("--num must be >= 1")
    if args.num > len(all_npz):
        raise ValueError(f"--num={args.num} but only {len(all_npz)} .npz files exist in {in_dir}")

    rng = random.Random(args.seed)
    chosen = rng.sample(all_npz, k=args.num)  # random subset without replacement 

    for clip_path in chosen:
        scores = calculate_sum_loss_anomaly_score(
            clip_path,
            ckpt=args.ckpt,
            cfg_root=args.cfg_root,
            device=args.device,
            smooth_range=args.smooth_range,
            smooth_mu=args.smooth_mu,
            normalise=(False if args.no_norm else None),
            use_dataset_gradients=not args.no_png_grad,
            neighbor_gap=args.neighbor_gap,
        )

        # Save reward with same filename into output_dir
        out_path = out_dir / clip_path.name
        np.savez_compressed(out_path, reward=np.asarray(scores))  # saves to .npz 

        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
