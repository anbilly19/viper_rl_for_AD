************************************************
**Forked repo for benchmarking and research, this code contains some minor changes to make runnable on a personal environment, check the original in the links below.**

*************************************************

# Video Prediction Models as Rewards for Reinforcement Learning

Code for VIPER (Video Predcition Rewards), a general algorithm which leverages video prediction models as priors for Reinforcement Learning.

<img src='https://github.com/Alescontrela/viper_rl/assets/13845012/b941627c-2ce1-49c3-9894-8d0b0e939462' width='100%'>

If you found this code useful, please reference it in your paper:

```
@article{escontrela2023viper,
  title={Video Prediction Models as Rewards for Reinforcement Learning},
  author={Alejandro Escontrela and Ademi Adeniji and Wilson Yan and Ajay Jain and Xue Bin Peng and Ken Goldberg and Youngwoon Lee and Danijar Hafner and Pieter Abbeel},
  journal={arXiv preprint arXiv:2305.14343},
  year={2023}
}
```

For more information:
- [Research paper][paper]
- [Project website][website]
- [Twitter summary][tweet]

## Install :
### adopted from original
Create a conda environment with Python 3.8:

```
conda create -n viper python=3.8
conda activate viper
```

Install [Jax][jax]. (need not be separately installed the exact .whl file is linked in requirements already)

Install dependencies:
```
pip install -r requirements.txt
```

## Downloading Data

Download the DeepMind Control Suite expert dataset with the following command:

```
python -m viper_rl_data.download dataset dmc
```

and the Atari dataset with:

```
python -m viper_rl_data.download dataset atari
```

This will produce datasets in `<VIPER_INSTALL_PATH>/viper_rl_data/datasets/` which are used for training the video prediction model. The location of the datasets can be retrieved via the `viper_rl_data.VIPER_DATASET_PATH` variable.

## Downloading Checkpoints

Download the DeepMind Control Suite videogpt/vqgan checkpoints with:

```
python -m viper_rl_data.download checkpoint dmc
```

and the Atari checkpoint with:

```
python -m viper_rl_data.download checkpoint atari
```

This will produce video model checkpoints in `<VIPER_INSTALL_PATH>/viper_rl_data/checkpoints/`, which are used downstream for RL. The location of the checkpoints can be retrieved via the `viper_rl_data.VIPER_CHECKPOINT_PATH` variable.

## Video Model Training

Use the following command to first train a VQ-GAN:
```
python scripts/train_vqgan.py -o viper_rl_data/checkpoints/dmc_vqgan -c viper_rl/configs/vqgan/dmc.yaml
```

To train the VideoGPT, update `ae_ckpt` in `viper_rl/configs/dmc.yaml` to point to the VQGAN checkpoint, and then run:
```
python scripts/train_videogpt.py -o viper_rl_data/checkpoints/dmc_videogpt_l16_s1 -c viper_rl/configs/videogpt/dmc.yaml
```

## Policy training

Checkpoints for various models can be found in `viper_rl/videogpt/reward_models/__init__.py`. To use one of these video models during policy optimization, simply specify it with the `--reward_model` argument.  e.g.

```
python scripts/train_dreamer.py --configs=dmc_vision videogpt_prior_rb --task=dmc_cartpole_balance --reward_model=dmc_clen16_fskip4 --logdir=~/logdir
```

Custom checkpoint directories can be specified with the `$VIPER_CHECKPOINT_DIR` environment variable. The default checkpoint path is set to `viper_rl_data/checkpoints/`.

**Note**: For Atari, you will need to install [atari-py][ataripy] and follow the Atari 2600 VCS ROM install instructions.

[jax]: https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier
[paper]: https://arxiv.org/pdf/2305.14343.pdf
[website]: https://escontrela.me/viper
[tweet]: https://twitter.com/AleEscontrela/status/1661363555495710721?s=20
[ataripy]: https://github.com/openai/atari-py


## Plotting function for Reward model vs. Anomaly GT

Refer file -> [reward_plots.py](./notebooks/reward_plots.py)

### Command line arguments

| Argument     | Type | Default              | Description                                                       |
| ------------ | ---- | -------------------- | ----------------------------------------------------------------- |
| --rm_key     | str  | dmc_clen16_fskip4    | Reward model identifier key from LOAD_REWARD_MODEL_DICT           |
| --task       | str  | dmc_cartpole_balance | DMC task name (e.g., dmc_cartpole_balance, dmc_walker_walk)       |
| --quality    | str  | random               | Data quality level: mixed, normal, or random                      |
| --input_dir  | str  | None                 | Custom input directory path. If None, uses default path structure |
| --output_dir | str  | notebooks/plots      | Base directory for saving output plots                            |
| --num_files  | int  | 5                    | Number of files to process. Use -1 to process all files           |
| --device     | str  | 0                    | GPU device ID for CUDA_VISIBLE_DEVICES                            |

### Expected Output

Each processed .npz file generates a PNG plot containing:

- Blue line: Frame-by-frame reward density scores (left y-axis)

- Red line (only in mixed): Anomaly labels from source_label (right y-axis)

- X-axis: Frame index

- Title: Original filename