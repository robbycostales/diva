<!-- omit in toc -->
# Enabling Adaptive Agent Training in Open-Ended Simulators by Targeting Diversity

The official repository for the NeurIPS 2024 paper *[Enabling Adaptive Agent Training in Open-Ended Simulators by Targeting Diversity (Costales &  Nikolaidis)](https://robbycostales.com/divapaper)*. If you find the code helpful, please cite the corresponding paper:

```bibtex
@inproceedings{
    costales2024enabling,
    title={Enabling Adaptive Agent Training in Open-Ended Simulators by Targeting Diversity},
    author={Robby Costales and Stefanos Nikolaidis},
    booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=Xo1Yqyw7Yx}
}
```

<!-- omit in toc -->
## Contents

- [Code structure](#code-structure)
- [Setup](#setup)
  - [Terminal commands](#terminal-commands)
  - [Troubleshooting](#troubleshooting)
  - [W\&B](#wb)
- [Reproducing results](#reproducing-results)
- [Miscellaneous](#miscellaneous)
- [License](#license)

## Code structure

All algorithmic code is in the `diva/` directory.
Below is the structure of the most notable files and directories. 

- `main.py` — **main entry point** for running the code
- `metalearner.py` — meta-RL **training loop**
- `cfg/` — **configuration files** for domains and algorithms
- `components/` — main algorithmic components
  - `components/level_replay/` — level replay code (for **ACCEL**, **PLR**)
  - `components/qd/` — **QD** code (for DIVA's archive)
    - `components/qd/qd_module.py` — code most relevant to **DIVA**
  - `components/policy/` — code relevant to the RL policy
  - `components/vae/` — VAE code (for **VariBAD**'s encoder)
- `environments/` — environments and environment-specific code
  - `environments/alchemy/` — **Alchemy** environment and related code
  - `environments/box2d/` — **Racing** environment and related code
  - `environments/toygrid/` — **GridNav** environment and related code
- `utils/` — miscellaneous helper code

## Setup

These setup instructions asume the user is running Ubuntu 20.04 and has CUDA 12.
We use Anaconda to manage the environment.
Ensure that you also have CUDA toolkit installed and the latest version of Anaconda. 
Additionally, we use virtual displays for certain environments, for which you will need to install Xvfb (via e.g. `sudo apt install xvfb`).

### Terminal commands

```bash
git clone git@github.com:robbycostales/diva.git    # clone repository
cd diva                                            # navigate to directory
pip install --upgrade pip                          # upgrade pip if necessary
sudo apt install swig                              # necessary for Racing environment
. ./setup.sh                                       # set up conda env and install deps
```

### Troubleshooting

The error:

```bash
AttributeError: module '_Box2D' has no attribute 'RAND_LIMIT_swigconstant'
```

can be resolved with:

```bash
pip uninstall box2d-py
pip install box2d-py
```

### W&B

We use `wandb` for logging, which is free for academic use. 
Without `wandb`, you can still run the code
and use `tensorboard` for logging, but some functionality may be limited.

Use the following commands to login and/or verify your credentials.

```bash
wandb login
wandb login --verify
```

From `diva`, run `wandb_init.sh` to set necessary environment variables.
For `entity`, either enter your username or organization, and for project, enter
`diva`, or any other name you would like to use:

```bash
. ./wandb_init.sh
```

## Reproducing results

The following command structure can be used for reproducing the main results (within `diva`): 

```bash
python main.py wandb_label=<wandb_label> domain=<domain> meta=<meta> dist=<dist>
```

`wandb_label` is for logging. `domain` specifies the environment (`toygrid`, `alchemy`, `racing`), `meta` specifies the meta-RL learner (`varibad`, `rl2`), and `dist` specifies the task distribution (`diva`, `rplr`, `accel`, `dr`, `oracle`, `diva_plus`). For `diva_plus`, the default configuration assumes you will load an archive from a prior DIVA run.

For F1 results, set `domain.reg_env_id=CarRacing-F1-v0`. For DIVA, you will need to save an archive from a normal run first, and load it in (since DIVA uses samples from `domain.reg_env_id` to parameterize archive).

## Miscellaneous

To describe meta-RL rollouts within the same MDP, our work uses the language "episodes in a trial", while the Alchemy work uses "trials in an episode". Expect conflicting convenctions in certain parts of the code, especially surrounding the Alchemy environment.

## License

This code is released under the MIT License. Some code is adpated from other repos (see below). Please see their respective licenses for more information.

- `components/level_replay` and `environments/box2d` are adapted from the [DCD](https://github.com/facebookresearch/dcd/tree/main) repo.
- `components/policy`, `components/exploration/` , and `components/vae` are adapted from the [HyperX](https://github.com/lmzintgraf/hyperx) repo (some are from the original [VariBAD](https://github.com/lmzintgraf/varibad/tree/master) repo). Exploration bonuses are not used in our work, but are included in the repo because they may be useful for certain environments others may wish to implement.
- `components/qd` adapts some elements from the [pyribs](https://github.com/icaros-usc/pyribs) repo, and the [DSAGE](https://github.com/icaros-usc/dsage) repo. 
- `environments/alchemy` is adapted from the [dm_alchemy](https://github.com/deepmind/dm_alchemy) repo.
