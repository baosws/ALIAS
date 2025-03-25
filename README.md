# Reinforcement Learning for Causal Discovery without Acyclicity Constraints

## Project structure
```bash
├── data                                    # Put nonlinear GP data and Sachs data here
│
├── alias                                   # implementation of ALIAS
│   ├── ALIAS.py
│   ├── DAGEnv.py
│   ├── DAGScore.py
│   └── VecDAGEnv.py
│
├── utils
│   ├── dag_pruners.py                      # defines pruning methods, e.g., by linear weight, CAM pruning, CIT pruning
│   ├── datagen.py                          # defines data simulations
│   └── metrics.py                          # defines performance metrics
│
├── demo.ipynb                              # Demo notebook
├── run_experiment.py                       # Entry point to run experiments
├── analyzers.py                            # result analyzers for different experiment, is called automatically in run_experiment.py
│
├── configs                                 # configurations for ALIAS and experiments
│   ├── ALIAS_default.yml                   # default hyperparameters, which can be updated with hyperparameters in each specific experiment
│   ├── linear.yml                          # "Small to moderate graphs with linear-Gaussian data" experiment (Figure 2)
│   ├── dense.yml                           # "Dense graphs" experiment (Table 2)
│   ├── large.yml                           # "High-dimensional graphs" experiment (Table 2)
│   ├── different_noises.yml                # "Noise misspecification" experiment (Table 3)
│   ├── different_samplesizes.yml           # "Different sample sizes" experiment (Figure 3, Table 14 & 15)
│   ├── nonlinear_gp.yml                    # "Nonlinear data with Gaussian processes" experiment (Table 4)
│   ├── sachs.yml                           # "Real-world Sachs dataset" experiment (Table 5)
│   ├── runtime.yml                         # "Runtime" experiment (Figure 4b)
│   ├── different_lr.yml                    # "Learning rate" ablation study (Table 16)
│   ├── different_ent.yml                   # "Entropy weight" ablation study (Table 17)
│   ├── noisy.yml                           # "Noisy data" experiment (Table 20)
│   ├── confounder.yml                      # "Hidden counfounder" experiment (Table 21)
│   └── nonlinear_mlp.yml                   # "Nonlinear data with MLPs" experiment (Table 22)
│
├── CAM_1.0.tar.gz                          # R package of CAM: https://cran.r-project.org/src/contrib/Archive/CAM/CAM_1.0.tar.gz
├── setup_CAM.py                            # CAM setup code from gCastle: https://github.com/huawei-noah/trustworthyAI/blob/master/research/Causal%20Discovery%20with%20RL/setup_CAM.py
├── README.md
└── environment.yml                         # Conda environment
```

## Setup

```bash
conda env create -n alias --file environment.yml
conda activate alias
python setup_CAM.py
```

## Running demo

Run [demo.ipynb](demo.ipynb)

## Running experiments

```bash
python run_experiment.py <experiment name>
```

Replace `<experiment name>` with the name of the experiment exactly as in `./configs` (excluding the `.yml` extension).

For example:
```bash
python run_experiment.py linear
```

For nonlinear experiments with GP, download this [dataset](https://github.com/kurowasan/GraN-DAG/blob/master/data/data_p10_e40_n1000_GP.zip) from GraN-DAG's repo then extract it to `./data` before running the experiment.
