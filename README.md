## Baseline CLIP-HBA Training

Can be found in clip_hba_behavior/training_scripts/functions/cvpr_train_behavior_things_baseline.py

The baseline config and setup were created using clip_hba_behavior/cvpr_train_behavior_baseline.py

## Perturbation Experiments

Can be found in clip_hba_behavior/training_scripts/cvpr_train_behavior_things_pipeline.py

The single epoch perturbation sweep's config and setup were created using clip_hba_behavior/cvpr_train_behavior.py

The perturbation length experiments' config and setup were created using cvpr_train_behavior_light.py, with hyperparameters set using a separate tsv file to specify starting epochs, perturbation lengths, seeds, etc.

## Analysis

The current analyses can be found in clip_hba_behavior/analysis/analysis_training_results.ipynb



your_project/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── perturbations/
│   ├── visualization/
│   ├── utils/
│   └── __init__.py
│
├── configs/
│
├── experiments/
│
├── scripts/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── external/
│   └── README.md
│
├── results/
│   ├── logs/
│   ├── checkpoints/
│   ├── figures/
│   └── metrics/
│
├── tests/
│
├── README.md
└── requirements.txt (or environment.yml)



your_project/
│
├── src/
│   ├── data/                           # Dataset classes 
│   │   ├── things_dataset.py
│   │   ├── nights_dataset.py
│   │   └── transforms.py
│   │
│   ├── models/                         # Model architectures
│   │   ├── clip_utils.py
│   │   ├── hba_head.py
│   │   └── dora.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── train_loop.py
│   │   ├── metrics.py
│   │   └── checkpoints.py
│   │
│   ├── evaluation/
│   │   ├── rsa.py
│   │   ├── behavior_eval.py
│   │   └── inference.py
│   │
│   ├── perturbations/
│   │   ├── random_target.py
│   │   ├── blank_target.py
│   │   └── epoch_window.py
│   │
│   ├── visualization/                  # Reusable plotting functions
│   │   ├── ba_loss.py
│   │   ├── log_ratio.py
│   │   └── embeddings.py
│   │
│   ├── utils/                          # Reusable helper functions that don't fit elsewhere
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── seed.py
│   │   ├── paths.py
│   │   └── io.py
│   │
│   └── __init__.py
│
├── configs/                            # Configuration files
│   ├── baseline.yaml
│   ├── perturb_random.yaml
│   ├── perturb_length.yaml
│   ├── model/
│   │    ├── clip_hba.yaml
│   │    └── vit_l14.yaml
│   └── sweep/
│        ├── sweep_lr.yaml
│        └── sweep_perturb.yaml
│
├── scripts/                            # Thin, entry-point training/evaluation scripts (mostsly calling function from other modules and loading 
                                        # configs)
│   ├── train.py
│   ├── eval.py
│   ├── run_sweep.py
│   ├── plot_ba_loss.py
│   └── apply_perturbation.py
│
├── experiments/                        # Experiment-specific code/logs
│   ├── 2025-02-01_baseline/
│   │   ├── config.yaml
│   │   └── README.md
│   ├── 2025-02-02_random_target/
│   └── 2025-02-03_length_sweep/
│
├── results/
│   ├── logs/
│   ├── checkpoints/
│   ├── figures/
│   └── metrics/
│
├── data/                               # Data storage (often gitignored)
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── tests/                              # Unit tests
│
├── notebooks/                          # Jupyter notebooks for exploration
│
├── README.md
└── requirements.txt                    # Dependencies