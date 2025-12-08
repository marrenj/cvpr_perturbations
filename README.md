This repository provides tools for training CLIP-HBA models with various behavioral perturbations and evaluating their effects on alignment performance. In particular, the training pipeline supports adding label or target noise (e.g. label shuffle, target noise/random targets, image noise, etc.) during training, so one can systematically study how these perturbations impact model behavior. The core model is a CLIP-based HBA network (with DoRA layers) that is instantiated in code as CLIPHBA. The training logic (run_training_experiment) reads a config file describing the architecture and perturbation schedule, applies the chosen perturbation strategy via choose_perturbation_strategy, and trains the model accordingly. The repository also includes an inference pipeline (run_inference) which loads a trained checkpoint, applies the model to a test dataset (e.g. the THINGS, NOD, or Nights dataset), and collects evaluation results.

The folder structure is now organized into clear, modular components:
- src/ – Core code. This includes submodules for models (e.g. CLIP-HBA definitions), data (dataset loaders like ThingsDataset, NODDataset, etc.), training (training loops and trainer functions), inference (scripts to run evaluation on held-out data), perturbations (perturbation strategy implementations like TargetNoisePerturbation, LabelShufflePerturbation, etc.), evaluation (e.g. RSA or behavioral analysis utilities), and utils (logging, seeding, path setup, etc.). For example, the training code imports from src.models.clip_hba, src.data, and src.perturbations to build and perturb the model.
- experiments/ – Experiment outputs and logs. Each run (defined by a config) saves its results here, including model checkpoints, training logs, random seeds, and any analysis outputs. Organizing outputs per experiment makes it easy to compare across runs.
- scripts/ – High-level entry-point scripts. For example:
  - scripts/run_training.py loads a training config and launches the training pipeline.
  - scripts/run_inference.py loads an inference config and runs the evaluation (e.g. computing predictions or metrics on a test set).
These scripts parse command-line arguments (e.g. --config) and call into the src.training or src.inference functions to execute the experiment.
- data/ – Data storage (often git-ignored). This may include raw stimulus annotations or preprocessed files used by the datasets (e.g. the image directories and embedding CSVs referenced in configs).
- results/ (or experiments/ as above) – Collected outputs. Typically contains subfolders for logs, figures, metrics, etc., corresponding to each experimental run.
- tests/, notebooks/, etc. – Additional code (e.g. unit tests or analysis notebooks) as needed.

Usage

Configure your experiment via the provided YAML files and run the training or inference scripts. For example, to train a model you might run:

'python scripts/run_training.py --config configs/training/baseline_seed3.yaml'

And to run inference (evaluation) on a trained model, e.g. on the “Nights” dataset:

'python scripts/run_inference.py --config configs/inference/nights.yaml'

Each --config path points to a YAML file under configs/ that sets all relevant parameters. To customize an experiment, copy or edit one of the YAML templates: change the backbone (e.g. backbone: ViT-L/14), dataset paths, or perturbation settings (perturb_type, perturb_epoch, etc.) in the YAML file. The scripts will load these configs and automatically apply your chosen settings.

'backbone: ViT-L/14
vision_layers: 2
transformer_layers: 1
rank: 32

epochs: 500
batch_size: 64
lr: 3e-4
criterion: MSELoss
random_seed: 3
cuda: 0

dataset_type: things
img_annotations_file: path/to/annotations.csv
img_dir: path/to/images

perturb_type: random_target   # e.g. 'None', 'random_target', 'label_shuffle', etc.
perturb_epoch: 10
perturb_length: 5
perturb_seed: 42

checkpoint_path: experiments/2025-...  # where to save model checkpoints
logger: None'

And similarly the inference configs include fields like model_weights_path, dataset, and inference_save_dir.

These settings are all loaded at runtime (see code in src/training/trainer.py and src/inference/inference_core.py). For example, the training code uses choose_perturbation_strategy(perturb_type, perturb_epoch, perturb_length, perturb_seed) to apply the specified noise or shuffle each epoch. The inference code loads a saved checkpoint, constructs the same CLIPHBA model, applies DoRA layers, and then calls a dataset-specific evaluation function (e.g. evaluate_nights) based on config['dataset'].

By editing or creating your own YAML files, you can run new experiments. For example, to sweep over different seeds or perturbation lengths, duplicate a config and change the random_seed, perturb_length, etc., then run run_training.py for each. The modular structure makes it easy to plug in new perturbations or datasets if needed.

### References

The core logic for training is implemented in src/training/ (see run_training_experiment in trainer.py). Perturbation strategies are defined in src/perturbations/perturbation_utils.py (classes like TargetNoisePerturbation, LabelShufflePerturbation, etc.). The inference logic is in src/inference/ (e.g. run_inference in inference_core.py). Configuration is fully driven by the YAML files under configs/, as shown above. Together, these components support running reproducible experiments on CLIP-HBA robustness.



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