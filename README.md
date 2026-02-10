## About

This repository provides tools for training CLIP-HBA models with various perturbations and evaluating the perturbations' effects on model performance and model-human alignment. In particular, the training pipeline supports adding input or target noise in the form of (1) label shuffling, (2) generating random target embeddings, (3) generaating random gaussian image noise, and (4) generating uniform grayscale input images. This way, one can systematically study how these perturbations impact model behavior.

The core model is a CLIP-based network with DoRA layers applied, known as CLIP-HBA-Behavior from Zhao et al., 2025<sup>1</sup>. It is instantiated in code as CLIPHBA.

The training logic (`run_training_experiment`) reads a config file describing the architecture and perturbation schedule, applies the chosen perturbation strategy via `choose_perturbation_strategy`, and trains the model accordingly. The repository also includes an inference pipeline (`run_inference`) which loads a trained model checkpoint, applies the model to a test dataset (e.g. the THINGS or NIGHTS dataset), and collects evaluation results.

The training and inference in this repository is built off of the original CLIP-HBA sourcecode published at https://github.com/stephenczhao/CLIP-HBA-Official/tree/main<sup>1</sup>.

## Data Availability

This project uses data from three datasets: (1) THINGS, (2) NOD, and (3) NIGHTS.

### THINGS

As in the original CLIP-HBA paper, the THINGS dataset in this project consists of the same 1,854 images that were used to approximate human perceptual embeddings in Hebart et al., . Of these 1,854 images in the paper by Hebart et al., , 48 of them were fully sampled in the triplet odd-one-out task, and therefore held out for inference in the current project. In other words, we used the 48x48 RDM created from the 48 fully-sampled image embeddings to measure model alignment with human behavior. 1,806 THINGS images were thus left for training, which we divided into an 80/20 train/validation split.

The 66-dimensional, human-derived embeddings for the 1,854 training images can be found in the `data/` folder as `spose_embedding66d_rescaled.csv`. The original file can be downloaded from [WEBSITE]. The embeddings for the 1,806 training images are further set aside in `data/spose_embedding66d_rescaled_1806train.csv`, and the embeddings for the 48 test images are further set aside in `data/spose_embedding66d_rescaled_48val_reordered.csv` for each of use.

The 48x48 behaviorally-derived RDM used for inference is also available in the `data/` folder as `RDM48_triplet.mat` for ease of use.

### NOD



### NIGHTS  



## Quickstart

### 1) Clone the repo

``` bash
git clone https://github.com/marrenj/cvpr_perturbations.git
cd cvpr_perturbations
```

### 2) Create environment

Use `environment.yml` for reproducible installs (recommended). `requirements.txt` is provided for convenience but may require manual PyTorch/CUDA selection.

``` bash
conda env create -f environment.yml
conda activate cvpr_perturbations
```

### 3) Verify installation

``` bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output: `True`

### 4) Run a training experiment (minimal example)

``` bash
python scripts/run_training.py \
  --config configs/training_config.yaml
```

This will:
- Load CLIP-HBA with DoRA layers
- Apply the perturbation schedule defined in the config
- Train the model
- Save checkpoints + logs

### 5) Run inference on a trained model

``` bash
python scripts/run_inference.py \
  --config configs/inference/nights.yaml
```

This will:
- Load a trained checkpoint
- Extract last-layer embeddings and create an RDM
- Compute behavioral or neural alignment metrics
- Save results

### 6) Configure your own experiment

Configuration files provide direct access to the main experimental parameters such as the type of perturbation to apply and when to apply it.

Configure your experiment via the provided YAML files and run the training or inference scripts. These settings are all loaded at runtime (see code in `src/training/trainer.py` and `src/inference/inference_core.py`). For example, the training code uses `choose_perturbation_strategy` to apply the specified perturbation at a given epoch(s). The inference code loads a saved checkpoint and calls a dataset-specific evaluation function (e.g. `evaluate_nights`) based on `config['dataset']`.

By editing or creating your own YAML files, you can run new experiments. For example, to sweep over different seeds or perturbation lengths, duplicate a config and change the random_seed, perturb_length, etc., then run `run_training.py` for each. The modular structure makes it easy to plug in new perturbations or datasets if needed.

### 7) Data paths

Datasets are not included in the repository.

Update paths in the YAML config:

``` yaml
img_dir: /path/to/images`
img_annotations_file: /path/to/annotations.csv
```

### Reproducibility / logging

Run naming: composed from backbone, rank, perturbation type/epoch/length/seed, initialization seed, and behavioral RSA flag (e.g., `vit_l_14_rank32_perturb-type-random_target_epoch29_length1_perturb-seed1_init-seed1_behavioral-rsa-True`).

Config snapshots: each run writes both `training_config_snapshot.yaml` and `resolved_config.yaml` inside its `save_path`.

WandB capture: metrics (losses, RSA correlation/p-value), model checkpoints on the configured cadence, and the full training_res.csv at the end of a training run; runs default to offline mode when wandb_project/wandb_entity are unset.

### A note about WandB logging

As mentioned above, this repo allows for experiment logging with Weights and Biases. If you wish to utilize the Weights and Biases API, simply point the arguments in the `WANDB CONFIGURATION` section of `training_config.yaml` to your WandB project name and, optionally, your WandB username. If these configurations are left blank, WandB will run in offline mode, and all logging will take place on your local file system.

## Folder Structure

The folder structure is organized into modular components as follows:

- `src/` – Core code. This includes submodules for models (e.g., CLIP-HBA definitions), data (dataset loaders like `ThingsBehavioralDataset`), training (training loops), inference (scripts to run evaluation on held-out data), perturbations (perturbation strategy implementations like `TargetNoisePerturbation`, `LabelShufflePerturbation`, etc.), evaluation (e.g., RSA against behavioral RDMs or NIGHTS triplet accuracy), and generic utility functions (logging, seeding, path setup, etc.). For example, the training code imports from `src.models.clip_hba`, `src.data`, and `src.perturbations` to build and perturb the model.
  
- `scripts/` – High-level entry-point scripts. These scripts parse command-line arguments (e.g., `--config`) and call into the `src.training` or `src.inference` functions to execute the experiment. For example:
  - `scripts/run_training.py` loads a training config and launches the training pipeline.
  - `scripts/run_inference.py` loads an inference config and runs the evaluation (e.g. computing predictions or metrics on a test set).
  
- `data/` – Data storage (often git-ignored). This may include raw stimulus annotations or preprocessed files used by the datasets (e.g., the CSVs containing the SPoSE embeddings for each image, or the 48x48 behavioral RDM for testing referenced in configs).

- `notebooks/` – Additional code (e.g. analysis notebooks) as needed.
  - `notebooks/figures/all_figures.ipynb` contains code to generate each figure used in the paper.

## Usage

The core logic for training is implemented in `src/training/` (see `run_training_experiment` in `trainer.py`). Perturbation strategies are defined in `src/perturbations/perturbation_utils.py` (classes like `TargetNoisePerturbation`, `LabelShufflePerturbation`, etc.). The inference logic is in `src/inference/` (see `run_inference` in `inference_core.py`). Configuration is fully driven by the YAML files under `configs/`, as shown above.

## Inference 

For inference, set `evaluation_type` to `behavioral` or `neural` to extract embeddings, build RDMs, and run RSA against a reference RDM (provided via `reference_rdm_path` or derived from dataset targets); set `evaluation_type: triplet` to evaluate the NIGHTS triplet task via `src/inference/evaluate_nights.py`. Each run returns and saves the score per checkpoint epoch.

### References

1. Zhao, S. C., Hu, Y., Lee, J., Bender, A., Mazumdar, T., Wallace, M., & Tovar, D. A. (2025). Shifting attention to you: Personalized brain-inspired AI models. arXiv. https://arxiv.org/abs/2502.04658