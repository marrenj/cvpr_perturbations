## Baseline CLIP-HBA Training

Can be found in training_scripts/functions/cvpr_train_behavior_things_baseline.py

The baseline training was conducted using cvpr_train_behavior_baseline.py

## Perturbation Experiments

Can be found in training_scripts/cvpr_train_behavior_things_pipeline.py

The single epoch perturbation sweep was conducted using cvpr_train_behavior.py

The perturbation lengths experiments were conducted using cvpr_train_behavior_light.py, with hyperparameters set using a separate tsv file to specify starting epochs, perturbation lengths, seeds, etc.

## Analysis

The current analyses can be found in analysis/analysis_training_results.ipynb