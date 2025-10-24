## Baseline CLIP-HBA Training

Can be found in clip_hba_behavior/training_scripts/functions/cvpr_train_behavior_things_baseline.py

The baseline config and setup were created using clip_hba_behavior/cvpr_train_behavior_baseline.py

## Perturbation Experiments

Can be found in clip_hba_behavior/training_scripts/cvpr_train_behavior_things_pipeline.py

The single epoch perturbation sweep's config and setup were created using clip_hba_behavior/cvpr_train_behavior.py

The perturbation length experiments' config and setup were created using cvpr_train_behavior_light.py, with hyperparameters set using a separate tsv file to specify starting epochs, perturbation lengths, seeds, etc.

## Analysis

The current analyses can be found in clip_hba_behavior/analysis/analysis_training_results.ipynb