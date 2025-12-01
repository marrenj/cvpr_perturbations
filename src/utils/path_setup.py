import os


def setup_paths(checkpoint_path):
    """
    Set up paths for the training run.
    """
    os.makedirs(checkpoint_path, exist_ok=True)
    training_res_path = os.path.join(checkpoint_path, 'training_res.csv')
    random_state_path = os.path.join(checkpoint_path, 'random_states')
    
    return checkpoint_path, training_res_path, random_state_path