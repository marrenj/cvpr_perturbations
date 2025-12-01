import os
import csv
from tqdm import tqdm

from src.training.test_loop import evaluate_model
from src.utils.save_random_states import save_random_states
from src.models.clip_hba.clip_hba_utils import save_dora_parameters


def train_model(
    model,
    train_loader,
    test_loader,
    device,
    optimizer,
    criterion,
    epochs,
    training_res_path,
    checkpoint_path,
    random_state_path,
    logger=None,
    early_stopping_patience=5,
    dataloader_generator=None,
    vision_layers=1,
    transformer_layers=1,
    perturb_strategy=None,
    start_epoch=0,
):
    """
    End-to-end training loop that logs metrics, checkpoints DoRA weights, and
    optionally performs early stopping.

    Parameters
    ----------
    model : torch.nn.Module | torch.nn.DataParallel
        Model to optimize; assumed compatible with ``train_loader`` batches.
    train_loader : DataLoader
        Iterable producing (metadata, images, targets) for training.
    test_loader : DataLoader
        Iterable used for computing validation loss each epoch.
    device : torch.device
        Device to which tensors are moved before forward/backward passes.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    criterion : Callable
        Loss function applied to ``(predictions, targets)``.
    epochs : int
        Maximum number of training epochs.
    training_res_path : str | Path
        CSV filepath where per-epoch losses are recorded.
    checkpoint_path : str | Path
        Directory where DoRA checkpoints are written.
    random_state_path : str | Path
        Directory that stores RNG and optimizer state snapshots.
    logger : logging.Logger, optional
        Logger for structured output; defaults to ``print`` when omitted.
    early_stopping_patience : int, default 5
        Number of consecutive epochs without validation improvement before
        stopping early.
    dataloader_generator : torch.Generator, optional
        Generator whose state is persisted for deterministic shuffling.
    vision_layers : int, default 1
        Number of final visual transformer blocks with DoRA adapters to save.
    transformer_layers : int, default 1
        Number of final text transformer blocks with DoRA adapters to save.
    start_epoch : int, default 0
        Epoch index to begin (useful when resuming from checkpoints).
    """
    best_test_loss = float('inf')
    epochs_no_improve = 0

    # Use logger if provided, otherwise use print
    log = logger.info if logger else print

    # initial evaluation
    log("*********************************")
    log("Evaluating initial model")
    best_test_loss = evaluate_model(model, test_loader, device, criterion)
    # initial_behavioral_rsa_rho, initial_behavioral_rsa_p_value, initial_model_rdm = behavioral_RSA(model, inference_loader, device)
    log(f"Initial Validation Loss: {best_test_loss:.4f}")
    # log(f"Initial Behavioral RSA Correlation & p-value: {initial_behavioral_rsa_rho:.4f}, {initial_behavioral_rsa_p_value:.4f}")
    log("*********************************\n")

    # Create folder to store checkpoints
    os.makedirs(checkpoint_path, exist_ok=True)

    # Create directory for training results CSV if it doesn't exist
    os.makedirs(os.path.dirname(training_res_path), exist_ok=True)

    headers = ['epoch', 'train_loss', 'test_loss']

    with open(training_res_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        epoch_idx = epoch

        if perturb_strategy.is_active_epoch(epoch_idx):
            logger.info(f"Applying {perturb_strategy.__class__.__name__} perturbation during epoch {epoch}")

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs}")
        for batch_idx, (_, images, targets) in progress_bar:

            images = images.to(device)
            targets = targets.to(device)

            images, targets = perturb_strategy.apply_to_batch(images, targets, device, epoch_idx, batch_idx)

            optimizer.zero_grad()
            predictions = model(images)
            
            loss = criterion(predictions, targets)
            progress_bar.set_postfix({'loss': loss.item()})
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
        avg_train_loss = total_loss / len(train_loader.dataset)

        # Evaluate after every epoch
        avg_test_loss = evaluate_model(model, test_loader, device, criterion)
        print(f"Epoch {epoch}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_test_loss:.4f}")
        log(f"Epoch {epoch}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_test_loss:.4f}")

        if perturb_strategy.is_active_epoch(epoch_idx):
            logger.info(f"*** Perturbation '{perturb_strategy.__class__.__name__}' was applied during epoch {epoch} ***")

        # # Conduct behavioral RSA at every epoch
        # rho, p_value, model_rdm = behavioral_RSA(model, inference_loader, device)
        # log(f"Behavioral RSA Correlation & p-value: {rho:.4f}, {p_value:.4f}")
        # model.train() # put the model back in training mode

        # Prepare the data row with the epoch number and loss values
        data_row = [epoch, avg_train_loss, avg_test_loss]

        # Append the data row to the CSV file
        with open(training_res_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data_row)

        # Save random states and optimizer after every epoch for full reproducibility
        save_random_states(optimizer, epoch, random_state_path, dataloader_generator, logger=logger)

        # Save the DoRA parameters (i.e., the checkpoint weights)
        save_dora_parameters(
            model,
            checkpoint_path,
            epoch,
            vision_layers,
            transformer_layers,
            log_fn=log,
        )
        log(f"Checkpoint saved for epoch {epoch}")

        # Check for early stopping and saving checkpoint
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == early_stopping_patience:
            log("\n\n*********************************")
            log(f"Early stopping triggered at epoch {epoch}")
            log("*********************************\n\n")
            break