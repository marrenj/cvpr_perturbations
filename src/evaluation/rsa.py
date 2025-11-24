import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity


def compute_rdm(embeddings, distance_metric='pearson'):
    """
    Compute a representational dissimilarity matrix from embeddings.

    Args:
        embeddings (np.ndarray | torch.Tensor): Row-wise embedding matrix.
        distance_metric (str): Dissimilarity to apply. Supports ``'pearson'`` and
            ``'cosine'``.

    Returns:
        np.ndarray: Square RDM whose entries represent pairwise dissimilarities.

    Raises:
        ValueError: If ``distance_metric`` is not a supported option.
    """

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    if distance_metric == 'pearson':
        rdm = 1 - np.corrcoef(embeddings)
        np.fill_diagonal(rdm, 0)
        return rdm
    elif distance_metric == 'cosine':
        rdm = 1 - cosine_similarity(embeddings)
        np.fill_diagonal(rdm, 0)
        return rdm
    else:
        raise ValueError(f"Invalid distance metric: {distance_metric}")


def compute_rdm_similarity(model_rdm, reference_rdm, similarity_metric='spearman'):
    """
    Compare two representational dissimilarity matrices using a rank or linear
    correlation on their upper-triangular entries.

    Args:
        model_rdm (np.ndarray): RDM derived from model embeddings.
        reference_rdm (np.ndarray): Target RDM to correlate against.
        similarity_metric (str): Correlation statistic to apply. Supports
            ``'spearman'`` (rank) or ``'pearson'`` (linear).

    Returns:
        tuple: ``(rho, p_value)`` as returned by the selected SciPy correlation.

    Raises:
        ValueError: If ``similarity_metric`` is not recognized.
    """
    # Extract upper triangular elements of RDMs (excluding diagonal) for similarity computation
    upper_tri_indices = np.triu_indices_from(reference_rdm, k=1)

    model_rdm_upper_tri_values = model_rdm[upper_tri_indices]
    reference_rdm_upper_tri_values = reference_rdm[upper_tri_indices]

    if similarity_metric == 'spearman':
        rho, p_value = spearmanr(model_rdm_upper_tri_values, reference_rdm_upper_tri_values)
        return rho, p_value
    elif similarity_metric == 'pearson':
        rho, p_value = pearsonr(model_rdm_upper_tri_values, reference_rdm_upper_tri_values)
        return rho, p_value
    else:
        raise ValueError(f"Invalid similarity metric: {similarity_metric}")