import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from babydragon.working_memory.associative_memory.metropolis_hastings import \
    metropolis_hastings


def nmi_similarity(community_labels1, community_labels2):
    """
    Calculate the Normalized Mutual Information (NMI) between two sets of community labels.

    Args:
    community_labels1 (list): The first set of community labels.
    community_labels2 (list): The second set of community labels.

    Returns:
    float: The NMI similarity score.
    """
    return normalized_mutual_info_score(community_labels1, community_labels2)


def nmi_stability_analysis(kernel, num_runs, num_communities, num_iterations=100):
    """
    Perform stability analysis on the kernel-based community detection algorithm using NMI.

    Args:
    kernel (function): The kernel function.
    num_runs (int): The number of times to run the community detection algorithm.
    num_communities (int): The number of communities.
    num_iterations (int): The number of iterations for the Metropolis-Hastings algorithm.

    Returns:
    float: The average NMI similarity score.
    """
    community_label_sets = [
        metropolis_hastings(
            kernel, num_communities=num_communities, num_iterations=num_iterations
        )
        for _ in range(num_runs)
    ]

    nmi_similarities = [
        nmi_similarity(community_label_sets[i], community_label_sets[j])
        for i in range(num_runs)
        for j in range(i + 1, num_runs)
    ]

    return np.mean(nmi_similarities)


def run_stability_analysis(
    kernels, kernel_labels, optimal_num_communities, num_runs=10
):
    kernel_stability_nmi = {
        label: nmi_stability_analysis(
            kernel, num_runs, optimal_communities, num_iterations=100
        )
        for label, kernel, optimal_communities in zip(
            kernel_labels, kernels, optimal_num_communities.values()
        )
    }

    # Print the stability results for each kernel using NMI
    out_dict = {}
    for kernel, stability in kernel_stability_nmi.items():
        out_dict[kernel] = stability
    return out_dict
