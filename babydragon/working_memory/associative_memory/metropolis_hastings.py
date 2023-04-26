import random
from collections import defaultdict

import numba as nb
import numpy as np

from babydragon.working_memory.associative_memory.probability_density_functions import (
    calc_shgo_mode, estimate_pdf, normal)


@nb.jit(nopython=True)
def boltzmann_acceptance_prob(new_score, current_score):
    if new_score <= current_score:
        return np.exp(new_score - current_score)
    else:
        return 1


@nb.jit(nopython=True)
def custom_acceptance_prob(new_score, current_score, alpha):
    epsilon = 1e-8
    if new_score <= current_score:
        return (new_score + epsilon) / (current_score + epsilon) ** alpha
    else:
        return 1


@nb.jit(nopython=True)
def simple_ratio_acceptance_prob(new_score, current_score):
    epsilon = 1e-8
    if new_score <= current_score:
        return (new_score + epsilon) / (current_score + epsilon)
    else:
        return 1


@nb.jit(nopython=False)
def gaussian_acceptance_prob(new_score, current_score, mu, sigma):
    curr_prob = normal(x=current_score, mu=mu, sigma=sigma)
    move_prob = normal(x=new_score, mu=mu, sigma=sigma)
    acceptance = min(move_prob / curr_prob, 1)
    return acceptance


def group_strings_by_index(strings, community_labels):
    groups = defaultdict(list)
    for i, s in enumerate(strings):
        groups[community_labels[i]].append(s)
    return groups


def metropolis_hastings(
    graph, num_communities, num_iterations, acceptance_metric="custom", custom_alpha=0.5
):
    """
    Implement the Metropolis-Hastings sampling-based community detection.

    Parameters:
    graph (numpy array): The producer-producer similarity graph.
    num_communities (int): The number of communities to be detected.
    num_iterations (int): The number of iterations for the Metropolis-Hastings algorithm.
    acceptance_metric (str): The acceptance probability metric ('boltzmann', 'custom', 'simple_ratio', or 'gaussian_mcmc').
    custom_alpha (float): Custom acceptance probability parameter (only used when acceptance_metric='custom').

    Returns:
    community_labels (list): A list containing the community affiliation for each producer.
    """

    num_producers = graph.shape[0]

    # Initialize the community labels randomly
    community_labels = [
        random.randint(0, num_communities - 1) for _ in range(num_producers)
    ]

    # Calculate scores for the graph to estimate the Gaussian parameters (mu and sigma)
    scores = [
        sum(
            graph[i, j]
            for j in range(num_producers)
            if community_labels[j] == community_labels[i]
        )
        for i in range(num_producers)
    ]
    if acceptance_metric == "gaussian_mcmc":
        distribution = estimate_pdf(scores)
        mu = calc_shgo_mode(scores, distribution)[0]
        sigma = np.std(scores)

    for _ in range(num_iterations):
        for producer in range(num_producers):
            # Calculate the current community score
            current_community = community_labels[producer]
            current_score = sum(
                graph[producer, j]
                for j in range(num_producers)
                if community_labels[j] == current_community
            )

            # Choose a new community for the producer
            new_community = random.randint(0, num_communities - 1)

            # Calculate the score for the new community
            new_score = sum(
                graph[producer, j]
                for j in range(num_producers)
                if community_labels[j] == new_community
            )

            # Calculate the acceptance probability
            if acceptance_metric == "boltzmann":
                acceptance_prob = boltzmann_acceptance_prob(new_score, current_score)
            elif acceptance_metric == "custom":
                acceptance_prob = custom_acceptance_prob(
                    new_score, current_score, custom_alpha
                )
            elif acceptance_metric == "simple_ratio":
                acceptance_prob = simple_ratio_acceptance_prob(new_score, current_score)
            elif acceptance_metric == "gaussian_mcmc":
                acceptance_prob = gaussian_acceptance_prob(
                    new_score, current_score, mu, sigma
                )
            else:
                raise ValueError(
                    "Invalid acceptance_metric value. Acceptable values are 'boltzmann', 'custom', 'simple_ratio', or 'gaussian_mcmc'."
                )

            # Accept or reject the new community
            if random.random() < acceptance_prob:
                community_labels[producer] = new_community

    return community_labels
