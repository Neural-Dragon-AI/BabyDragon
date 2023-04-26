import numpy as np
import scipy.optimize
import scipy.stats


def normal(x, mu, sigma):
    numerator = np.exp((-((x - mu) ** 2)) / (2 * sigma**2))
    denominator = sigma * np.sqrt(2 * np.pi)
    return numerator / denominator


def calc_shgo_mode(scores, distribution):
    """
    Calculates the mode of a distribution using the SHGO optimization method.
    :scores: list of distance scores
    :distribution: probability density function estimated from the scores
    :return: mode of the distribution
    """

    def objective(x):
        return -distribution(x)

    bounds = [(min(scores), max(scores))]
    result = scipy.optimize.shgo(objective, bounds)
    return result.x


def estimate_pdf(scores):
    """
    estimate scores probability density function
    :scores: list of distance scores from topic features to topic centroid
    :return: mean and standard deviation of the estimated distribution
    """
    pdf = scipy.stats.gaussian_kde(scores)
    return pdf
