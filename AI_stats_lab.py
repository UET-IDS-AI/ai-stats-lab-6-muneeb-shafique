import math
import numpy as np


def bernoulli_log_likelihood(data, theta):
    data = np.asarray(data)

    if len(data) == 0:
        raise ValueError("Data must be non-empty.")
    if not (0 < theta < 1):
        raise ValueError("theta must satisfy 0 < theta < 1.")
    if not np.all((data == 0) | (data == 1)):
        raise ValueError("Data must contain only 0s and 1s.")

    return np.sum(data * np.log(theta) + (1 - data) * np.log(1 - theta))


def bernoulli_mle_with_comparison(data, candidate_thetas=None):
    data = np.asarray(data)

    if len(data) == 0:
        raise ValueError("Data must be non-empty.")
    if not np.all((data == 0) | (data == 1)):
        raise ValueError("Data must contain only 0s and 1s.")

    if candidate_thetas is None:
        candidate_thetas = [0.2, 0.5, 0.8]

    mle = float(np.mean(data))
    num_successes = int(np.sum(data == 1))
    num_failures = int(np.sum(data == 0))

    log_likelihoods = {}
    for theta in candidate_thetas:
        try:
            log_likelihoods[theta] = bernoulli_log_likelihood(data, theta)
        except ValueError:
            log_likelihoods[theta] = float('-inf')

    best_candidate = max(candidate_thetas, key=lambda t: log_likelihoods[t])

    return {
        "mle": mle,
        "num_successes": num_successes,
        "num_failures": num_failures,
        "log_likelihoods": log_likelihoods,
        "best_candidate": best_candidate,
    }


def poisson_log_likelihood(data, lam):
    data = np.asarray(data)

    if len(data) == 0:
        raise ValueError("Data must be non-empty.")
    if lam <= 0:
        raise ValueError("lam must be > 0.")
    if not np.all(data >= 0):
        raise ValueError("Data must contain nonnegative values.")
    if not np.all(data == np.floor(data)):
        raise ValueError("Data must contain integer values.")

    return float(np.sum(data * np.log(lam) - lam - np.array([math.lgamma(x + 1) for x in data])))


def poisson_mle_analysis(data, candidate_lambdas=None):
    data = np.asarray(data)

    if len(data) == 0:
        raise ValueError("Data must be non-empty.")
    if not np.all(data >= 0):
        raise ValueError("Data must contain nonnegative values.")
    if not np.all(data == np.floor(data)):
        raise ValueError("Data must contain integer values.")

    if candidate_lambdas is None:
        candidate_lambdas = [1.0, 3.0, 5.0]

    mle = float(np.mean(data))
    sample_mean = mle
    total_count = int(np.sum(data))
    n = len(data)

    log_likelihoods = {lam: poisson_log_likelihood(data, lam) for lam in candidate_lambdas}
    best_candidate = max(candidate_lambdas, key=lambda l: log_likelihoods[l])

    return {
        "mle": mle,
        "sample_mean": sample_mean,
        "total_count": total_count,
        "n": n,
        "log_likelihoods": log_likelihoods,
        "best_candidate": best_candidate,
    }