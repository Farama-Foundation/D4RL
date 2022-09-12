"""
Metrics for off-policy evaluation.
"""
import numpy as np

from d4rl import infos

UNDISCOUNTED_POLICY_RETURNS = {
    "halfcheetah-medium": 3985.8150261686337,
    "halfcheetah-random": -199.26067391425954,
    "halfcheetah-expert": 12330.945945279545,
    "hopper-medium": 2260.1983114487352,
    "hopper-random": 1257.9757846810203,
    "hopper-expert": 3624.4696022560997,
    "walker2d-medium": 2760.3310101980005,
    "walker2d-random": 896.4751989935487,
    "walker2d-expert": 4005.89370727539,
}


DISCOUNTED_POLICY_RETURNS = {
    "halfcheetah-medium": 324.83583782709877,
    "halfcheetah-random": -16.836944753939207,
    "halfcheetah-expert": 827.7278887047698,
    "hopper-medium": 235.7441494727478,
    "hopper-random": 215.04955086664955,
    "hopper-expert": 271.6925087260701,
    "walker2d-medium": 202.23983424823822,
    "walker2d-random": 78.46052021427765,
    "walker2d-expert": 396.8752247768766,
}


def get_returns(policy_id, discounted=False):
    if discounted:
        return DISCOUNTED_POLICY_RETURNS[policy_id]
    return UNDISCOUNTED_POLICY_RETURNS[policy_id]


def normalize(policy_id, score):
    key = policy_id + "-v0"
    min_score = infos.REF_MIN_SCORE[key]
    max_score = infos.REF_MAX_SCORE[key]
    return (score - min_score) / (max_score - min_score)


def ranking_correlation_metric(policies, discounted=False):
    """
    Computes Spearman's rank correlation coefficient.
    A score of 1.0 means the policies are ranked correctly according to their values.
    A score of -1.0 means the policies are ranked inversely.

    Args:
        policies: A list of policy string identifiers.
            Valid identifiers must be contained in POLICY_RETURNS.

    Returns:
        A correlation value between [-1, 1]
    """
    return_values = np.array(
        [get_returns(policy_key, discounted=discounted) for policy_key in policies]
    )
    ranks = np.argsort(-return_values)
    N = len(policies)
    diff = ranks - np.arange(N)
    return 1.0 - (6 * np.sum(diff**2)) / (N * (N**2 - 1))


def precision_at_k_metric(policies, k=1, n_rel=None, discounted=False):
    """
    Computes precision@k.

    Args:
        policies: A list of policy string identifiers.
        k (int): Number of top items.
        n_rel (int): Number of relevant items. Default is k.

    Returns:
        Fraction of top k policies in the top n_rel of the true rankings.
    """
    assert len(policies) >= k
    if n_rel is None:
        n_rel = k
    top_k = sorted(
        policies, reverse=True, key=lambda x: get_returns(x, discounted=discounted)
    )[:n_rel]
    policy_k = policies[:k]
    score = sum(policy in top_k for policy in policy_k)
    return float(score) / k


def recall_at_k_metric(policies, k=1, n_rel=None, discounted=False):
    """
    Computes recall@k.

    Args:
        policies: A list of policy string identifiers.
        k (int): Number of top items.
        n_rel (int): Number of relevant items. Default is k.

    Returns:
        Fraction of top n_rel true policy rankings in the top k of the given policies
    """
    assert len(policies) >= k
    if n_rel is None:
        n_rel = k
    top_k = sorted(
        policies, reverse=True, key=lambda x: get_returns(x, discounted=discounted)
    )[:n_rel]
    policy_k = policies[:k]
    score = sum(policy in policy_k for policy in top_k)
    return float(score) / k


def value_error_metric(policy, value, discounted=False):
    """
    Returns the absolute error in estimated value.

    Args:
        policy (str): A policy string identifier.
        value (float): Estimated value
    """
    return abs(
        normalize(policy, value) - normalize(policy, get_returns(policy, discounted))
    )


def policy_regret_metric(policy, expert_policies, discounted=False):
    """
    Returns the regret of the given policy against a set of expert policies.

    Args:
        policy (str): A policy string identifier.
        expert_policies (list[str]): A list of expert policies
    Returns:
        The regret, which is value of the best expert minus the value of the policy.
    """
    best_returns = max(
        get_returns(policy_key, discounted=discounted) for policy_key in expert_policies
    )
    return normalize(policy, best_returns) - normalize(
        policy, get_returns(policy, discounted=discounted)
    )
