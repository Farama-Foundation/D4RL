"""
Metrics for off-policy evaluation.
"""

POLICY_RETURNS = {
    'halfcheetah-medium' : 3985.8150261686337,
    'halfcheetah-random' : -199.26067391425954,
    'halfcheetah-expert' : 12330.945945279545,
    'hopper-medium' : 2260.1983114487352,
    'hopper-random' : 1257.9757846810203,
    'hopper-expert' : 3624.4696022560997,
    'walker2d-medium' : 2760.3310101980005,
    'walker2d-random' : 896.4751989935487,
    'walker2d-expert' : .005.89370727539,
}

def ranking_metric(policies):
    """
    Returns true if policies given are ranked properly, from
    highest to lowest returns.

    Args:
        policies: A list of policy string identifiers.
            Valid identifiers must be contained in POLICY_RETURNS.

    Returns:
        true if the returns of policies are monotonically decreasing.
    """
    return_values = [POLICY_RETURNS[policy_key] for policy_key in policies]
    return all(x>y for x, y in zip(return_values, return_values[1:]))


def return_mse_metric(policy, returns):
    """
    Returns the squared error in estimated returns.

    Args:
        policy (str): A policy string identifier.
        returns (float): Estimated returns
    """
    return (returns - POLICY_RETURNS[policy])**2


def policy_regret_metric(policy, expert_policies):
    """
    Returns the regret of the given policy against a set of expert policies.

    Args:
        policy (str): A policy string identifier.
        expert_policies (list[str]): A list of expert policies
    Returns:
        The regret, which is returns of the best expert minus the returns of the policy.
    """
    best_returns = max([POLICY_RETURNS[policy_key] for policy_key in policies])
    return best_returns - POLICY_RETURNS[best_guess_policy]
