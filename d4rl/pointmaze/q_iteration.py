"""
Use q-iteration to solve for an optimal policy

Usage: q_iteration(env, gamma=discount factor, ent_wt= entropy bonus)
"""
import numpy as np
from scipy.special import logsumexp as sp_lse

def softmax(q, alpha=1.0):
    q = (1.0/alpha)*q
    q = q-np.max(q)
    probs = np.exp(q)
    probs = probs/np.sum(probs)
    return probs

def logsumexp(q, alpha=1.0, axis=1):
    if alpha == 0:
        return np.max(q, axis=axis)
    return alpha*sp_lse((1.0/alpha)*q, axis=axis)


def get_policy(q_fn, ent_wt=1.0):
    v_rew = logsumexp(q_fn, alpha=ent_wt)
    adv_rew = q_fn - np.expand_dims(v_rew, axis=1)
    if ent_wt == 0:
        pol_probs = adv_rew
        pol_probs[pol_probs >= 0 ] = 1.0
        pol_probs[pol_probs < 0 ] = 0.0
    else:
        pol_probs = np.exp((1.0/ent_wt)*adv_rew)
    pol_probs /= np.sum(pol_probs, axis=1, keepdims=True)
    assert np.all(np.isclose(np.sum(pol_probs, axis=1), 1.0)), str(pol_probs)
    return pol_probs


def softq_iteration(env, transition_matrix=None, reward_matrix=None, num_itrs=50, discount=0.99, ent_wt=0.1, warmstart_q=None, policy=None):
    """
    Perform tabular soft Q-iteration
    """
    dim_obs = env.num_states
    dim_act = env.num_actions
    if reward_matrix is None:
        reward_matrix = env.reward_matrix()
    reward_matrix = reward_matrix[:,:,0]

    if warmstart_q is None:
        q_fn = np.zeros((dim_obs, dim_act))
    else:
        q_fn = warmstart_q

    if transition_matrix is None:
        t_matrix = env.transition_matrix()
    else:
        t_matrix = transition_matrix

    for k in range(num_itrs):
        if policy is None:
            v_fn = logsumexp(q_fn, alpha=ent_wt)
        else:
            v_fn = np.sum((q_fn - ent_wt*np.log(policy))*policy, axis=1)
        new_q = reward_matrix + discount*t_matrix.dot(v_fn)
        q_fn = new_q
    return q_fn


def q_iteration(env, **kwargs):
    return softq_iteration(env, ent_wt=0.0, **kwargs)


def compute_visitation(env, q_fn, ent_wt=1.0, env_time_limit=50, discount=1.0):
  pol_probs = get_policy(q_fn, ent_wt=ent_wt)

  dim_obs = env.num_states
  dim_act = env.num_actions
  state_visitation = np.zeros((dim_obs, 1))
  for (state, prob) in env.initial_state_distribution.items():
    state_visitation[state] = prob
  t_matrix = env.transition_matrix()  # S x A x S
  sa_visit_t = np.zeros((dim_obs, dim_act, env_time_limit))

  for i in range(env_time_limit):
    sa_visit = state_visitation * pol_probs
    # sa_visit_t[:, :, i] = (discount ** i) * sa_visit
    sa_visit_t[:, :, i] = sa_visit
    # sum-out (SA)S
    new_state_visitation = np.einsum('ij,ijk->k', sa_visit, t_matrix)
    state_visitation = np.expand_dims(new_state_visitation, axis=1)
  return np.sum(sa_visit_t, axis=2) / float(env_time_limit)


def compute_occupancy(env, q_fn, ent_wt=1.0, env_time_limit=50, discount=1.0):
  pol_probs = get_policy(q_fn, ent_wt=ent_wt)

  dim_obs = env.num_states
  dim_act = env.num_actions
  state_visitation = np.zeros((dim_obs, 1))
  for (state, prob) in env.initial_state_distribution.items():
    state_visitation[state] = prob
  t_matrix = env.transition_matrix()  # S x A x S
  sa_visit_t = np.zeros((dim_obs, dim_act, env_time_limit))

  for i in range(env_time_limit):
    sa_visit = state_visitation * pol_probs
    sa_visit_t[:, :, i] = (discount ** i) * sa_visit
    # sa_visit_t[:, :, i] = sa_visit
    # sum-out (SA)S
    new_state_visitation = np.einsum('ij,ijk->k', sa_visit, t_matrix)
    state_visitation = np.expand_dims(new_state_visitation, axis=1)
  return np.sum(sa_visit_t, axis=2) #/ float(env_time_limit)
