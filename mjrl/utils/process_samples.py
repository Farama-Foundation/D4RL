import numpy as np

def compute_returns(paths, gamma):
    for path in paths:
        path["returns"] = discount_sum(path["rewards"], gamma)

def compute_advantages(paths, baseline, gamma, gae_lambda=None, normalize=False):
    # compute and store returns, advantages, and baseline 
    # standard mode
    if gae_lambda == None or gae_lambda < 0.0 or gae_lambda > 1.0:
        for path in paths:
            path["baseline"] = baseline.predict(path)
            path["advantages"] = path["returns"] - path["baseline"]
        if normalize:
            alladv = np.concatenate([path["advantages"] for path in paths])
            mean_adv = alladv.mean()
            std_adv = alladv.std()
            for path in paths:
                path["advantages"] = (path["advantages"]-mean_adv)/(std_adv+1e-8)
    # GAE mode
    else:
        for path in paths:
            b = path["baseline"] = baseline.predict(path)
            if b.ndim == 1:
                b1 = np.append(path["baseline"], 0.0 if path["terminated"] else b[-1])
            else:
                b1 = np.vstack((b, np.zeros(b.shape[1]) if path["terminated"] else b[-1]))
            td_deltas = path["rewards"] + gamma*b1[1:] - b1[:-1]
            path["advantages"] = discount_sum(td_deltas, gamma*gae_lambda)
        if normalize:
            alladv = np.concatenate([path["advantages"] for path in paths])
            mean_adv = alladv.mean()
            std_adv = alladv.std()
            for path in paths:
                path["advantages"] = (path["advantages"]-mean_adv)/(std_adv+1e-8)

def discount_sum(x, gamma, terminal=0.0):
    y = []
    run_sum = terminal
    for t in range( len(x)-1, -1, -1):
        run_sum = x[t] + gamma*run_sum
        y.append(run_sum)

    return np.array(y[::-1])