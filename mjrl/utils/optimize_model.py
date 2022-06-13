import numpy as np
import copy
import torch
import torch.nn as nn


def fit_data(model, x, y, optimizer, loss_func, batch_size, epochs):
    """
    :param model:           pytorch model of form y_hat = f(x) (class)
    :param x:               inputs to the model (tensor)
    :param y:               desired outputs or targets (tensor)
    :param optimizer:       optimizer to be used (class)
    :param loss_func:       loss criterion (callable)
    :param batch_size:      mini-batch size for optimization (int)
    :param epochs:          number of epochs (int)
    :return:
    """

    num_samples = x.shape[0]
    epoch_losses = []
    for ep in range(epochs):
        rand_idx = torch.LongTensor(np.random.permutation(num_samples))
        ep_loss = 0.0
        num_steps = int(num_samples / batch_size) - 1
        for mb in range(num_steps):
            data_idx = rand_idx[mb*batch_size:(mb+1)*batch_size]
            batch_x = x[data_idx]
            batch_y = y[data_idx]
            optimizer.zero_grad()
            yhat = model(batch_x)
            loss = loss_func(yhat, batch_y)
            loss.backward()
            optimizer.step()
            ep_loss += loss.detach()
        epoch_losses.append(ep_loss.to('cpu').data.numpy().ravel() / num_steps)
    return epoch_losses
