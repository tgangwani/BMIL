import errno
import os
import logging
import time
import pickle
import numpy as np
import torch
import torch.nn as nn

def activation(nonlinearity):
    if nonlinearity == 'tanh':
        return torch.nn.Tanh, nn.init.calculate_gain('tanh')
    elif nonlinearity == 'relu':
        return torch.nn.ReLU, nn.init.calculate_gain('relu')
    elif nonlinearity == 'leaky_relu':
        return torch.nn.LeakyReLU, nn.init.calculate_gain('leaky_relu')
    else: raise NotImplementedError

def l2_loss_criterion(x, y):
    assert x.dim() == y.dim() and x.size(0) == y.size(0)
    return (x-y).pow(2).mean(-1)

class RunningMeanStd(object):
    """
    https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count        
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / (count + batch_count)
    new_var = M2 / (count + batch_count)
    new_count = batch_count + count
    
    return new_mean, new_var, new_count

def print_header():
    logging.info('                          Progr | FPS | ToL | avg | med | min | max | len || Losses: | PoG | Dsc | Val | Ent | On. | Off | --- |')
    logging.info('                          ------|-----|-----|-----|-----|-----|-----|-----||---------|-----|-----|-----|-----|-----|-----|-----|')

def log_and_print(j, num_updates, T, final_rewards, final_lens, tracking,
        num_ended_episodes, total_loss, value_loss, pg_loss, dist_entropy,
        discriminator_loss, algorithm, _run):
    """
    Logs values to Observer and outputs the some numbers to command line.

    Args:
        j: Current gradient update
        num_updates: Total number of gradient updates to be performed
        T: total time passed
        final_rewards: Total return on last completed episode
        final_lens: Length on last completed episode
        tracking: `tracked_values` from function `track_values`
        num_ended_episodes: How many episodes have ended
        algorithm: Config dict (see default.yaml for contents)
        _run: `Run` object from sacred. Needed to send stuff to the observer.
    """

    total_num_steps = (j + 1) * algorithm['num_processes'] * algorithm['num_steps']
    fps = int(total_num_steps / T)

    num_frames = j * algorithm['num_steps'] * algorithm['num_processes']

    # Log scalars
    _run.log_scalar("result.mean", final_rewards.mean().item(), num_frames)
    _run.log_scalar("result.median", final_rewards.median().item(), num_frames)
    _run.log_scalar("result.min", final_rewards.min().item(), num_frames)
    _run.log_scalar("result.max", final_rewards.max().item(), num_frames)
    _run.log_scalar("result.len", final_lens.mean().item(), num_frames)

    _run.log_scalar("episodes.num_ended", num_ended_episodes.item(), num_frames)
    _run.log_scalar("obs.fps", fps, num_frames)

    _run.log_scalar("loss.total", total_loss.item(), num_frames)
    _run.log_scalar("loss.value", value_loss.item(), num_frames)
    _run.log_scalar("loss.pg", pg_loss.item(), num_frames)
    _run.log_scalar("loss.entropy", dist_entropy.item(), num_frames)
    _run.log_scalar("loss.discriminator", discriminator_loss.item(), num_frames)

    reg_loss_scalar_onPol = reg_loss_scalar_offPol = "-----"

    # Forward-, inverse- and action-regularization loss during on-policy training
    if 'reg_loss_scalar_onPol' in tracking.keys():
        reg_loss_scalar_onPol = np.mean(tracking['reg_loss_scalar_onPol'])
    # Forward-, inverse- and action-regularization loss during off-policy training (includes multi-step)
    if 'reg_loss_scalar_offPol' in tracking.keys():
        reg_loss_scalar_offPol = np.mean(tracking['reg_loss_scalar_offPol'])

    logging.info('({}) Updt: {:5} |{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}||         |{:5}|{:5}|{:5}|{:5}|{:5}|{:5}|{:5}'.format(
        str(time.strftime('%x %X %z'))[:-6],
        str(j / num_updates)[:5],
        str(fps),
        str(total_loss.item())[:5],
        str(final_rewards.mean().item())[:5],
        str(final_rewards.median().item())[:5],
        str(final_rewards.min().item())[:5],
        str(final_rewards.max().item())[:5],
        str(final_lens.mean().item())[:5],
        str(pg_loss.item())[:5],
        str(discriminator_loss.item())[:5],
        str(value_loss.item())[:5],
        str(dist_entropy.item())[:5],
        str(reg_loss_scalar_onPol)[:5],
        str(reg_loss_scalar_offPol)[:5],
        str("-----")[:5]))

def safe_make_dirs(path):
    """
    Given a path, makes a directory. Doesn't make directory if it already exists. Treats possible
    race conditions safely.
    http://stackoverflow.com/questions/273192/how-to-check-if-a-directory-exists-and-create-it-if-necessary
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
