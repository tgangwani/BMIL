"""
Code for the paper: Learning Belief Representations for Imitation Learning in POMDPs 
[Acknowledgement:] Code based off: https://github.com/maximilianigl/DVRL

High level setup:
Sacred (https://sacred.readthedocs.io/en/latest/) is used to configure, run, save the experiments.
Please read the docs. In short: @ex.config changes configuration of experiment, @ex.command and @ex.capture
makes configuration available, and @ex.automain is the entry point of the program.

Configuration happens in 4 places:
- code/conf/mujocoEnv.yaml (environment-independent parameters)
- environment.config_file: (environment-dependent parameters; path provided via command line)
- The command line (overrides everything when specified)
- The general_config() function below updates some values
"""

import sys
import os
import time
import logging
import collections
import itertools
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gym.envs.registration import register
from envs import make_env, make_expert_envs
from storage import RolloutStorage
from experience_replay import ExpReplay
from utils.vec_env.dummy_vec_env import DummyVecEnv
from utils.vec_env.subproc_vec_env import SubprocVecEnv
from utils import helpers

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

# Create Sacred Experiment
ex = Experiment("BMIL")
ex.captured_out_filter = apply_backspaces_and_linefeeds

# Add yaml files to the sacred configuration
DIR = os.path.dirname(sys.argv[0])
DIR = '.' if DIR == '' else DIR
ex.add_config(DIR + '/conf/mujocoEnv.yaml')

from sacred.observers import FileStorageObserver
ex.observers.append(FileStorageObserver.create('saved_runs'))

@ex.config
def general_config(cuda, algorithm, environment, model_setting):
    """
    This function is called by sacred before the experiment is started
    All args are provided by sacred and filled with configuration values
    """

    if cuda == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cuda

    # Load information from the environment configuration yaml file
    with open(environment['config_path'], 'r') as f:
        env_params = yaml.load(f)

    try:
        env_parameters = list(filter(lambda x, environment=environment: x['env'] == environment['name'], env_params['tasks']))[0]
    except:
        raise ValueError('Environment not found in yaml config: %s'%(environment.config_path))

    # Task-specific params are contained in 'env_parameters'
    environment['expert_db'] = env_parameters['expert_db']
    environment['occlusion'] = env_parameters['occlusion']
    algorithm['init_logstd'] = env_parameters['init_logstd']

    # Delete keys so we don't have them in the sacred configuration
    del env_params
    del env_parameters

    if algorithm['belief_loss_type'] == 'task_agnostic':
        model_setting['detach_belief_module'] = True
    else:
        assert algorithm['belief_loss_type'] == 'task_aware', "Unknown belief_loss_type. Options {task_aware,task_agnostic}."

@ex.command(unobserved=True)
def setup(model_setting, algorithm, device, _run, _log, log, seed, cuda):
    """
    All args are automatically provided by sacred
    Some of the important objects created in this function are:
        - parallel environments (using SubprocVecEnv from OpenAI baselines)
        - instance of model (BMIL)
        - experience replay
        - RolloutStorage: a helper class to save rewards and compute the advantage loss
    """

    # Create working dir
    id_tmp_dir = "{}/{}/".format(log['tmp_dir'], _run._id)
    helpers.safe_make_dirs(id_tmp_dir)

    np.set_printoptions(precision=2)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    logger = logging.getLogger()
    if _run.debug or _run.pdb:
        logger.setLevel(logging.DEBUG)

    envs = register_and_create_envs(id_tmp_dir)
    model = create_model(envs)

    # Experience replay buffer to store off-policy data.
    replay = ExpReplay(batch_size=algorithm['num_processes_offPol'], max_trajs=1000, fwd_jump=algorithm['forward_jump'], bwd_jump=algorithm['backward_jump'])

    rollouts = RolloutStorage(algorithm['num_steps'], algorithm['num_processes'])
    rollouts.to(device)

    # Reset all environments
    obs = envs.reset()
    curr_ob = torch.from_numpy(obs).float()

    init_state = torch.zeros(algorithm['num_processes'], model_setting['belief_dim']).to(device)
    init_state_offPol = torch.zeros(algorithm['num_processes_offPol'], model_setting['belief_dim']).to(device)
    init_episode_reward_info = torch.zeros([algorithm['num_processes'], 1])
    init_ac = torch.zeros(algorithm['num_processes'], envs.action_space.shape[0]).to(device)

    # Buffer to hold information along the current "on-policy" path.
    curr_memory = {
        'curr_ob': curr_ob,    # o_t
        'prev_belief': init_state,   # b_{t-1}
        'prev_ac': init_ac,  # a_{t-1}
        'prev_ob': curr_ob.clone(), # o_{t-1}
        'expert_ac': init_ac.clone(),
        'episode_reward_info' : init_episode_reward_info
    }

    # Buffer to hold information along the current "off-policy" path.
    curr_memory_offPol = {
            'curr_ob': None,
            'prev_ob': None,
            'prev_belief': init_state_offPol,
            'prev_ac': None,
            'ob_tpk': None,   # o_{t+k}
            'ob_tmkm1': None,  # o_{t-k-1}
            'future_k_acs': None,  # a_t:a_{t+k-1}
            'past_k_acs': None, # a_{t-k-1}:a_{t-2}
            'future_mask': None,  # mask for o_{t+k}
            'past_mask': None  # mask for o_{t-k-1}
    }

    return envs, model, rollouts, curr_memory, curr_memory_offPol, replay

@ex.command
def create_model(envs, algorithm, model_setting, device):
    """
    Args:
        envs: Vector of environments, creeated by register_and_create_envs()
        All other args are automatically provided by sacred
    """

    from model import BMIL
    x = BMIL(
        device,
        envs.action_space,
        envs.observation_space.shape[0],
        **algorithm,
        **model_setting)

    return x.to(device)

@ex.capture
def register_and_create_envs(id_tmp_dir, seed, environment, algorithm):
    """
    Args:
        id_temp_dir (str): Working directory.
        All other args are automatically provided by sacred
    """

    if environment['entry_point']:
        try:
            register(
                id=environment['name'],
                entry_point=environment['entry_point'],
                kwargs=environment['config'],
                max_episode_steps=environment['max_episode_steps']
            )
        except Exception:
            pass

    num_envs = algorithm['num_processes']
    num_expert_envs = algorithm['num_expert_processes']

    envs = [make_env(environment['name'], seed, i, id_tmp_dir,
                      occlusion=list(environment['occlusion']), sensor_noise=float(environment['sensor_noise']))
            for i in range(num_envs-num_expert_envs)]

    # Create "expert environments" which only replay trajectories from an expert database, rather than interacting with Gym.
    # See expert_envs.py for the exposed API. These environments are appended to the list of the Gym environments.
    if num_expert_envs:
        e_envs = [make_expert_envs(environment['name'], seed, i, num_expert_envs, environment['expert_db'])
                for i in range(num_expert_envs)]
        envs.extend(e_envs)

    # Vectorise envs
    if algorithm['num_processes'] > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    return envs

def run_model_offPol(model, curr_memory_offPol):
    """
    Run model (1-step) with "off-policy" data
    """
    model_return = model.forward_offPol(curr_memory_offPol=curr_memory_offPol)
    return model_return

@ex.capture
def run_model(model, curr_memory, envs, device):
    """
    Run model (1-step) with "on-policy" data
    Args 'device' is provided by sacred
    """

    # Fetch the action taken by the expert in the current observation (from database). For interactive Gym environments, no-op (0) is returned
    curr_memory['expert_ac'] = torch.from_numpy(envs.expert_ac())

    #forward()
    model_return = model(curr_memory=curr_memory)

    # Execute on environment. Environmental reward unavailable
    cpu_acs = model_return.action.detach().squeeze(1).cpu().numpy()
    obs, _, done, info = envs.step(cpu_acs)

    # Use dictionary created by bench.monitor to obtain true episodic returns (for printing and plotting only)
    if np.sum(done) > 0:
        for idx, done_ in enumerate(done.tolist()):
            if done_:
                # Expert envs. do not insert this info. at episode termination. Add a dummy (-1e3) value for them
                curr_memory['episode_reward_info'][idx] = info[idx].get('episode')['r'] if 'episode' in info[idx].keys() else -1e3

    # Get "synthetic reward" from the discriminator
    reward = - torch.log(1 - model_return.discriminator_out_d.detach() + 1e-3)

    # If trajectory ended, create mask to reset previous belief and previous action
    mask = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

    # Update curr_memory
    curr_memory['prev_ob'] = curr_memory['curr_ob']
    curr_memory['curr_ob'] = torch.from_numpy(obs).float()

    # For start of a new episode, make the prev_ob same as the curr_ob
    curr_memory['prev_ob'] = mask * curr_memory['prev_ob'] + (1 - mask) * curr_memory['curr_ob']
    mask = mask.to(device)

    # Resets for new episodes
    curr_memory['prev_belief'] = model_return.belief_state * mask
    curr_memory['prev_ac'] = model_return.action * mask

    return model_return, curr_memory, mask, reward

@ex.capture
def track_values(tracked_values, model_return):
    """
    Stores information from model_return for computation of loss and gradients
    """

    tracked_values['values'].append(model_return.value_estimate)
    tracked_values['ac_log_probs'].append(model_return.ac_log_probs)
    tracked_values['dist_entropy'].append(model_return.dist_entropy)
    tracked_values['reg_loss'].append(model_return.reg_loss)
    tracked_values['reg_loss_scalar_onPol'].append(model_return.reg_loss_item) # used only for printing
    tracked_values['discriminator_sigmoids_d'].append(model_return.discriminator_out_d)
    tracked_values['discriminator_sigmoids_b'].append(model_return.discriminator_out_b)
    return tracked_values

def update_paths(batch_size, ob, prev_ob, prev_ac, mask, running_paths, replay):
    """
    Update "running_paths" with the new environmental step
    If any of the paths is complete, add it to the experience replay

    ob: o_{t}, prev_ob: o_{t-1}, prev_ac: a_{t-1}
    batch_size: number of agent environments
    """

    # Move all tensors to CPU, detach and convert to numpy arrays
    ob, prev_ob, prev_ac, done = list(map(lambda x: list(x.cpu().detach().numpy())[:batch_size],
        [ob, prev_ob, prev_ac, torch.ones_like(mask) - mask]))

    # Iterate over the single-timestep batch of agent environments
    for idx, ob_, prev_ob_, prev_ac_, done_ in zip(itertools.count(), ob, prev_ob, prev_ac, done):

        if running_paths[idx] is None:
            running_paths[idx] = dict(
                ob=[],
                prev_ob=[],
                prev_ac=[],
            )
            assert np.all(ob_ == prev_ob_)

        running_paths[idx]["ob"].append(ob_)
        running_paths[idx]["prev_ob"].append(prev_ob_)
        running_paths[idx]["prev_ac"].append(prev_ac_)
        if done_:
            # Add completed episode to experience replay
            replay.add_path(dict(
                        obs=np.array(running_paths[idx]["ob"]),
                        prev_obs=np.array(running_paths[idx]["prev_ob"]),
                        prev_acs=np.array(running_paths[idx]["prev_ac"])
                    ))
            running_paths[idx] = None

@ex.capture
def track_rewards(tracked_rewards, mask, curr_memory, algorithm):
    mask = mask.cpu()

    # Track episode rewards and lengths
    tracked_rewards['episode_lens'] += 1
    tracked_rewards['num_ended_episodes'] += algorithm['num_processes'] - sum(mask)[0]
    tracked_rewards['final_rewards'] *= mask
    tracked_rewards['final_rewards'] += (1 - mask) * curr_memory['episode_reward_info']
    tracked_rewards['final_lens'] *= mask
    tracked_rewards['final_lens'] += (1 - mask) * tracked_rewards['episode_lens']
    tracked_rewards['episode_lens'] *= mask

    return tracked_rewards['final_rewards'], tracked_rewards['final_lens'], tracked_rewards['num_ended_episodes']

@ex.automain
def main(_run,
         opt,
         device,
         log,
         algorithm,
         loss_function):
    """
    Entry point. Contains main training loop.
    """

    # Setup directory, vector of environments, instance of model (BMIL), a 'rollouts' helper class to compute target values,
    # experience replay, and curr_memory (curr_memory_offPol) buffers to hold information along the running "on-policy" ("off-policy") path.
    envs, model, rollouts, curr_memory, curr_memory_offPol, replay = setup()

    tracked_rewards = {
        'final_rewards': torch.zeros([algorithm['num_processes'], 1]),
        'episode_lens': torch.zeros([algorithm['num_processes'], 1]),
        'final_lens': torch.zeros([algorithm['num_processes'], 1]),
        'num_ended_episodes': 0
    }

    num_updates = int(float(algorithm['num_timesteps'])
                      // algorithm['num_steps']
                      // algorithm['num_processes'])

    # Configurations.
    expert_bs = algorithm['num_expert_processes']
    agent_bs = algorithm['num_processes'] - algorithm['num_expert_processes']  # num. of agent (Gym.) environments
    center_advantage = algorithm['center_advantage']
    num_steps = algorithm['num_steps']
    fwd_jump = algorithm['forward_jump']
    bwd_jump = algorithm['backward_jump']

    # Disable experience replay if belief-reg=False in the Task-aware setting.
    disable_replay = (algorithm['belief_loss_type'] == 'task_aware' and algorithm['belief_regularization'] == False)

    # Count parameters
    num_parameters = 0
    for p in model.parameters():
        num_parameters += p.nelement()

    # Create optimisers. We have 2 optimizers:
    # 1. bp_optimizer: for parameters of the belief module and policy
    # 2. d_optimizer: for discriminator parameters

    d_params = [p for n, p in model.named_parameters() if n.__contains__("discriminator")]
    bp_params = [p for n, p in model.named_parameters() if not n.__contains__("discriminator")]

    if opt['optimizer'] == 'RMSProp':
        bp_optimizer = optim.RMSprop(bp_params, opt['lr'], eps=opt['eps'], alpha=opt['alpha'])
        d_optimizer = optim.RMSprop(d_params, opt['lr'], eps=opt['eps'], alpha=opt['alpha'])
    elif opt['optimizer'] == 'Adam':
        bp_optimizer = optim.Adam(bp_params, opt['lr'], eps=opt['eps'], betas=opt['betas'])
        d_optimizer = optim.Adam(d_params, opt['lr'], eps=opt['eps'], betas=opt['betas'])

    adversarial_criterion = torch.nn.BCELoss(reduction='none')

    print(model); sys.stdout.flush() 
    logging.info('Number of parameters =\t{}'.format(num_parameters))
    logging.info("Total number of updates: {}".format(num_updates))
    logging.info("Learning rate: {}".format(opt['lr']))
    helpers.print_header()

    num_offPol_updates = 0
    running_paths = [None]*agent_bs
    start = time.time()

    # == Main training loop ==
    for j in range(num_updates):

        # Learning-rate schedule.
        if opt['lr_schedule'] == 'linear_decay':
            cur_lrmult = max(1.0 - float(j) / num_updates, 0)
            for param_group in bp_optimizer.param_groups:
                param_group['lr'] = cur_lrmult * opt['lr']
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = cur_lrmult * opt['lr']

        # Loop over "num_steps" timesteps for one A2C gradient update
        # ====
        # On-policy learning
        # ====

        tracked_values = collections.defaultdict(lambda: [])
        for step in range(num_steps):

            ob_t = curr_memory['curr_ob']    # o_{t}
            ob_tm1 = curr_memory['prev_ob']   # o_{t-1}
            ac_tm1 = curr_memory['prev_ac']    # a_{t-1}

            model_return, curr_memory, mask, reward = run_model(
                model=model,
                curr_memory=curr_memory,
                envs=envs)

            if not disable_replay:
                update_paths(agent_bs, ob_t, ob_tm1, ac_tm1, mask, running_paths, replay)

            # Save in rollouts (for loss computation)
            rollouts.insert(step, reward, mask, model_return.value_estimate.detach())

            # Accumulate data from model_return
            tracked_values = track_values(tracked_values, model_return)

            final_rewards, final_lens, num_ended_episodes = track_rewards(
                tracked_rewards, mask, curr_memory)

        # Compute bootstrapped critic value
        with torch.no_grad():
            model_return = model(curr_memory=curr_memory)
            next_value = model_return.value_estimate
            rollouts.compute_vtarg_and_adv(next_value, algorithm['gamma'], algorithm['lam'])

        # Stack into (num_steps, num_processes, ...)
        values = torch.stack(tuple(tracked_values['values']), dim=0)
        ac_log_probs = torch.stack(tuple(tracked_values['ac_log_probs']), dim=0)
        reg_loss = torch.stack(tuple(tracked_values['reg_loss']), dim=0)
        dist_entropy = torch.stack(tuple(tracked_values['dist_entropy']), dim=0)
        discriminator_sigmoids_d = torch.stack(tuple(tracked_values['discriminator_sigmoids_d']), dim=0)
        discriminator_sigmoids_b = torch.stack(tuple(tracked_values['discriminator_sigmoids_b']), dim=0)

        # Value target TD (lambda)
        v_target = rollouts.tdlamret[1:].detach()

        # Advantage calculation
        advantages = rollouts.advantages[1:].detach()
        if center_advantage:
            advantages = (advantages - advantages[:, :agent_bs, ...].mean()) / (advantages[:, :agent_bs, ...].std() + 1e-6)

        # "Soft" targets for adversarial loss
        discriminator_targets = torch.cat([torch.empty(num_steps, agent_bs).uniform_(0, 0.2),
            torch.empty(num_steps, expert_bs).uniform_(0.8, 1.)], dim=1).to(device)

        # Compute losses
        # =======
        # Critic-loss
        value_loss = (v_target - values).pow(2)
        value_loss = value_loss[:, :agent_bs, ...].mean()

        # Actor-loss
        pg_loss = -(advantages * ac_log_probs)
        pg_loss = pg_loss[:, :agent_bs, ...].mean()

        # Entropy- and Belief-regularizations
        dist_entropy = dist_entropy[:, :agent_bs, ...].mean()
        reg_loss = reg_loss.mean()

        # Discriminator loss
        discriminator_loss = adversarial_criterion(discriminator_sigmoids_d.squeeze(dim=2), discriminator_targets)
        discriminator_loss = discriminator_loss[:, :agent_bs].mean() + discriminator_loss[:, agent_bs:].mean()

        # Adversarial loss for belief RNN
        adversarial_b_loss = adversarial_criterion(discriminator_sigmoids_b.squeeze(dim=2), discriminator_targets)
        adversarial_b_loss = adversarial_b_loss[:, :agent_bs].mean() + adversarial_b_loss[:, agent_bs:].mean()
        # =======

        total_loss = (value_loss * loss_function['value_loss_coef']
                      + pg_loss * loss_function['pg_loss_coef']
                      + reg_loss * loss_function['reg_loss_coef']
                      - adversarial_b_loss * loss_function['adversarial_loss_coef']
                      - dist_entropy * loss_function['entropy_coef'])

        # Only reset the (recurrent part of the) computation graph every 'multiplier_backprop_length' iterations
        retain_graph = j % algorithm['multiplier_backprop_length'] != 0

        bp_optimizer.zero_grad()
        total_loss.backward(retain_graph=retain_graph)

        d_optimizer.zero_grad()
        discriminator_loss_ = discriminator_loss * loss_function['adversarial_loss_coef']
        discriminator_loss_.backward(retain_graph=False)

        if opt['max_grad_norm'] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt['max_grad_norm'])

        # Update parameters for all networks
        bp_optimizer.step()
        d_optimizer.step()

        if not retain_graph:
            curr_memory['prev_belief'] = curr_memory['prev_belief'].detach()

        rollouts.after_update()

        # ====
        # Off-policy learning
        # ====
        if replay.initialized:

            # Do certain number of off-policy updates "per" on-policy update
            for _ in range(algorithm['offPol_updates_mult']):

		# Buffer to store the computation output for gradient calculation
                offPol_reg_loss = []

                # Loop over "num_steps" timesteps for one off-policy update
                for _ in range(num_steps):

                    offPol_ob, offPol_prev_ob, offPol_prev_ac, offPol_future_k_obs, offPol_future_k_acs, offPol_future_mask, \
                            offPol_past_k_obs, offPol_past_k_acs, offPol_past_mask, offPol_done = replay.step()

                    curr_memory_offPol['curr_ob'] = torch.from_numpy(offPol_ob)     # o_{t}
                    curr_memory_offPol['prev_ob'] = torch.from_numpy(offPol_prev_ob)  # o_{t-1}
                    curr_memory_offPol['prev_ac'] = torch.from_numpy(offPol_prev_ac)   # a_{t-1}

                    ob_tpk = np.split(offPol_future_k_obs, fwd_jump, axis=1)[-1]   # o_{t+k}
                    curr_memory_offPol['ob_tpk'] = torch.from_numpy(ob_tpk)
                    curr_memory_offPol['future_k_acs'] = torch.from_numpy(offPol_future_k_acs) # a_{t}, a_{t+1}, ... a_{t+k-1}
                    curr_memory_offPol['future_mask'] = torch.from_numpy(offPol_future_mask).float()

                    ob_tmkm1 = np.split(offPol_past_k_obs, bwd_jump, axis=1)[0]  # o_{t-k-1}
                    curr_memory_offPol['ob_tmkm1'] = torch.from_numpy(ob_tmkm1)
                    curr_memory_offPol['past_k_acs'] = torch.from_numpy(offPol_past_k_acs)  # a_{t-2}, a_{t-3}, ... a_{t-k-1}
                    curr_memory_offPol['past_mask'] = torch.from_numpy(offPol_past_mask).float()

                    # Run models with off-policy data
                    model_return = run_model_offPol(model, curr_memory_offPol)

                    # Refresh belief state for new episode
                    offPol_mask = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in offPol_done]).to(device)
                    curr_memory_offPol['prev_belief'] = model_return.belief_state * offPol_mask

                    offPol_reg_loss.append(model_return.reg_loss)
                    tracked_values['reg_loss_scalar_offPol'].append(model_return.reg_loss_item) # used only for printing

                offPol_reg_loss = torch.stack(tuple(offPol_reg_loss), dim=0).mean() * loss_function['reg_loss_coef']

                # Only reset the (recurrent part of the) computation graph every 'multiplier_backprop_length' iterations
                retain_offPol_graph = num_offPol_updates % algorithm['multiplier_backprop_length'] != 0

                bp_optimizer.zero_grad()
                offPol_reg_loss.backward(retain_graph=retain_offPol_graph)

                if opt['max_grad_norm'] > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), opt['max_grad_norm'])

                bp_optimizer.step()
                num_offPol_updates += 1

                if not retain_offPol_graph:
                    curr_memory_offPol['prev_belief'] = curr_memory_offPol['prev_belief'].detach()

        # Logging
        if j % log['log_interval'] == 0:
            end = time.time()
            helpers.log_and_print(j, num_updates, end - start, final_rewards[:agent_bs], final_lens[:agent_bs],
                                tracked_values, num_ended_episodes, total_loss, value_loss, pg_loss, dist_entropy,
                                discriminator_loss, algorithm, _run)
