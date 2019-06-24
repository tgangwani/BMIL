#!/usr/bin/env python
"""
Buffer to hold trajectories from the agent
"""

from collections import deque
import random
import numpy as np

class ExpReplay():
    def __init__(self, batch_size, max_trajs, fwd_jump, bwd_jump):
        """
        Replay self.batch_size number of parallel episodes
        """
        self.batch_size = batch_size
        self._traj_q = deque(maxlen=max_trajs)
        self.initialized = False
        self.running_paths = []
        self.current_idx = np.zeros(shape=(batch_size), dtype=np.int)  # current index (timestep) for each episode
        self.path_len = np.zeros(shape=(batch_size), dtype=np.int)
        self.fwd_jump = fwd_jump
        self.bwd_jump = bwd_jump

    def _initialize(self):
        """
        Initialize the parallel episodes
        """
        for i in range(self.batch_size):
            path = self._traj_q[i]
            self.running_paths.append(path)
            self.path_len[i] = len(path["obs"])

        ob_shape = self.running_paths[0]["obs"][0].shape[0]
        ac_shape = self.running_paths[0]["prev_acs"][0].shape[0]

        # Initialize buffers to hold state
        self.current_ob = np.zeros((self.batch_size, ob_shape), np.float32)
        self.prev_ob = np.zeros((self.batch_size, ob_shape), np.float32)
        self.prev_ac = np.zeros((self.batch_size, ac_shape), np.float32)
        self.future_k_obs = np.zeros((self.batch_size, self.fwd_jump*ob_shape), np.float32)
        self.future_k_acs = np.zeros((self.batch_size, self.fwd_jump*ac_shape), np.float32)
        self.past_k_obs = np.zeros((self.batch_size, self.bwd_jump*ob_shape), np.float32)
        self.past_k_acs = np.zeros((self.batch_size, self.bwd_jump*ac_shape), np.float32)
        self.future_mask = np.ones((self.batch_size, 1), np.float32)
        self.past_mask = np.ones((self.batch_size, 1), np.float32)
        self.done = np.zeros(self.batch_size, np.float32)

        self.initialized = True
        print("Experience-Replay initialized with %d parallel episodes.."%(self.batch_size))

    def process_path(self, path):
        """
        For the trajectory, {future/past}-k observations, {future/past}-k actions, and corresponding masks
        """
        future_k_obs = []   # o_{t+1}, .. o_{t+k}
        future_k_acs = []   # a_{t}, .. a_{t+k-1}
        past_k_obs = []  # o_{t-k-1}, .. o_{t-2}
        past_k_acs = []   # a_{t-k-1}, .. a_{t-2}

        # padding (append) with 0
        padded_obs = np.pad(path["obs"], ((0, self.fwd_jump), (0, 0)), mode='constant')
        padded_prev_acs = np.pad(path["prev_acs"], ((0, self.fwd_jump), (0, 0)), mode='constant')

        for i in range(len(path["obs"])):
            future_k_obs.append(padded_obs[i+1:i+1+self.fwd_jump])
            future_k_acs.append(padded_prev_acs[i+1:i+1+self.fwd_jump])

        # mask for k-step future predictions
        future_masks = np.ones(len(path["obs"]))
        future_masks[-self.fwd_jump:] = 0

        path.update({"future_k_obs":np.array(future_k_obs), "future_k_acs":np.array(future_k_acs), "future_masks":future_masks})

        # padding (prepend) with 0
        padded_obs = np.pad(path["prev_obs"], ((self.bwd_jump, 0), (0, 0)), mode='constant')
        padded_prev_acs = np.pad(path["prev_acs"], ((self.bwd_jump, 0), (0, 0)), mode='constant')

        for i in range(self.bwd_jump, self.bwd_jump + len(path["obs"])):
            past_k_obs.append(padded_obs[i-self.bwd_jump:i])
            #past_k_acs.append(padded_prev_acs[i-self.bwd_jump:i]) # store in order: a_{t-k-1}, .. a_{t-2}
            past_k_acs.append(padded_prev_acs[i-self.bwd_jump:i][::-1]) # store in reverse order: a_{t-2}, a_{t-3}, ... a_{t-k-1}

        # masks for k-step past predictions
        past_masks = np.ones(len(path["obs"]))
        past_masks[:self.bwd_jump] = 0

        path.update({"past_k_obs":np.array(past_k_obs), "past_k_acs":np.array(past_k_acs), "past_masks":past_masks})

    def add_path(self, path):
        """
        path is a dictionary with keys: obs, prev_obs, prev_acs
        """
        self.process_path(path)
        self._traj_q.append(path)

        if not self.initialized and len(self._traj_q) == self.batch_size:
            self._initialize()

    def step(self):
        """
        Advance all episodes by one timestep. If any episode is complete, start a new one.
        """

        self.done *= 0
        for i in range(self.batch_size):
            self.current_ob[i] = self.running_paths[i]["obs"][self.current_idx[i]]
            self.prev_ob[i] = self.running_paths[i]["prev_obs"][self.current_idx[i]]
            self.prev_ac[i] = self.running_paths[i]["prev_acs"][self.current_idx[i]]
            self.future_k_obs[i] = self.running_paths[i]["future_k_obs"][self.current_idx[i]].flatten()
            self.future_k_acs[i] = self.running_paths[i]["future_k_acs"][self.current_idx[i]].flatten()
            self.past_k_obs[i] = self.running_paths[i]["past_k_obs"][self.current_idx[i]].flatten()
            self.past_k_acs[i] = self.running_paths[i]["past_k_acs"][self.current_idx[i]].flatten()
            self.future_mask[i] = self.running_paths[i]["future_masks"][self.current_idx[i]]
            self.past_mask[i] = self.running_paths[i]["past_masks"][self.current_idx[i]]

            self.current_idx[i] += 1

            if self.current_idx[i] == self.path_len[i]:   # episode over
                self.done[i] = 1

                # Start a new random episode
                random_path_idx = random.choices(range(len(self._traj_q)))[0]
                self.running_paths[i] = random_path = self._traj_q[random_path_idx]
                self.path_len[i] = len(random_path["obs"])
                self.current_idx[i] = 0  # reset

        return self.current_ob, self.prev_ob, self.prev_ac, self.future_k_obs, self.future_k_acs, self.future_mask, \
                self.past_k_obs, self.past_k_acs, self.past_mask, self.done
