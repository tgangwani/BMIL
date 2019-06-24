import os
import numpy as np

import gym
from gym.spaces.box import Box

from baselines import bench
from expert_envs import ExpertEnv

BIG = 1e6

def make_env(env_id, seed, rank, log_dir, occlusion, sensor_noise):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)

        assert not (hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)), "Atari not supported. Please use gym MuJoCo envs!"

        if log_dir is not None:
            print("Create Monitor in {}".format(log_dir))
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))

        if sensor_noise > 0:
            # wrapper to add noise to observations
            print("Noise Wrapper")
            env = NoisyWrapper(env)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            print("Wrap Pytorch")
            env = WrapPyTorch(env)

        if len(occlusion):
            print("Occlusion Wrapper")

            # Since occlusion-list for Ant-v2 and Humanoid-v2 is too-long, we generate it here using codewords used in envParams.yaml
            if occlusion == [9999]:  # Ant-v2
                print("Generating Ant-v2 occlusion list")
                generated_occlusion = [x for x in range(13)] + [x for x in range(27, 111)]
                env = OccludeWrapper(env, generated_occlusion)
            elif occlusion == [7777]:  # Humanoid:-v2
                print("Generating Humanoid-v2 occlusion list")
                generated_occlusion = [x for x in range(22)] + [x for x in range(45, 185)] + [x for x in range(269, 376)]
                env = OccludeWrapper(env, generated_occlusion)
            else:
                env = OccludeWrapper(env, occlusion)

        print(env)
        return env

    return _thunk

def make_expert_envs(*args):
    def _thunk():
        env = ExpertEnv(*args)
        return env
    return _thunk

class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]]
        )

    def _observation(self, observation):
        return observation.transpose(2, 0, 1)

class NoisyWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        """
        Replace o_t with a uniform sample in [o_t*(1-self.delta), o_t*(1+self.delta)] with a probability self.p
        """
        super(NoisyWrapper, self).__init__(env)
        self.p = 0.9
        self.delta = .5

    def _observation(self, observation):
        if np.random.binomial(1, self.p, 1).item():
            observation_perturbed = np.random.uniform(low=observation*(1-self.delta), high=observation*(1+self.delta), size=observation.shape)
            return observation_perturbed

        return observation

# https://github.com/rll/rllab/blob/master/rllab/envs/occlusion_env.py
class OccludeWrapper(gym.ObservationWrapper):
    def __init__(self, env, sensor_idx):
        """
        :param sensor_idx: list or ndarray of indices to be shown. Other indices will be occluded. Can be either list of
            integer indices or boolean mask.
        """
        super(OccludeWrapper, self).__init__(env)
        self._set_sensor_mask(sensor_idx)

        # set new observation space.
        ub = BIG * np.ones(env.observation_space.shape)
        ub = self.occlude(ub)
        self.observation_space = Box(ub * -1, ub)

    def _set_sensor_mask(self, sensor_idx):
        obsdim = np.prod(self.env.observation_space.high.shape)
        if len(sensor_idx) > obsdim:
            raise ValueError("Length of sensor mask ({0}) cannot be greater than observation dim ({1})".format(len(sensor_idx), obsdim))
        if len(sensor_idx) == obsdim and not np.any(np.array(sensor_idx) > 1):
            sensor_mask = np.array(sensor_idx, dtype=np.bool)
        elif np.any(np.unique(sensor_idx, return_counts=True)[1] > 1):
            raise ValueError("Double entries or boolean mask with dim ({0}) < observation dim ({1})".format(len(sensor_idx), obsdim))
        else:
            sensor_mask = np.zeros((obsdim,), dtype=np.bool)
            sensor_mask[sensor_idx] = 1
        self._sensor_mask = sensor_mask

    def _observation(self, observation):
        return self.occlude(observation)

    def occlude(self, observation):
        return observation[self._sensor_mask]
