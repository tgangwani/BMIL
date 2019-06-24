import random
import numpy as np

def storage_requirement():
    s = ("Expert trajectories should be stored as a list. Each list-entry should be a dictionary"
         "with keys {'obs', 'acs'}. In each dictionary d, d['obs'] & d['acs'] should be lists, with"
         "each entry of this list being a numpy.ndarray"
         )
    return s

class ExpertEnv():
    def __init__(self, env_id, seed, rank, num_expert_envs, database):
        """
        This "environment" only replays trajectories from an expert database, rather than interacting with Gym.
        """

        self.env_id = env_id
        random.seed(seed + rank)
        if database.endswith('.npy'):
            expert_paths = np.load(database).tolist()
        elif database.endswith('.pickle'):
            import pickle
            with open(database, 'rb') as handle:
                expert_paths = pickle.load(handle)
        else: raise NotImplementedError

        assert isinstance(expert_paths, list), storage_requirement()
        self.rank_paths = expert_paths

        # If we have more expert paths than the number of expert envs., we partition the paths b/w the expert envs.
        if len(expert_paths) > num_expert_envs:
            self.rank_paths = expert_paths[rank:][::num_expert_envs]
            del expert_paths

        print('Rank:{} loaded {} paths from {}'.format(rank, len(self.rank_paths), database))

    def reset(self):
        self.current_path = random.choice(self.rank_paths)
        self.current_path_idx = 0
        return self._get_curr_ob()

    def _get_curr_ob(self):
        return self.current_path['obs'][self.current_path_idx]

    def get_curr_ac(self):
        return self.current_path['acs'][self.current_path_idx]

    def step(self, action):
        done = False

        self.current_path_idx += 1
        if self.current_path_idx == len(self.current_path['obs']):
            self.current_path_idx = 0
            done = True

        return self._get_curr_ob(), 0, done, {}  # ob, rew, done, info.

    def render(self):
        raise NotImplementedError

    @property
    def observation_space(self):
        raise NotImplementedError

    @property
    def action_space(self):
        raise NotImplementedError
