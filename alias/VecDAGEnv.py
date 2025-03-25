# adapted from StableBasline3: https://github.com/DLR-RM/stable-baselines3/blob/a9273f968eaf8c6e04302a07d803eebfca6e7e86/stable_baselines3/common/vec_env/dummy_vec_env.py#L14
import logging
import os
from collections import OrderedDict
from copy import deepcopy
import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info
from multiprocessing import get_context

class VecDAGEnv(VecEnv):
    def __init__(self, scorer, n_envs, n_workers, env):
        self.scorer = scorer
        self.envs = [env] * n_envs
        env = self.envs[0]
        super().__init__(num_envs=n_envs, observation_space=env.observation_space, action_space=env.action_space)

        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)
        self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        if n_workers < 0:
            n_workers += os.cpu_count() + 1
        self.n_workers = n_workers

        if n_workers > 0:
            ctx = get_context('fork')

            self.in_queue = ctx.Queue()
            self.out_queue = ctx.Queue()
            self.error_queue = ctx.Queue()

            self.processes = []
            for index in range(n_workers):
                process = ctx.Process(
                    target=self.scorer.loop_mp,
                    args=(index, self.in_queue, self.out_queue, self.error_queue),
                    daemon=True
                )
                process.start()
                self.processes.append(process)
            logging.info(f'Launched {n_workers} processes.')

    def reset(self):
        obses = []
        for env_idx in range(self.num_envs):
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(seed=self._seeds[env_idx])
            obses.append(obs)
        self._reset_seeds()
        return np.asarray(obses)

    def step_async(self, actions: np.ndarray):
        self.avails = []
        self.n_queued = 0
        if self.n_workers > 0:
            put_set = set()
            for env, action in zip(self.envs, actions):
                n_queued, res = env.step_async(action, self.in_queue, put_set)
                self.avails.append(res)
                self.n_queued += n_queued
        else:
            for env, action in zip(self.envs, actions):
                self.avails.append(env.step(action))
    
    def step_wait(self):
        for _ in range(self.n_queued):
            is_success, key, value = self.out_queue.get()

            if is_success:
                self.scorer.score_cache[key] = value
            else:
                _, exctype, value = self.error_queue.get()
                raise exctype(value)
        
        for env_idx, data in enumerate(self.avails):
            obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step_wait(data) if self.n_workers > 0 else data
            self.buf_dones[env_idx] = terminated or truncated
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

            if self.buf_dones[env_idx]:
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        ret = (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))
        return ret

    def close(self):
        for env in self.envs:
            env.close()
        for t in self.threads if hasattr(self, 'threads') else self.processes:
            t.join()

    def _save_obs(self, env_idx: int, obs):
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]  # type: ignore[call-overload]

    def _obs_from_buf(self):
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))
    
    def get_attr(self, attr_name: str, indices = None):
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value, indices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices = None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def env_is_wrapped(self, wrapper_class, indices = None):
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices):
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]