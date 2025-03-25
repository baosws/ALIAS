import gymnasium as gym
import numpy as np
from gymnasium import spaces

class DAGEnv(gym.Env):
    def __init__(self, nodes, dag_scorer):
        super().__init__()
        self.dag_scorer = dag_scorer
        self.nodes = nodes
        action_shape = (nodes + nodes * (nodes - 1) // 2,)
        self.action_space = spaces.Box(-10, 10, action_shape)
        self.observation_space = spaces.Discrete(1)
        self.tril_indices = np.tril_indices(nodes, -1)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self._obs = np.array(0)
        return self._obs, {}

    def vec2dag(self, z):
        d = self.nodes
        p = z[:d]					    # R^d
        E = np.zeros((d, d))
        E[self.tril_indices] = z[d:]	# R^(d(d-1)/2)

        A = (E + E.T > 0) * (p[:, None] < p[None, :]) * 1
        return A

    def step(self, action):
        dag = self.vec2dag(action)
        self._obs = np.array(0)

        reward = self.dag_scorer.evaluate(dag)
        terminated = True
        truncated = False
        info = {}

        return self._obs, reward, terminated, truncated, info
    
    def step_async(self, action, in_queue, put_set):
        dag = self.vec2dag(action)
        self._obs = np.array(0)

        n_queued = 0
        for node in range(self.nodes):
            parents = tuple(np.nonzero(dag[:, node])[0])
            key = (node, parents)
            if key not in self.dag_scorer.score_cache and key not in put_set:
                put_set.add(key)
                in_queue.put(key)
                n_queued += 1

        return n_queued, (dag, self._obs)
    
    def step_wait(self, ref):
        dag, obs = ref
        score = self.dag_scorer.evaluate(dag)
        return obs, score, True, False, {}