from copy import deepcopy
import logging
from alias.DAGEnv import DAGEnv
from alias.VecDAGEnv import VecDAGEnv
from alias.DAGScore import DAGScore
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import A2C, PPO

def ALIAS(
        X,
        standardize_data=False,
        steps_per_env=20_000,
        n_envs=64,
        rl_method='PPO',
        rl_params=None,
        dag_scorer_cls='BIC',
        dag_scorer_kwargs={},
        n_score_workers=0,
        score_cache_capacity=10_000_000,
        verbose=True,
        random_state=0,
        **unused
    ):
    assert rl_method in ['PPO', 'A2C']
    rl_params = deepcopy(rl_params or {})
    rl_params['n_steps'] = 1 # our trajectories are only 1-step
    if rl_method == 'A2C':
        rl_params.pop('n_epochs', None)
    else:
        rl_params.setdefault('n_epochs', 1)
    n, d = X.shape
    steps = steps_per_env * n_envs

    if standardize_data:
        X = StandardScaler().fit_transform(X)

    dag_scorer_cls = DAGScore(dag_scorer=dag_scorer_cls, X=X, scorer_kwargs=dag_scorer_kwargs, cache_capacity=score_cache_capacity)

    env = DAGEnv(nodes=d, dag_scorer=dag_scorer_cls)
    vec_env = VecDAGEnv(scorer=dag_scorer_cls, n_envs=n_envs,
                        n_workers=n_score_workers, env=env) if n_envs > 1 else env

    model = eval(rl_method)("MlpPolicy", vec_env, seed=random_state, **rl_params)
    model.learn(total_timesteps=steps, progress_bar=verbose)

    vec_env = model.get_env()
    obs = vec_env.reset()
    action, _states = model.predict(obs, deterministic=True)
    est = env.vec2dag(action[0])
    logging.info(f'Est reward = {env.dag_scorer.evaluate(est)}')
    logging.info(f'Est =\n{est}')

    return est