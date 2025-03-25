import sys
import traceback
from lru import LRU
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from sklearn.metrics import euclidean_distances

def med_width(data):
    K = euclidean_distances(data, squared=False)
    return np.median(K[K > 0])

class BIC:
    def __init__(self, data, variant, regressor, regression_kwargs=None, med_w=False, **kwargs):
        self.data = data
        self.regressor = regressor
        self.regression_kwargs = regression_kwargs or {}
        self.med_w = med_w
        self.n, self.d = data.shape
        self.penalty = np.log(self.n) / (self.n * self.d)
        self.variant = variant
        self.cov = np.dot(data.T, data)

    def partial_result(self, target, parents):
        X = self.data[:, parents]
        y = self.data[:, target]
        if parents:
            if self.regressor == 'LinearRegression':
                theta = np.linalg.solve(self.cov[np.ix_(parents, parents)], self.cov[parents, target])
                y_pred = X.dot(theta)
            elif self.regressor == 'GP':
                if self.med_w:
                    X = X / med_width(X)
                model = GaussianProcessRegressor(kernel=RBF(length_scale_bounds=(1, 1e5)), **self.regression_kwargs).fit(X, y)
                y_pred = model.predict(X)
            else:
                raise ValueError(f'Regressor "{self.regressor}" not implemented.')
        else:
            y_pred = y.mean()
        mse = ((y - y_pred) ** 2).mean()

        # https://github.com/huawei-noah/trustworthyAI/blob/b7ae2820da1ff4408264e0eba95a3083be54294c/research/Causal%20Discovery%20with%20RL/src/rewards/Reward_BIC.py#L121
        # "if the regresors include the true parents, GPR would result in very samll values, e.g., 10^-13, so we add 1.0, which does not affect the monotoniticy of the score"
        if self.regressor == 'GP':
            mse += 1 / len(y)
        return mse

    def merge_partial_results(self, dag, partial_results):
        n_edges = np.sum(dag)
        if self.variant == 'EV':
            sum_mse = sum(partial_results.values())
            bic_ev = -np.log(sum_mse / self.d) - n_edges * self.penalty 
            return bic_ev
        elif self.variant == 'NV':
            bic_nv = -np.mean(np.log(np.asarray(list(partial_results.values())) + 1e-8)) - n_edges * self.penalty
            return bic_nv

class DAGScore:
    def __init__(self, X, dag_scorer, scorer_kwargs, cache_capacity):
        self.scorer_obj = {'BIC': BIC}[dag_scorer](data=X, **scorer_kwargs)
        self.score_cache = dict() if cache_capacity is None else LRU(cache_capacity)
        self.data = X
        self.n, self.d = X.shape

    def evaluate(self, A):
        partial_results = {}
        for i in range(self.d):
            parents, = np.nonzero(A[:, i])
            k = (i, tuple(parents))
            if k in self.score_cache:
                partial_results[k] = self.score_cache[k]
            else:
                self.score_cache[k] = partial_results[k] = self.scorer_obj.partial_result(i, tuple(parents))
                
        whole_score = self.scorer_obj.merge_partial_results(A, partial_results)
        return whole_score
    
    def loop_mp(self, index, in_queue, out_queue, error_queue):
        try:
            while True:
                key = in_queue.get()
                if key is None:
                    break

                target, parents = key
                partial_result = self.scorer_obj.partial_result(target, parents)

                out_queue.put((True, key, partial_result))

        except (KeyboardInterrupt, Exception):
            error_queue.put((index,) + sys.exc_info()[:2])
            print(traceback.format_exc())
            out_queue.put((False, None, None))