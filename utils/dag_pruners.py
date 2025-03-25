from sklearn.linear_model import Ridge
import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import ListVector
from rpy2.robjects import numpy2ri
from causallearn.utils.cit import KCI

def prune_by_linear_weight(X, dag, threshold=0.3, l2=0):
    n, d = X.shape
    ret = []

    for node in range(d):
        parents, = np.nonzero(dag[:, node])
        if not len(parents):
            ret.append(np.zeros(d))
        else:
            x = X[:, parents]
            y = X[:, node]

            regressor = Ridge(alpha=l2).fit(x, y)

            all_weights = np.zeros(d, )
            all_weights[parents] = regressor.coef_

            ret.append(all_weights)

    return (np.abs(ret) >= threshold).T.astype(int)

def no_prune(X, dag):
    return dag

base = rpackages.importr('base')
utils = rpackages.importr('utils')

cam = rpackages.importr('CAM')
mboost = rpackages.importr('mboost')
def _pruning(X, G, pruneMethod = robjects.r.selGam, pruneMethodPars = ListVector({'cutOffPVal': 0.001, 'numBasisFcts': 10}), output = False):
    # X is a r matrix
    # G is a python numpy array adj matrix,

    d = G.shape[0]
    X = robjects.r.matrix(numpy2ri.py2rpy(X), ncol=d)
    G = robjects.r.matrix(numpy2ri.py2rpy(G), d, d)
    finalG = robjects.r.matrix(0, d, d)
    for i in range(d):
        parents = robjects.r.which(G.rx(True, i + 1).ro == 1)
        lenpa = robjects.r.length(parents)[0]
        if lenpa > 0:
            Xtmp = robjects.r.cbind(X.rx(True, parents), X.rx(True, i+1))
            selectedPar = pruneMethod(Xtmp, k = lenpa + 1, pars = pruneMethodPars, output = output)
            finalParents = parents.rx(selectedPar)
            finalG.rx[finalParents, i+1] = 1

    return np.array(finalG)

def pruning_cam(XX, Adj):
    X2 = numpy2ri.py2rpy(XX)
    Adj = _pruning(X = X2, G = Adj, pruneMethod = robjects.r.selGam, pruneMethodPars = ListVector({'cutOffPVal': 0.001, 'numBasisFcts': 10}), output = False)

    return Adj

def pruning_cit(X, dag):
    ret = dag.copy()
    n, d = X.shape
    cit = KCI(data=X)
    for i in range(d):
        parents, = np.nonzero(dag[:, i])
        for j in parents:
            if cit(i, j, [k for k in parents if k != j]) >= 0.001:
                ret[j, i] = 0

    return ret