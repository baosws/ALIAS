from castle.datasets import IIDSimulation, DAG
import numpy as np
import logging
import pandas as pd

def sim_pe(nodes, samples, graph_type, linear, edges_per_node, noise_scale, weight_range, sem_type='gauss', random_state=0, **kwargs):
    rng = np.random.RandomState(random_state)
    edges = nodes * edges_per_node
    if graph_type == 'ER':
        weighted_random_dag = DAG.erdos_renyi(n_nodes=nodes, n_edges=edges, weight_range=weight_range, seed=random_state)
    elif graph_type == 'SF':
        weighted_random_dag = DAG.scale_free(n_nodes=nodes, n_edges=edges, weight_range=weight_range, seed=random_state)
    else:
        raise ValueError(f'Unknown graph type: {graph_type}')

    random_order = rng.randn(nodes).argsort()
    weighted_random_dag = weighted_random_dag[np.ix_(random_order, random_order)]
    dataset = IIDSimulation(W=weighted_random_dag, n=samples, noise_scale=noise_scale, method='linear' if linear else 'nonlinear', sem_type=sem_type)
    graph, data = dataset.B, dataset.X
    return dict(X=data, GT=graph)

def sim_noisy(nodes, samples, graph_type, linear, edges_per_node, noise_scale, p, weight_range, sem_type, random_state=0, **kwargs):
    rng = np.random.RandomState(random_state)
    edges = nodes * edges_per_node
    if graph_type == 'ER':
        weighted_random_dag = DAG.erdos_renyi(n_nodes=nodes, n_edges=edges, weight_range=weight_range, seed=random_state)
    elif graph_type == 'SF':
        weighted_random_dag = DAG.scale_free(n_nodes=nodes, n_edges=edges, weight_range=weight_range, seed=random_state)

    random_order = rng.randn(nodes).argsort()
    weighted_random_dag = weighted_random_dag[np.ix_(random_order, random_order)]
    dataset = IIDSimulation(W=weighted_random_dag, n=samples, noise_scale=noise_scale, method='linear' if linear else 'nonlinear', sem_type=sem_type)
    graph, data = dataset.B, dataset.X

    data = np.where(rng.rand(samples, nodes) <= p, rng.randn(samples, nodes) + data, data)
    return dict(X=data, GT=graph)

def sim_confounder(nodes, k, samples, graph_type, linear, edges_per_node, noise_scale, weight_range, sem_type, random_state=0, **kwargs):
    rng = np.random.RandomState(random_state)
    nodes += k
    edges = nodes * edges_per_node
    if graph_type == 'ER':
        weighted_random_dag = DAG.erdos_renyi(n_nodes=nodes, n_edges=edges, weight_range=weight_range, seed=random_state)
    elif graph_type == 'SF':
        weighted_random_dag = DAG.scale_free(n_nodes=nodes, n_edges=edges, weight_range=weight_range, seed=random_state)
    else:
        raise ValueError(f'Unknown graph type: {graph_type}')

    random_order = rng.randn(nodes).argsort()
    weighted_random_dag = weighted_random_dag[np.ix_(random_order, random_order)]
    logging.info(f'W =\n{np.round(weighted_random_dag, 2)}')
    dataset = IIDSimulation(W=weighted_random_dag, n=samples, noise_scale=noise_scale, method='linear' if linear else 'nonlinear', sem_type=sem_type)
    graph, data = dataset.B, dataset.X
    logging.info(f'GT =\n{graph}')
    logging.info(f'{data.min() = :.4f}, {data.max() = :.4f}')
    if k:
        chosen_nodes = np.random.choice(nodes, size=nodes - k, replace=False)
        data = data[:, chosen_nodes]
        graph = graph[np.ix_(chosen_nodes, chosen_nodes)]

    return dict(X=data, GT=graph)

def real_data(name, n_samples=None, random_state=None, **kwargs):
    if random_state is None:
        random_state = 0
    rng = np.random.RandomState(random_state)
    X = pd.read_csv(f'data/{name}/observations.csv').values
    GT = pd.read_csv(f'data/{name}/dag.csv', index_col='source').values
    if n_samples is None:
        n_samples = X.shape[0]
    idx = rng.choice(X.shape[0], size=n_samples, replace=False)

    return dict(X=X[idx], GT=GT)

def nonlinear_gp(name, id, **kwargs):
    data = np.load(f'data/{name}/data{id}.npy')
    dag = np.load(f'data/{name}/DAG{id}.npy')
    return dict(X=data, GT=1 * (np.abs(dag) > 1e-3))