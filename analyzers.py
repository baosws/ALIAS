import numpy as np, pandas as pd
from utils.dag_pruners import prune_by_linear_weight, no_prune, pruning_cam, pruning_cit
from utils.metrics import MetricsDAG

pd.set_option("display.precision", 2)

def analyze_linear(df):
    df['mins'] = df['runtime'] / 60
    df['graph'] = df['data.params.graph_type'] + '-' + df['data.params.edges_per_node'].astype(str)

    d = pd.concat([df, df.apply(lambda row: MetricsDAG(prune_by_linear_weight(row['data.samples.X'], row['pred.raw'].astype(int)), row['data.samples.GT'].astype(int)), axis=1).apply(pd.Series)], axis=1)
    metric_names = ['shd', 'fdr', 'tpr', 'mins']
    d = d.groupby(['graph', 'data.params.nodes', 'method.name'])[metric_names]
    print(res := np.round(d.mean(), 2).astype(str) + '±' + np.round(d.sem(), 2).astype(str))

def analyze_sachs(df):
    metric_names = ['nnz', 'correct', 'shd', 'extra', 'missing', 'reverse', 'tpr', 'fdr', 'precision', 'recall', 'F1']
    for pruner in [no_prune, pruning_cam, pruning_cit]:
        print(f'{pruner = }')
        d = pd.concat([df, df.apply(lambda row: MetricsDAG(pruner(row['data.samples.X'], row['pred.raw'].astype(int)), row['data.samples.GT'].astype(int)), axis=1).apply(pd.Series)], axis=1)
        print(d.groupby(['data.params.name', 'method.name'])[metric_names].mean())

def analyze_gp(df):
    for pruner in [no_prune, pruning_cam]:
        print(f'{pruner = }')
        d = pd.concat([df, df.apply(lambda row: MetricsDAG(pruner(row['data.samples.X'], row['pred.raw'].astype(int)), row['data.samples.GT'].astype(int)), axis=1).apply(pd.Series)], axis=1)
        metric_names = ['nnz', 'fdr', 'tpr', 'shd', 'missing', 'extra', 'reverse']
        print((np.round(d.groupby(['method.name'])[metric_names].mean(), 2).astype(str) + '±' + np.round(d.groupby(['method.name'])[metric_names].std(), 2).astype(str)))
        print()

def analyze_samplesizes(df):
    df['graph'] = df['data.params.graph_type'] + '-' + df['data.params.edges_per_node'].astype(str)

    d = pd.concat([df, df.apply(lambda row: MetricsDAG(prune_by_linear_weight(row['data.samples.X'], row['pred.raw'].astype(int)), row['data.samples.GT'].astype(int)), axis=1).apply(pd.Series)], axis=1)
    metric_names = ['shd', 'fdr', 'tpr']
    d = d.groupby(['graph', 'data.params.nodes', 'data.params.samples', 'method.name'])[metric_names]
    print(np.round(d.mean(), 2).astype(str) + '±' + np.round(d.sem(), 2).astype(str))

def analyze_different_noises(df):
    df['graph'] = df['data.params.graph_type'] + '-' + df['data.params.edges_per_node'].astype(str)
    d = pd.concat([df, df.apply(lambda row: MetricsDAG(prune_by_linear_weight(row['data.samples.X'], row['pred.raw'].astype(int)), row['data.samples.GT'].astype(int)), axis=1).apply(pd.Series)], axis=1)
    metric_names = ['shd', 'fdr', 'tpr']
    id_cols = ['graph', 'data.params.nodes', 'data.params.sem_type', 'method.name']
    d = d.groupby(id_cols)[metric_names]
    print(np.round(d.mean(), 2).astype(str) + '±' + np.round(d.sem(), 2).astype(str))

def analyze_ablation(df):
    d = pd.concat([df, df.apply(lambda row: MetricsDAG(prune_by_linear_weight(row['data.samples.X'], row['pred.raw'].astype(int)), row['data.samples.GT'].astype(int)), axis=1).apply(pd.Series)], axis=1)
    metric_names = ['shd', 'fdr', 'tpr']
    varying_cols = ['method.name'] + [col for col in d.columns if col.startswith('method.params') and d[col].nunique() > 1]
    print(f'{varying_cols =}')
    d = d.groupby(varying_cols)[metric_names]
    mean = np.round(d.mean(), 2)
    sem = np.round(d.sem(), 2)
    res = mean.astype(str) + '±' + sem.astype(str)
    print(res)

def analyze_different_graphs(df):
    df['graph'] = df['data.params.graph_type'] + '-' + df['data.params.edges_per_node'].astype(str)
    d = pd.concat([df, df.apply(lambda row: MetricsDAG(prune_by_linear_weight(row['data.samples.X'], row['pred.raw'].astype(int)), row['data.samples.GT'].astype(int)), axis=1).apply(pd.Series)], axis=1)
    metric_names = ['shd', 'fdr', 'tpr']
    id_cols = ['data.params.nodes', 'data.params.samples', 'method.name']
    d = d.groupby(id_cols)[metric_names]
    mean = np.round(d.mean(), 2)
    sem = np.round(d.sem(), 2)
    res = mean.astype(str) + '±' + sem.astype(str)
    print(res)

def analyze_runtime(df):
    df['mins'] = np.round(df['runtime'] / 60, 2)
    df['graph'] = df['data.params.graph_type'] + '-' + df['data.params.edges_per_node'].astype(str)

    metric_names = ['shd', 'mins']
    d = pd.concat([df, df.apply(lambda row: MetricsDAG(prune_by_linear_weight(row['data.samples.X'], row['pred.raw'].astype(int)), row['data.samples.GT'].astype(int)), axis=1).apply(pd.Series)], axis=1)
    d = d.groupby(['graph', 'data.params.nodes', 'method.name'])[metric_names]
    res = np.round(d.mean(), 2).astype(str) + '±' + np.round(d.sem(), 2).astype(str)
    print(res)

def analyze_noisy(df):
    df['data.params.p'].fillna(0, inplace=True)
    df['graph'] = df['data.params.graph_type'] + '-' + df['data.params.edges_per_node'].astype(str)

    metric_names = ['shd']
    d = pd.concat([df, df.apply(lambda row: MetricsDAG(prune_by_linear_weight(row['data.samples.X'], row['pred.raw'].astype(int)), row['data.samples.GT'].astype(int)), axis=1).apply(pd.Series)], axis=1)
    d = d.groupby(['graph', 'data.params.nodes', 'method.name', 'data.params.p'])[metric_names]
    res = np.round(d.mean(), 1).astype(str) + '±' + np.round(d.sem(), 1).astype(str)
    print(res)

def analyze_confounder(df):
    df['data.params.k'].fillna(0, inplace=True)
    df['graph'] = df['data.params.graph_type'] + '-' + df['data.params.edges_per_node'].astype(str)

    metric_names = ['shd']
    d = pd.concat([df, df.apply(lambda row: MetricsDAG(prune_by_linear_weight(row['data.samples.X'], row['pred.raw'].astype(int)), row['data.samples.GT'].astype(int)), axis=1).apply(pd.Series)], axis=1)
    d = d.groupby(['graph', 'data.params.nodes', 'method.name', 'data.params.k'])[metric_names]
    res = np.round(d.mean(), 1).astype(str) + '±' + np.round(d.std(), 1).astype(str)
    res = pd.pivot(res.reset_index(), index='method.name', columns='data.params.k', values='shd')
    print(res)