import warnings
import os
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
import pandas as pd
from time import perf_counter
from copy import deepcopy
from functools import partial
from itertools import product
import logging
from pathlib import Path
import yaml
import sys
from alias.ALIAS import ALIAS
import analyzers
from utils import datagen

def _merge_dicts(target, source):
    target = deepcopy(target)
    source = deepcopy(source)

    for key, value in source.items():
        if key in target:
            # assert type(target[key]) == type(value), f'{type(target[key]) = } != {type(source[key]) = }'
            if isinstance(value, dict) and isinstance(target[key], dict):
                target[key] = _merge_dicts(target[key], value)
            else:
                target[key] = value
        else:
            target[key] = value

    return target

def _range_constructor(loader, node):
    # !range {count} = range(count)
    # !range start:end = range(start, end)
    # !range start:+count = range(start, start + count)
    vals = loader.construct_scalar(node).replace(' ', '').split(':')
    start = 0
    count = int(vals[-1])
    if len(vals) == 2:
        start = int(vals[0])
        if vals[1].startswith('+'):
            count = int(vals[1][1:])
        else:
            count = int(vals[1]) - start
    return list(range(start, start + count))

def _overwrite_constructor(loader, node):
    kwargs = loader.construct_mapping(node, deep=True)
    src = kwargs.pop('FROM', {})
    return _merge_dicts(target=src, source=kwargs)

def _yaml_constructor(loader, node, visited):
    # !yaml {path} or !yml {path}
    assert isinstance(node, (yaml.nodes.ScalarNode, yaml.nodes.SequenceNode))
    paths = loader.construct_scalar(node).split(
    ) if node.__class__ == yaml.nodes.ScalarNode else loader.construct_sequence(node)
    datas = [read_config(f'configs/{path}', visited=visited)
             for path in paths]
    data = {}
    for d in datas:
        data = _merge_dicts(data, d)
    return data

def _get_loader(visited):
    loader = yaml.SafeLoader
    loader.add_constructor("!range", _range_constructor)
    loader.add_constructor("!yml", partial(_yaml_constructor, visited=visited))
    loader.add_constructor("!overwrite", _overwrite_constructor)
    return loader

def read_config(path, **kwargs):
    visited = kwargs.get('visited', {})
    cur_path = str(Path(path).resolve().absolute())
    if visited.get(cur_path, 0) == 2:
        raise RuntimeError(
            f'Recursive YAML resolution detected: {cur_path = }, {visited = }')
    visited[cur_path] = 2
    with open(path, 'r') as f:
        params = yaml.load(f, Loader=_get_loader(visited)) or {}
    visited[cur_path] = 1
    return params

def _flatten_dict(tree):
    q = list(tree.items())
    res = {}
    for name, node in q:
        assert name != ''
        if not isinstance(node, dict) or not node:
            res[name] = deepcopy(node)
        else:
            for child_name, child_node in node.items():
                q.append((name + '.' + child_name, child_node))
    return res

def _unflatten_dict(d):
    res = {}
    for key, value in d.items():
        root = res
        levels = key.split('.')
        for level in levels[:-1]:
            if level not in root:
                root[level] = dict()
            root = root[level]
        root[levels[-1]] = value
    return res

def _eval_params_grid(grid):
    flatten_grid = _flatten_dict(grid)

    for key, value in flatten_grid.items():
        if not isinstance(value, list):
            flatten_grid[key] = [value]

    keys = list(flatten_grid.keys())
    values = list(flatten_grid.values())

    grid = [_unflatten_dict(dict(zip(keys, deepcopy(params)))) for params in product(*values)]
    return grid

if __name__ == '__main__':
    logging.basicConfig(level='ERROR', force=True)
    exp_name = sys.argv[1]
    print(f'Running experiment "{exp_name}"...')

    cfg = read_config(f'configs/{exp_name}.yml')

    method_params_grid = _eval_params_grid(cfg['methods']['ALIAS'])

    data_cfg = cfg['data']
    generator = getattr(datagen, data_cfg['generator'])
    data_params_grid = _eval_params_grid(data_cfg['params'])
    datasets = [{'params': params, 'samples': generator(**params)} for params in data_params_grid]

    results = []
    for method_params, dataset in product(method_params_grid, datasets):
        start_time = perf_counter()
        est_dag = ALIAS(**dataset['samples'], **method_params)
        end_time = perf_counter()
        results.append(_flatten_dict({'method.name': 'ALIAS', 'method.params': method_params, 'data': dataset, 'pred.raw': est_dag, 'runtime': end_time - start_time}))
    
    df = pd.DataFrame(results)
    analyzer = getattr(analyzers, cfg['analyzer'])
    analyzer(df)
    