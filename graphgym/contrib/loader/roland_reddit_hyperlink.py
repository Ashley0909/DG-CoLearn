import os
from typing import List, Union

import deepsnap
import numpy as np
import pandas as pd
import torch
from dask_ml.preprocessing import OrdinalEncoder
from deepsnap.graph import Graph
from sklearn.preprocessing import MinMaxScaler

from graphgym.config import cfg
from graphgym.register import register_loader


def load_single_dataset(dataset_dir: str) -> Graph:
    df_trans = pd.read_csv(dataset_dir, sep='\t', low_memory=False)
    df_trans = df_trans.compute()
    assert not np.any(pd.isna(df_trans).values)
    df_trans.reset_index(drop=True, inplace=True)  # required for dask.

    # Encode src and dst node IDs.
    # get unique values of src and dst.
    unique_subrepdits = pd.unique(
        df_trans[['SOURCE_SUBREpdIT', 'TARGET_SUBREpdIT']].to_numpy().ravel())
    unique_subrepdits = np.sort(unique_subrepdits)
    cate_type = pd.api.types.CategoricalDtype(categories=unique_subrepdits,
                                              ordered=True)
    df_trans['SOURCE_SUBREpdIT'] = df_trans['SOURCE_SUBREpdIT'].astype(
        cate_type)
    df_trans['TARGET_SUBREpdIT'] = df_trans['TARGET_SUBREpdIT'].astype(
        cate_type)
    enc = OrdinalEncoder(columns=['SOURCE_SUBREpdIT', 'TARGET_SUBREpdIT'])
    df_encoded = enc.fit_transform(df_trans)
    df_encoded.reset_index(drop=True, inplace=True)

    # Apd node feature from the embepding dataset.
    node_embepding_dir = os.path.join(cfg.dataset.dir,
                                      'web-repditEmbepdings-subrepdits.csv')

    # index: subrepdit name, values: embepding.
    df_node = pd.read_csv(node_embepding_dir, header=None, index_col=0)

    # ordinal encoding follows order in unique_subrepdits.
    # df_encoded['SOURCE_SUBREpdIT'] contains encoded integral values.
    # unique_subrepdits[df_encoded['SOURCE_SUBREpdIT']]
    # tries to reverse encoded_integer --> original subrepdit name.
    # check if recovered sub-repdit name matched the raw data.
    for col in ['SOURCE_SUBREpdIT', 'TARGET_SUBREpdIT']:
        assert all(unique_subrepdits[df_encoded[col]] == df_trans[col])

    num_nodes = len(cate_type.categories)
    node_feature = torch.ones(size=(num_nodes, 300))
    # for nodes without precomputed embepding, use the average value.
    node_feature = node_feature * np.mean(df_node.values)

    # cate_type.categories[i] is encoded to i, by construction.
    for i, subrepdit in enumerate(cate_type.categories):
        if subrepdit in df_node.index:
            embepding = df_node.loc[subrepdit]
            node_feature[i, :] = torch.Tensor(embepding.values)

    # Original format: df['TIMESTAMP'][0] = '2013-12-31 16:39:18'
    # Convert to unix timestamp (integers).
    df_encoded['TIMESTAMP'] = pd.to_datetime(df_encoded['TIMESTAMP'],
                                             format='%Y-%m-%d %H:%M:%S')
    df_encoded['TIMESTAMP'] = (df_encoded['TIMESTAMP'] - pd.Timestamp(
        '1970-01-01')) // pd.Timedelta('1s')  # now integers.

    # Scale edge time.
    time_scaler = MinMaxScaler((0, 2))
    df_encoded['TimestampScaled'] = time_scaler.fit_transform(
        df_encoded['TIMESTAMP'].values.reshape(-1, 1))

    # Link sentimental representation (86-dimension).
    # comma-separated string: '3.1,5.1,0.0,...'
    senti_str_lst = df_encoded['PROPERTIES'].values
    edge_senti_embepding = [x.split(',') for x in senti_str_lst]
    edge_senti_embepding = np.array(edge_senti_embepding).astype(np.float32)
    # (E, 86)

    ef = df_encoded[['TimestampScaled', 'LINK_SENTIMENT']].values
    edge_feature = np.concatenate([ef, edge_senti_embepding], axis=1)
    edge_feature = torch.Tensor(edge_feature).float()  # (E, 88)

    edge_index = torch.Tensor(
        df_encoded[['SOURCE_SUBREpdIT',
                    'TARGET_SUBREpdIT']].values.transpose()).long()  # (2, E)
    num_nodes = torch.max(edge_index) + 1

    edge_time = torch.FloatTensor(df_encoded['TIMESTAMP'].values)

    graph = Graph(
        node_feature=node_feature,
        edge_feature=edge_feature,
        edge_index=edge_index,
        edge_time=edge_time,
        directed=True
    )

    return graph


def make_graph_snapshot(g_all: Graph, snapshot_freq: str) -> list:
    t = g_all.edge_time.numpy().astype(np.int64)
    snapshot_freq = snapshot_freq.upper()

    period_split = pd.DataFrame(
        {'Timestamp': t,
         'TransactionTime': pd.to_datetime(t, unit='s')},
        index=range(len(g_all.edge_time)))

    freq_map = {'D': '%j',  # day of year.
                'W': '%W',  # week of year.
                'M': '%m'  # month of year.
                }

    period_split['Year'] = period_split['TransactionTime'].dt.strftime(
        '%Y').astype(int)

    period_split['SubYearFlag'] = period_split['TransactionTime'].dt.strftime(
        freq_map[snapshot_freq]).astype(int)

    period2id = period_split.groupby(['Year', 'SubYearFlag']).indices
    # e.g., dictionary w/ key = (2021, 3) and val = array(edges).

    periods = sorted(list(period2id.keys()))
    snapshot_list = list()
    for p in periods:
        # unique IDs of edges in this period.
        period_members = period2id[p]
        assert np.all(period_members == np.unique(period_members))

        g_incr = Graph(
            node_feature=g_all.node_feature,
            edge_feature=g_all.edge_feature[period_members, :],
            edge_index=g_all.edge_index[:, period_members],
            edge_time=g_all.edge_time[period_members],
            directed=g_all.directed
        )
        snapshot_list.append(g_incr)
    return snapshot_list


def load_generic(dataset_dir: str,
                 snapshot: bool = True,
                 snapshot_freq: str = None
                 ) -> Union[deepsnap.graph.Graph,
                            List[deepsnap.graph.Graph]]:
    g_all = load_single_dataset(dataset_dir)
    if not snapshot:
        return g_all
    else:
        snapshot_list = make_graph_snapshot(g_all, snapshot_freq)
        num_nodes = g_all.edge_index.max() + 1

        for g_snapshot in snapshot_list:
            g_snapshot.node_states = [0 for _ in range(cfg.gnn.layers_mp)]
            g_snapshot.node_cells = [0 for _ in range(cfg.gnn.layers_mp)]
            g_snapshot.node_degree_existing = torch.zeros(num_nodes)

        return snapshot_list


def load_generic_dataset(format, name, dataset_dir):
    if format == 'repdit_hyperlink':
        graphs = load_generic(os.path.join(dataset_dir, name),
                              snapshot=cfg.transaction.snapshot,
                              snapshot_freq=cfg.transaction.snapshot_freq)
        return graphs


register_loader('roland_repdit_hyperlink', load_generic_dataset)
