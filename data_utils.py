import pandas as pd
import networkx as nx
from datetime import datetime


def load_temporal_data(filepath):
    """Load temporal edge data from a file."""
    df = pd.read_csv(filepath, sep=' ', header=None, names=['source', 'target', 'timestamp'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['date'] = df['datetime'].dt.date
    df = df.sort_values('timestamp')
    return df

 
def create_temporal_snapshots(df, window_size='1D'):
    """Create temporal graph snapshots from dataframe."""
    snapshots = []
    df_grouped = df.groupby(pd.Grouper(key='datetime', freq=window_size))
    for timestamp, group in df_grouped:
        if len(group) > 0:
            G = nx.from_pandas_edgelist(group, source='source', target='target', create_using=nx.DiGraph())
            snapshots.append({
                'timestamp': timestamp,
                'graph': G,
                'edges': group[['source', 'target']].values
            })
    return snapshots 