# Should this be a class??

import cudf
import cugraph

DECIDER_COL_NAME = "decider_id"
OTHER_COL_NAME = "other_id"
LIKE_COL_NAME = "interaction_type"
TIMESTAMP_COL_NAME = "timestamp"

def _build_graph(interactions_df: cudf.DataFrame, decider_col: str, other_col: str, like_col: str):
    # Build a directed graph from the interactions DataFrame
    graph = cugraph.Graph(directed=True)

    # Pick 'like' as edge weight
    graph.from_cudf_edgelist(
        interactions_df,
        source=decider_col,
        destination=other_col,
        edge_attr=like_col,  # only one column allowed
        store_transposed=True)
    
    return graph
    

def load_interactions(data_path: str, decider_col: str, other_col: str, like_col: str, timestamp_col: str):
    # Load interactions data from a CSV file into a cuDF DataFrame
    interactions = cudf.read_csv(data_path)

    # Select and rename relevant columns
    interactions = interactions[[decider_col, other_col, like_col, timestamp_col]]
    interactions = interactions.rename(columns={decider_col: DECIDER_COL_NAME, 
                                                other_col: OTHER_COL_NAME, 
                                                like_col: LIKE_COL_NAME,
                                                timestamp_col: TIMESTAMP_COL_NAME})
    
    # Convert timestamp column to datetime
    interactions = interactions[interactions['timestamp'].notnull()]
    interactions['timestamp'] = cudf.to_datetime(interactions['timestamp'])

    graph = _build_graph(interactions, decider_col=DECIDER_COL_NAME, other_col=OTHER_COL_NAME, like_col=LIKE_COL_NAME)

    return interactions, graph

