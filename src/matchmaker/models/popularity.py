import cugraph

def get_pagerank(graph, alpha=0.85, max_iter=100, tol=1e-06):
    """
    Calculate PageRank scores for the nodes in the graph using cuGraph.

    Parameters:
    graph (cugraph.Graph): The input graph.
    alpha (float): Damping factor for PageRank.
    max_iter (int): Maximum number of iterations.
    tol (float): Tolerance for convergence.

    Returns:
    cudf.DataFrame: DataFrame containing nodes and their PageRank scores.
    """
    pagerank_df = cugraph.pagerank(graph, alpha=alpha, max_iter=max_iter, tol=tol)
    pagerank_df = pagerank_df.rename(columns={'vertex': 'user_id'})
    return pagerank_df

def get_like_stats(interactions_df, decider_col, other_col, like_col):
    """
    Returns:
    cudf.DataFrame: DataFrame containing user IDs and their like ratios.
    """
    # Likes received per user
    likes_received = interactions_df.groupby(other_col)[like_col].sum().reset_index()
    likes_received = likes_received.rename(columns={other_col:'user_id', like_col:'likes_received'})

    # Likes given per user
    likes_given = interactions_df.groupby(decider_col)[like_col].sum().reset_index()
    likes_given = likes_given.rename(columns={decider_col:'user_id', like_col:'likes_given'})

    # Merge like stats together
    like_stats_df = likes_received.merge(likes_given, on='user_id', how='outer')

    # Fill NaNs with 0
    like_stats_df = like_stats_df.fillna(0)

    # Compute like ratio (add small epsilon to avoid div by 0)
    like_stats_df['like_ratio'] = like_stats_df['likes_received'] / (like_stats_df['likes_given'] + 1e-6)

    return like_stats_df[['user_id', 'likes_received', 'likes_given', 'like_ratio']]
