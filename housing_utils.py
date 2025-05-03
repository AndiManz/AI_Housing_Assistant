# housing_utils.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def apply_clustering(df, features, n_clusters=3, random_state=42):
    """
    Applies KMeans clustering to the input dataframe based on selected features.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input dataframe containing housing data.
    features : list of str
        List of column names to use for clustering (e.g., ['rent', 'size', etc.]).
    n_clusters : int, optional (default=3)
        Number of clusters to form.
    random_state : int, optional (default=42)
        Random seed for reproducibility.

    Returns:
    -------
    df : pandas.DataFrame
        Original dataframe with two new columns:
        - 'cluster': Numeric cluster label
        - 'cluster_label': Human-readable category label
    """

    # Step 1: Scale the selected features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    # Step 2: Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Step 3: Map numeric clusters to human-readable labels
    cluster_labels = {
        0: "Moderate Budget",
        1: "High-Cost, Low Convenience",
        2: "Student Friendly"
    }
    df['cluster_label'] = df['cluster'].map(cluster_labels)

    return df
