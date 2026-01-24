"""
Social Network Analyzer - Stage 3b of pipeline.

Calculates social network metrics from GBI matrices:
- Node-level metrics (centrality, strength, etc.)
- Network-level metrics (density, modularity, etc.)

Uses Simple Ratio Index (SRI) for edge weights.

Equivalent to R script: 3a_create_MOVEBOUT_GBI_sn_node_net.R
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, Tuple
import networkx as nx

from ..config import RFIDConfig


class SocialNetworkAnalyzer:
    """
    Analyzer for social networks from GBI matrices.

    Calculates network metrics using NetworkX.
    """

    def __init__(self, config: RFIDConfig):
        """
        Initialize social network analyzer with configuration.

        Args:
            config: RFID configuration object
        """
        self.config = config

    def analyze_networks(
        self,
        gbi_dict: Dict[str, pd.DataFrame],
        metadata_df: pd.DataFrame,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Analyze social networks from GBI matrices.

        Args:
            gbi_dict: Dictionary mapping trial_id to GBI DataFrame
            metadata_df: Metadata DataFrame with animal information
            progress_callback: Optional callback function(message: str)

        Returns:
            Tuple of (node_stats_df, net_stats_df)

        Raises:
            ValueError: If GBI matrices are invalid
        """
        if progress_callback:
            progress_callback("Analyzing social networks...")

        all_node_stats = []
        all_net_stats = []

        for trial_id, gbi_df in gbi_dict.items():
            if progress_callback:
                progress_callback(f"Processing network for trial: {trial_id}")

            if len(gbi_df) == 0:
                if progress_callback:
                    progress_callback(f"Warning: Empty GBI for {trial_id}")
                continue

            # Get metadata for this trial
            trial_metadata = metadata_df[metadata_df['trial'] == trial_id]

            # Calculate adjacency matrix using SRI
            adjacency_matrix, animals = self._calculate_sri_matrix(gbi_df)

            if adjacency_matrix is None:
                continue

            # Create network graph
            G = self._create_network_graph(adjacency_matrix, animals)

            # Calculate node-level metrics
            node_stats = self._calculate_node_metrics(G, trial_id, trial_metadata, adjacency_matrix, animals)
            all_node_stats.append(node_stats)

            # Calculate network-level metrics
            net_stats = self._calculate_network_metrics(G, trial_id)
            all_net_stats.append(net_stats)

        # Combine all results
        if all_node_stats:
            node_stats_df = pd.concat(all_node_stats, ignore_index=True)
        else:
            node_stats_df = pd.DataFrame()

        if all_net_stats:
            net_stats_df = pd.DataFrame(all_net_stats)
        else:
            net_stats_df = pd.DataFrame()

        # Save outputs
        self._save_outputs(node_stats_df, net_stats_df, progress_callback)

        return node_stats_df, net_stats_df

    def _calculate_sri_matrix(
        self,
        gbi_df: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], Optional[list]]:
        """
        Calculate Simple Ratio Index (SRI) adjacency matrix from GBI.

        SRI(A,B) = Number of times A and B seen together / Number of times A or B seen

        Args:
            gbi_df: GBI DataFrame

        Returns:
            Tuple of (adjacency_matrix, animal_list) or (None, None) if invalid
        """
        # Get animal columns (exclude metadata columns)
        metadata_cols = ['zone_id', 'start_time', 'end_time', 'center_time',
                        'duration', 'group_size', 'm_sum', 'f_sum', 'mf_sum']

        animal_cols = [col for col in gbi_df.columns if col not in metadata_cols]

        if len(animal_cols) == 0:
            return None, None

        # Extract binary presence matrix
        gbi_matrix = gbi_df[animal_cols].values

        # Calculate SRI
        n_animals = len(animal_cols)
        adjacency = np.zeros((n_animals, n_animals))

        for i in range(n_animals):
            for j in range(i+1, n_animals):
                # Count co-occurrences (both present)
                both_present = np.sum((gbi_matrix[:, i] == 1) & (gbi_matrix[:, j] == 1))

                # Count when either is present
                either_present = np.sum((gbi_matrix[:, i] == 1) | (gbi_matrix[:, j] == 1))

                # Calculate SRI
                if either_present > 0:
                    sri = both_present / either_present
                    adjacency[i, j] = sri
                    adjacency[j, i] = sri  # Symmetric

        return adjacency, animal_cols

    def _create_network_graph(
        self,
        adjacency_matrix: np.ndarray,
        animals: list
    ) -> nx.Graph:
        """
        Create NetworkX graph from adjacency matrix.

        Args:
            adjacency_matrix: SRI adjacency matrix
            animals: List of animal names

        Returns:
            NetworkX Graph object
        """
        G = nx.Graph()

        # Add nodes
        G.add_nodes_from(animals)

        # Add edges (only if weight > 0)
        n_animals = len(animals)
        for i in range(n_animals):
            for j in range(i+1, n_animals):
                weight = adjacency_matrix[i, j]
                if weight > 0:
                    G.add_edge(animals[i], animals[j], weight=weight)

        return G

    def _calculate_node_metrics(
        self,
        G: nx.Graph,
        trial_id: str,
        metadata_df: pd.DataFrame,
        adjacency_matrix: np.ndarray,
        animals: list
    ) -> pd.DataFrame:
        """
        Calculate node-level network metrics.

        Args:
            G: NetworkX graph
            trial_id: Trial identifier
            metadata_df: Metadata for this trial
            adjacency_matrix: SRI adjacency matrix
            animals: List of animal names

        Returns:
            DataFrame with node metrics
        """
        metrics = []

        # Get sex information for opposite-sex calculations
        animal_sex = metadata_df.set_index('name')['sex'].to_dict()

        for i, animal in enumerate(animals):
            # Edge strength (sum of edge weights)
            edge_strength = np.sum(adjacency_matrix[i, :])

            # Degree centrality
            degree_cent = nx.degree_centrality(G)[animal] if animal in G else 0

            # Eigenvector centrality (may fail for disconnected graphs)
            try:
                eigen_cent = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)[animal]
            except:
                eigen_cent = 0

            # Betweenness centrality
            try:
                between_cent = nx.betweenness_centrality(G, weight='weight')[animal]
            except:
                between_cent = 0

            # Closeness centrality
            try:
                close_cent = nx.closeness_centrality(G, distance='weight')[animal]
            except:
                close_cent = 0

            # PageRank
            try:
                pagerank = nx.pagerank(G, weight='weight')[animal]
            except:
                pagerank = 0

            # Authority score (HITS algorithm)
            try:
                authority = nx.hits(G)[1][animal]  # [1] is authorities
            except:
                authority = 0

            # Opposite-sex edge strength
            animal_sex_val = animal_sex.get(animal, '')
            opposite_sex = 'F' if animal_sex_val == 'M' else 'M'

            opposite_sex_strength = 0
            for j, other_animal in enumerate(animals):
                if animal_sex.get(other_animal, '') == opposite_sex:
                    opposite_sex_strength += adjacency_matrix[i, j]

            # Compile metrics
            node_metrics = {
                'trial': trial_id,
                'name': animal,
                'sex': animal_sex_val,
                'edge_strength': edge_strength,
                'degree_centrality': degree_cent,
                'eigenvector_centrality': eigen_cent,
                'betweenness_centrality': between_cent,
                'closeness_centrality': close_cent,
                'pagerank': pagerank,
                'authority': authority,
                'opposite_sex_edge_strength': opposite_sex_strength
            }

            metrics.append(node_metrics)

        return pd.DataFrame(metrics)

    def _calculate_network_metrics(
        self,
        G: nx.Graph,
        trial_id: str
    ) -> dict:
        """
        Calculate network-level metrics.

        Args:
            G: NetworkX graph
            trial_id: Trial identifier

        Returns:
            Dictionary of network metrics
        """
        metrics = {'trial': trial_id}

        # Number of nodes and edges
        metrics['num_nodes'] = G.number_of_nodes()
        metrics['num_edges'] = G.number_of_edges()

        # Density
        try:
            metrics['density'] = nx.density(G)
        except:
            metrics['density'] = 0

        # Transitivity (clustering coefficient)
        try:
            metrics['transitivity'] = nx.transitivity(G)
        except:
            metrics['transitivity'] = 0

        # Degree centralization
        try:
            degree_cents = list(nx.degree_centrality(G).values())
            if degree_cents:
                max_cent = max(degree_cents)
                sum_diff = sum(max_cent - c for c in degree_cents)
                n = len(degree_cents)
                metrics['degree_centralization'] = sum_diff / ((n - 1) * (n - 2)) if n > 2 else 0
            else:
                metrics['degree_centralization'] = 0
        except:
            metrics['degree_centralization'] = 0

        # Mean distance (average shortest path length)
        try:
            if nx.is_connected(G):
                metrics['mean_distance'] = nx.average_shortest_path_length(G, weight='weight')
            else:
                # For disconnected graphs, calculate for largest component
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                metrics['mean_distance'] = nx.average_shortest_path_length(subgraph, weight='weight')
        except:
            metrics['mean_distance'] = np.nan

        # Modularity (community detection)
        try:
            communities = nx.community.greedy_modularity_communities(G, weight='weight')
            metrics['modularity'] = nx.community.modularity(G, communities, weight='weight')
            metrics['num_communities'] = len(communities)
        except:
            metrics['modularity'] = np.nan
            metrics['num_communities'] = 0

        return metrics

    def _save_outputs(
        self,
        node_stats_df: pd.DataFrame,
        net_stats_df: pd.DataFrame,
        progress_callback: Optional[Callable] = None
    ) -> None:
        """
        Save network analysis outputs.

        Args:
            node_stats_df: Node-level statistics
            net_stats_df: Network-level statistics
            progress_callback: Optional callback function
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save node stats
        if len(node_stats_df) > 0:
            node_path = output_dir / "ALLTRIAL_SNA_node_stats.csv"
            node_stats_df.to_csv(node_path, index=False)

            if progress_callback:
                progress_callback(f"Saved node statistics: {len(node_stats_df)} nodes")

        # Save network stats
        if len(net_stats_df) > 0:
            net_path = output_dir / "ALLTRIAL_SNA_net_stats.csv"
            net_stats_df.to_csv(net_path, index=False)

            if progress_callback:
                progress_callback(f"Saved network statistics: {len(net_stats_df)} networks")
