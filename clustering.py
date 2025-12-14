"""
Unsupervised Pattern Recognition Module
Implements hierarchical clustering and visualization for AMR resistance patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class ResistancePatternClusterer:
    """
    Perform hierarchical clustering on resistance fingerprints
    """
    
    def __init__(self, feature_matrix: pd.DataFrame, metadata: pd.DataFrame):
        """
        Initialize clusterer
        
        Args:
            feature_matrix: Resistance fingerprints (isolates × antibiotics)
            metadata: Isolate metadata
        """
        self.feature_matrix = feature_matrix.copy()
        self.metadata = metadata.copy()
        self.linkage_matrix = None
        self.clusters = None
        self.n_clusters = None
        
        # Remove rows with all NaN values
        self._clean_data()
    
    def _clean_data(self):
        """Remove rows with all NaN or no valid data"""
        # Keep only rows with at least some resistance data
        valid_rows = self.feature_matrix.notna().any(axis=1)
        self.feature_matrix = self.feature_matrix[valid_rows]
        self.metadata = self.metadata[valid_rows]
        
        # Reset indices
        self.feature_matrix.reset_index(drop=True, inplace=True)
        self.metadata.reset_index(drop=True, inplace=True)
        
        print(f"Clean data: {len(self.feature_matrix)} isolates with resistance data")
    
    def perform_clustering(self, method='ward', metric='euclidean', n_clusters=5):
        """
        Perform hierarchical clustering
        
        Args:
            method: Linkage method ('ward', 'average', 'complete', 'single')
            metric: Distance metric ('euclidean', 'manhattan', 'cosine')
            n_clusters: Number of clusters to create
            
        Returns:
            Cluster assignments
        """
        print(f"\nPerforming hierarchical clustering...")
        print(f"Method: {method}, Metric: {metric}, Clusters: {n_clusters}")
        
        # Fill NaN values with -1 (indicating not tested)
        # This allows us to cluster based on available data
        data_for_clustering = self.feature_matrix.fillna(-1)
        
        # Perform hierarchical clustering
        if method == 'ward':
            # Ward requires euclidean distance
            self.linkage_matrix = linkage(data_for_clustering, method=method)
        else:
            self.linkage_matrix = linkage(data_for_clustering, method=method, metric=metric)
        
        # Get cluster assignments
        self.n_clusters = n_clusters
        self.clusters = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
        
        # Add cluster assignments to metadata
        self.metadata['Cluster'] = self.clusters
        
        print(f"Clustering complete. Cluster distribution:")
        print(self.metadata['Cluster'].value_counts().sort_index())
        
        return self.clusters
    
    def get_cluster_summaries(self) -> pd.DataFrame:
        """
        Generate cluster summaries
        
        Returns:
            DataFrame with cluster characteristics
        """
        if self.clusters is None:
            raise ValueError("Must perform clustering first")
        
        summaries = []
        
        for cluster_id in range(1, self.n_clusters + 1):
            cluster_mask = self.metadata['Cluster'] == cluster_id
            cluster_data = self.feature_matrix[cluster_mask]
            cluster_meta = self.metadata[cluster_mask]
            
            # Calculate statistics
            n_isolates = len(cluster_data)
            
            # MDR proportion
            if 'MDR' in cluster_meta.columns:
                mdr_proportion = cluster_meta['MDR'].sum() / n_isolates
            else:
                mdr_proportion = 0
            
            # Species composition
            if 'Species_Standardized' in cluster_meta.columns:
                species_counts = cluster_meta['Species_Standardized'].value_counts()
                top_species = species_counts.index[0] if len(species_counts) > 0 else 'Unknown'
                species_diversity = len(species_counts)
            else:
                top_species = 'Unknown'
                species_diversity = 0
            
            # Region distribution
            if 'Region' in cluster_meta.columns:
                region_counts = cluster_meta['Region'].value_counts()
                top_region = region_counts.index[0] if len(region_counts) > 0 else 'Unknown'
            else:
                top_region = 'Unknown'
            
            # Environment distribution
            if 'Environment' in cluster_meta.columns:
                env_counts = cluster_meta['Environment'].value_counts()
                top_environment = env_counts.index[0] if len(env_counts) > 0 else 'Unknown'
            else:
                top_environment = 'Unknown'
            
            # Average resistance count
            avg_resistance = (cluster_data == 2).sum(axis=1).mean()
            
            summaries.append({
                'Cluster': cluster_id,
                'N_Isolates': n_isolates,
                'MDR_Proportion': mdr_proportion,
                'Avg_Resistance_Count': avg_resistance,
                'Top_Species': top_species,
                'Species_Diversity': species_diversity,
                'Top_Region': top_region,
                'Top_Environment': top_environment
            })
        
        return pd.DataFrame(summaries)


class ResistanceVisualizer:
    """
    Create visualizations for resistance patterns
    """
    
    def __init__(self, feature_matrix: pd.DataFrame, metadata: pd.DataFrame):
        """
        Initialize visualizer
        
        Args:
            feature_matrix: Resistance fingerprints
            metadata: Isolate metadata
        """
        self.feature_matrix = feature_matrix.copy()
        self.metadata = metadata.copy()
        
        # Set visualization style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100
    
    def create_heatmap(self, linkage_matrix=None, save_path=None, max_isolates=200):
        """
        Create resistance heatmap
        
        Args:
            linkage_matrix: Optional linkage matrix for ordering
            save_path: Path to save figure
            max_isolates: Maximum number of isolates to display
            
        Returns:
            Figure object
        """
        print("\nCreating resistance heatmap...")
        
        # Limit number of isolates for visualization
        if len(self.feature_matrix) > max_isolates:
            print(f"Sampling {max_isolates} isolates for visualization")
            sample_idx = np.random.choice(len(self.feature_matrix), max_isolates, replace=False)
            data = self.feature_matrix.iloc[sample_idx]
            metadata = self.metadata.iloc[sample_idx]
        else:
            data = self.feature_matrix
            metadata = self.metadata
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create custom colormap
        # -1 = not tested (white), 0 = susceptible (green), 1 = intermediate (yellow), 2 = resistant (red)
        colors = ['white', 'green', 'yellow', 'red']
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(colors)
        
        # Prepare data (fill NaN with -1)
        heatmap_data = data.fillna(-1)
        
        # Create heatmap
        sns.heatmap(heatmap_data, cmap=cmap, vmin=-1, vmax=2,
                   cbar_kws={'label': 'Resistance Level', 
                            'ticks': [-1, 0, 1, 2],
                            'format': 'S'},
                   xticklabels=True, yticklabels=False,
                   ax=ax)
        
        # Customize colorbar labels
        cbar = ax.collections[0].colorbar
        cbar.set_ticklabels(['Not Tested', 'Susceptible', 'Intermediate', 'Resistant'])
        
        ax.set_xlabel('Antibiotics', fontsize=12)
        ax.set_ylabel('Isolates', fontsize=12)
        ax.set_title('AMR Resistance Fingerprint Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        
        return fig
    
    def create_dendrogram(self, linkage_matrix, save_path=None):
        """
        Create dendrogram for hierarchical clustering
        
        Args:
            linkage_matrix: Linkage matrix from clustering
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        print("\nCreating dendrogram...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        dendrogram(linkage_matrix, ax=ax, 
                  truncate_mode='lastp', p=30,
                  show_leaf_counts=True,
                  leaf_font_size=10)
        
        ax.set_xlabel('Cluster Size', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        ax.set_title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dendrogram saved to {save_path}")
        
        return fig
    
    def create_cluster_distribution_plot(self, cluster_summaries: pd.DataFrame, save_path=None):
        """
        Create visualization of cluster characteristics
        
        Args:
            cluster_summaries: DataFrame from get_cluster_summaries()
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        print("\nCreating cluster distribution plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Isolate count per cluster
        axes[0, 0].bar(cluster_summaries['Cluster'], cluster_summaries['N_Isolates'], 
                       color='steelblue')
        axes[0, 0].set_xlabel('Cluster')
        axes[0, 0].set_ylabel('Number of Isolates')
        axes[0, 0].set_title('Isolate Distribution Across Clusters')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Plot 2: MDR proportion per cluster
        axes[0, 1].bar(cluster_summaries['Cluster'], cluster_summaries['MDR_Proportion'], 
                       color='coral')
        axes[0, 1].set_xlabel('Cluster')
        axes[0, 1].set_ylabel('MDR Proportion')
        axes[0, 1].set_title('MDR Proportion by Cluster')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Plot 3: Average resistance count
        axes[1, 0].bar(cluster_summaries['Cluster'], cluster_summaries['Avg_Resistance_Count'], 
                       color='forestgreen')
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Average Resistance Count')
        axes[1, 0].set_title('Average Number of Resistances per Cluster')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Plot 4: Species diversity
        axes[1, 1].bar(cluster_summaries['Cluster'], cluster_summaries['Species_Diversity'], 
                       color='purple')
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Number of Species')
        axes[1, 1].set_title('Species Diversity by Cluster')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cluster distribution plot saved to {save_path}")
        
        return fig


if __name__ == "__main__":
    # Test the clustering
    from data_ingestion import AMRDataLoader, AMRDataHarmonizer
    from resistance_encoding import ResistanceFingerprintEncoder
    
    # Load and process data
    loader = AMRDataLoader(".")
    data = loader.load_all_csv_files()
    
    harmonizer = AMRDataHarmonizer(data)
    harmonized = harmonizer.harmonize()
    
    encoder = ResistanceFingerprintEncoder(harmonized)
    feature_matrix, metadata = encoder.create_resistance_fingerprints()
    
    # Perform clustering
    clusterer = ResistancePatternClusterer(feature_matrix, metadata)
    clusters = clusterer.perform_clustering(method='ward', n_clusters=5)
    
    # Get cluster summaries
    summaries = clusterer.get_cluster_summaries()
    print("\nCluster Summaries:")
    print(summaries.to_string(index=False))
    
    # Create visualizations
    visualizer = ResistanceVisualizer(feature_matrix, metadata)
    
    # Create heatmap
    fig_heatmap = visualizer.create_heatmap(
        linkage_matrix=clusterer.linkage_matrix,
        save_path='heatmap.png'
    )
    
    # Create dendrogram
    fig_dendrogram = visualizer.create_dendrogram(
        clusterer.linkage_matrix,
        save_path='dendrogram.png'
    )
    
    # Create cluster distribution plot
    fig_distribution = visualizer.create_cluster_distribution_plot(
        summaries,
        save_path='cluster_distribution.png'
    )
    
    plt.show()
    
    print("\nClustering and visualization complete!")
