"""
Regional & Environmental Pattern Analysis Module
Implements PCA and regional/environmental distribution analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class RegionalPatternAnalyzer:
    """
    Analyze resistance patterns across regions and environments
    """
    
    def __init__(self, feature_matrix: pd.DataFrame, metadata: pd.DataFrame):
        """
        Initialize analyzer
        
        Args:
            feature_matrix: Resistance fingerprints
            metadata: Isolate metadata
        """
        self.feature_matrix = feature_matrix.copy()
        self.metadata = metadata.copy()
        
        # Clean data
        self._clean_data()
        
        self.pca = None
        self.pca_components = None
        self.scaler = None
    
    def _clean_data(self):
        """Remove rows with no valid data"""
        valid_rows = self.feature_matrix.notna().any(axis=1)
        self.feature_matrix = self.feature_matrix[valid_rows]
        self.metadata = self.metadata[valid_rows]
        
        self.feature_matrix.reset_index(drop=True, inplace=True)
        self.metadata.reset_index(drop=True, inplace=True)
    
    def perform_pca(self, n_components=3):
        """
        Perform PCA on resistance fingerprints
        
        Args:
            n_components: Number of principal components
            
        Returns:
            PCA-transformed data
        """
        print("\nPerforming PCA analysis...")
        
        # Prepare data
        X = self.feature_matrix.fillna(-1)  # -1 for not tested
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform PCA
        self.pca = PCA(n_components=n_components)
        self.pca_components = self.pca.fit_transform(X_scaled)
        
        # Add PCA components to metadata
        for i in range(n_components):
            self.metadata[f'PC{i+1}'] = self.pca_components[:, i]
        
        # Print explained variance
        print(f"\nExplained variance ratio:")
        for i, var in enumerate(self.pca.explained_variance_ratio_):
            print(f"PC{i+1}: {var:.3f} ({var*100:.1f}%)")
        print(f"Total variance explained: {self.pca.explained_variance_ratio_.sum():.3f} "
              f"({self.pca.explained_variance_ratio_.sum()*100:.1f}%)")
        
        return self.pca_components
    
    def analyze_cluster_distribution(self, cluster_column='Cluster'):
        """
        Analyze how clusters are distributed across regions and environments
        
        Args:
            cluster_column: Name of cluster column in metadata
            
        Returns:
            Distribution DataFrames
        """
        if cluster_column not in self.metadata.columns:
            print(f"Warning: {cluster_column} not found in metadata")
            return None, None
        
        print("\nAnalyzing cluster distribution...")
        
        # Regional distribution
        regional_dist = None
        if 'Region' in self.metadata.columns:
            regional_dist = pd.crosstab(
                self.metadata[cluster_column],
                self.metadata['Region'],
                normalize='index'
            )
            print("\nCluster distribution by Region:")
            print(regional_dist.to_string())
        
        # Environmental distribution
        env_dist = None
        if 'Environment' in self.metadata.columns:
            env_dist = pd.crosstab(
                self.metadata[cluster_column],
                self.metadata['Environment'],
                normalize='index'
            )
            print("\nCluster distribution by Environment:")
            print(env_dist.to_string())
        
        return regional_dist, env_dist
    
    def plot_pca_by_region(self, save_path=None):
        """
        Plot PCA colored by region
        
        Args:
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        if self.pca_components is None:
            print("Must perform PCA first")
            return None
        
        if 'Region' not in self.metadata.columns:
            print("No region information available")
            return None
        
        print("\nCreating PCA plot by region...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get unique regions
        regions = self.metadata['Region'].dropna().unique()
        colors = sns.color_palette('Set2', len(regions))
        
        for i, region in enumerate(regions):
            mask = self.metadata['Region'] == region
            ax.scatter(self.pca_components[mask, 0], 
                      self.pca_components[mask, 1],
                      label=region, alpha=0.6, s=50, color=colors[i])
        
        ax.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_title('PCA of Resistance Patterns - Regional Separation', 
                    fontsize=14, fontweight='bold')
        ax.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PCA plot saved to {save_path}")
        
        return fig
    
    def plot_pca_by_environment(self, save_path=None):
        """
        Plot PCA colored by environment
        
        Args:
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        if self.pca_components is None:
            print("Must perform PCA first")
            return None
        
        if 'Environment' not in self.metadata.columns:
            print("No environment information available")
            return None
        
        print("\nCreating PCA plot by environment...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get unique environments
        environments = self.metadata['Environment'].dropna().unique()
        colors = sns.color_palette('Set3', len(environments))
        
        for i, env in enumerate(environments):
            mask = self.metadata['Environment'] == env
            ax.scatter(self.pca_components[mask, 0], 
                      self.pca_components[mask, 1],
                      label=env, alpha=0.6, s=50, color=colors[i])
        
        ax.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_title('PCA of Resistance Patterns - Environmental Overlap', 
                    fontsize=14, fontweight='bold')
        ax.legend(title='Environment', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PCA plot saved to {save_path}")
        
        return fig
    
    def plot_pca_by_species(self, save_path=None):
        """
        Plot PCA colored by species
        
        Args:
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        if self.pca_components is None:
            print("Must perform PCA first")
            return None
        
        if 'Species_Standardized' not in self.metadata.columns:
            print("No species information available")
            return None
        
        print("\nCreating PCA plot by species...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get unique species
        species = self.metadata['Species_Standardized'].dropna().unique()
        species = [s for s in species if s != 'Unknown']
        colors = sns.color_palette('husl', len(species))
        
        for i, sp in enumerate(species):
            mask = self.metadata['Species_Standardized'] == sp
            ax.scatter(self.pca_components[mask, 0], 
                      self.pca_components[mask, 1],
                      label=sp, alpha=0.6, s=50, color=colors[i])
        
        ax.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_title('PCA of Resistance Patterns - Species Pattern Proximity', 
                    fontsize=14, fontweight='bold')
        ax.legend(title='Species', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PCA plot saved to {save_path}")
        
        return fig
    
    def plot_pca_by_mdr(self, save_path=None):
        """
        Plot PCA colored by MDR status
        
        Args:
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        if self.pca_components is None:
            print("Must perform PCA first")
            return None
        
        if 'MDR' not in self.metadata.columns:
            print("No MDR information available")
            return None
        
        print("\nCreating PCA plot by MDR status...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot non-MDR and MDR separately
        mdr_mask = self.metadata['MDR'] == 1
        non_mdr_mask = self.metadata['MDR'] == 0
        
        ax.scatter(self.pca_components[non_mdr_mask, 0], 
                  self.pca_components[non_mdr_mask, 1],
                  label='Non-MDR', alpha=0.5, s=50, color='steelblue')
        ax.scatter(self.pca_components[mdr_mask, 0], 
                  self.pca_components[mdr_mask, 1],
                  label='MDR', alpha=0.7, s=50, color='red')
        
        ax.set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_title('PCA of Resistance Patterns - MDR Separation', 
                    fontsize=14, fontweight='bold')
        ax.legend(title='MDR Status', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PCA plot saved to {save_path}")
        
        return fig


if __name__ == "__main__":
    # Test the regional analysis
    from data_ingestion import AMRDataLoader, AMRDataHarmonizer
    from resistance_encoding import ResistanceFingerprintEncoder
    from clustering import ResistancePatternClusterer
    
    # Load and process data
    loader = AMRDataLoader(".")
    data = loader.load_all_csv_files()
    
    harmonizer = AMRDataHarmonizer(data)
    harmonized = harmonizer.harmonize()
    
    encoder = ResistanceFingerprintEncoder(harmonized)
    feature_matrix, metadata = encoder.create_resistance_fingerprints()
    
    # Perform clustering first
    clusterer = ResistancePatternClusterer(feature_matrix, metadata)
    clusters = clusterer.perform_clustering(n_clusters=5)
    
    # Update metadata with clusters
    metadata = clusterer.metadata
    
    # Perform regional analysis
    analyzer = RegionalPatternAnalyzer(feature_matrix, metadata)
    
    # PCA
    pca_components = analyzer.perform_pca(n_components=3)
    
    # Cluster distribution
    regional_dist, env_dist = analyzer.analyze_cluster_distribution()
    
    # Create visualizations
    fig_region = analyzer.plot_pca_by_region(save_path='pca_region.png')
    fig_env = analyzer.plot_pca_by_environment(save_path='pca_environment.png')
    fig_species = analyzer.plot_pca_by_species(save_path='pca_species.png')
    fig_mdr = analyzer.plot_pca_by_mdr(save_path='pca_mdr.png')
    
    plt.show()
    
    print("\nRegional and environmental analysis complete!")
