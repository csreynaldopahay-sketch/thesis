"""
Main Analysis Pipeline
Runs the complete AMR resistance fingerprint analysis
"""

import os
import warnings
warnings.filterwarnings('ignore')

from data_ingestion import AMRDataLoader, AMRDataHarmonizer
from resistance_encoding import ResistanceFingerprintEncoder
from clustering import ResistancePatternClusterer, ResistanceVisualizer
from supervised_learning import SupervisedPatternRecognizer
from regional_analysis import RegionalPatternAnalyzer

import matplotlib.pyplot as plt


def create_output_directory():
    """Create output directory for results"""
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    return output_dir


def main():
    """Run complete analysis pipeline"""
    
    print("="*80)
    print("AMR RESISTANCE FINGERPRINT ANALYSIS SYSTEM")
    print("="*80)
    print("\nThis system performs exploratory pattern recognition on AMR data.")
    print("Not for clinical decision-making or prediction.\n")
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Phase 1: Data Loading and Preprocessing
    print("\n" + "="*80)
    print("PHASE 1: DATA LOADING AND PREPROCESSING")
    print("="*80)
    
    loader = AMRDataLoader(".")
    data = loader.load_all_csv_files()
    
    harmonizer = AMRDataHarmonizer(data)
    harmonized = harmonizer.harmonize()
    
    encoder = ResistanceFingerprintEncoder(harmonized)
    feature_matrix, metadata = encoder.create_resistance_fingerprints()
    
    print(f"\n✓ Data loading complete")
    print(f"  - Total isolates: {len(feature_matrix)}")
    print(f"  - Antibiotics tested: {len(feature_matrix.columns)}")
    print(f"  - MDR isolates: {metadata['MDR'].sum() if 'MDR' in metadata.columns else 'N/A'}")
    
    # Save processed data
    feature_matrix.to_csv(os.path.join(output_dir, "feature_matrix.csv"))
    metadata.to_csv(os.path.join(output_dir, "metadata.csv"))
    print(f"\n✓ Processed data saved to {output_dir}/")
    
    # Phase 2: Unsupervised Clustering
    print("\n" + "="*80)
    print("PHASE 2: UNSUPERVISED PATTERN RECOGNITION (CLUSTERING)")
    print("="*80)
    
    clusterer = ResistancePatternClusterer(feature_matrix, metadata)
    clusters = clusterer.perform_clustering(method='ward', n_clusters=5)
    
    summaries = clusterer.get_cluster_summaries()
    print("\nCluster Summaries:")
    print(summaries.to_string(index=False))
    
    # Save cluster summaries
    summaries.to_csv(os.path.join(output_dir, "cluster_summaries.csv"), index=False)
    
    # Create visualizations
    visualizer = ResistanceVisualizer(feature_matrix, metadata)
    
    print("\nGenerating visualizations...")
    fig_heatmap = visualizer.create_heatmap(
        linkage_matrix=clusterer.linkage_matrix,
        save_path=os.path.join(output_dir, "heatmap.png"),
        max_isolates=200
    )
    plt.close()
    
    fig_dendrogram = visualizer.create_dendrogram(
        clusterer.linkage_matrix,
        save_path=os.path.join(output_dir, "dendrogram.png")
    )
    plt.close()
    
    fig_distribution = visualizer.create_cluster_distribution_plot(
        summaries,
        save_path=os.path.join(output_dir, "cluster_distribution.png")
    )
    plt.close()
    
    print(f"✓ Clustering visualizations saved to {output_dir}/")
    
    # Phase 3: Supervised Pattern Differentiation
    print("\n" + "="*80)
    print("PHASE 3: SUPERVISED PATTERN DIFFERENTIATION")
    print("="*80)
    
    recognizer = SupervisedPatternRecognizer(feature_matrix, metadata)
    
    # Species pattern recognition
    species_results = recognizer.species_pattern_recognition()
    
    if species_results:
        fig_cm = recognizer.plot_confusion_matrix(
            species_results, 'Species',
            save_path=os.path.join(output_dir, "confusion_matrix_species.png")
        )
        plt.close()
        
        fig_fi = recognizer.plot_feature_importance(
            species_results, 'Species',
            save_path=os.path.join(output_dir, "feature_importance_species.png")
        )
        plt.close()
        
        # Save feature importance
        species_results['feature_importance'].to_csv(
            os.path.join(output_dir, "feature_importance_species.csv"),
            index=False
        )
        
        print(f"✓ Species pattern differentiation results saved to {output_dir}/")
    
    # MDR pattern recognition
    mdr_results = recognizer.mdr_pattern_recognition()
    
    if mdr_results:
        fig_cm = recognizer.plot_confusion_matrix(
            mdr_results, 'MDR',
            save_path=os.path.join(output_dir, "confusion_matrix_mdr.png")
        )
        plt.close()
        
        fig_fi = recognizer.plot_feature_importance(
            mdr_results, 'MDR',
            save_path=os.path.join(output_dir, "feature_importance_mdr.png")
        )
        plt.close()
        
        # Save feature importance
        mdr_results['feature_importance'].to_csv(
            os.path.join(output_dir, "feature_importance_mdr.csv"),
            index=False
        )
        
        print(f"✓ MDR pattern differentiation results saved to {output_dir}/")
    
    # Phase 4: Regional and Environmental Analysis
    print("\n" + "="*80)
    print("PHASE 4: REGIONAL & ENVIRONMENTAL ANALYSIS")
    print("="*80)
    
    # Use the cleaned feature matrix and metadata from clusterer
    clean_feature_matrix = clusterer.feature_matrix
    metadata_with_clusters = clusterer.metadata
    
    analyzer = RegionalPatternAnalyzer(clean_feature_matrix, metadata_with_clusters)
    
    # Perform PCA
    pca_components = analyzer.perform_pca(n_components=3)
    
    # Cluster distribution analysis
    regional_dist, env_dist = analyzer.analyze_cluster_distribution()
    
    if regional_dist is not None:
        regional_dist.to_csv(os.path.join(output_dir, "cluster_distribution_regional.csv"))
    
    if env_dist is not None:
        env_dist.to_csv(os.path.join(output_dir, "cluster_distribution_environmental.csv"))
    
    # Create PCA visualizations
    print("\nGenerating PCA visualizations...")
    
    fig = analyzer.plot_pca_by_region(
        save_path=os.path.join(output_dir, "pca_region.png")
    )
    if fig:
        plt.close()
    
    fig = analyzer.plot_pca_by_environment(
        save_path=os.path.join(output_dir, "pca_environment.png")
    )
    if fig:
        plt.close()
    
    fig = analyzer.plot_pca_by_species(
        save_path=os.path.join(output_dir, "pca_species.png")
    )
    if fig:
        plt.close()
    
    fig = analyzer.plot_pca_by_mdr(
        save_path=os.path.join(output_dir, "pca_mdr.png")
    )
    if fig:
        plt.close()
    
    print(f"✓ PCA visualizations saved to {output_dir}/")
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results have been saved to the '{output_dir}/' directory:")
    print("\nData Files:")
    print("  - feature_matrix.csv")
    print("  - metadata.csv")
    print("  - cluster_summaries.csv")
    print("  - feature_importance_species.csv")
    print("  - feature_importance_mdr.csv")
    print("  - cluster_distribution_regional.csv")
    print("  - cluster_distribution_environmental.csv")
    print("\nVisualization Files:")
    print("  - heatmap.png")
    print("  - dendrogram.png")
    print("  - cluster_distribution.png")
    print("  - confusion_matrix_species.png")
    print("  - feature_importance_species.png")
    print("  - confusion_matrix_mdr.png")
    print("  - feature_importance_mdr.png")
    print("  - pca_region.png")
    print("  - pca_environment.png")
    print("  - pca_species.png")
    print("  - pca_mdr.png")
    
    print("\n✓ Pipeline execution complete!")
    print("\nTo launch the interactive dashboard, run:")
    print("  streamlit run app.py")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
