"""
AMR Resistance Pattern Explorer - Streamlit App
Interactive dashboard for exploring antimicrobial resistance patterns
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Import our modules
from data_ingestion import AMRDataLoader, AMRDataHarmonizer
from resistance_encoding import ResistanceFingerprintEncoder
from clustering import ResistancePatternClusterer, ResistanceVisualizer
from supervised_learning import SupervisedPatternRecognizer
from regional_analysis import RegionalPatternAnalyzer


# Page configuration
st.set_page_config(
    page_title="AMR Pattern Explorer",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
    }
    .disclaimer {
        background-color: #fff3cd;
        padding: 15px;
        border-left: 5px solid #ffc107;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_default_data():
    """Load default CSV files from directory"""
    try:
        loader = AMRDataLoader(".")
        data = loader.load_all_csv_files()
        
        harmonizer = AMRDataHarmonizer(data)
        harmonized = harmonizer.harmonize()
        
        encoder = ResistanceFingerprintEncoder(harmonized)
        feature_matrix, metadata = encoder.create_resistance_fingerprints()
        
        return feature_matrix, metadata
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None


def main():
    """Main application"""
    
    # Header
    st.markdown('<p class="main-header">🦠 AMR Resistance Pattern Explorer</p>', 
                unsafe_allow_html=True)
    st.markdown("### Interactive Dashboard for Antimicrobial Resistance Pattern Analysis")
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ IMPORTANT DISCLAIMER</strong><br>
        This system supports <strong>exploratory pattern recognition</strong> and 
        <strong>AMR surveillance research</strong>. It is NOT designed for:
        <ul>
            <li>Clinical decision-making</li>
            <li>Predictive forecasting</li>
            <li>Treatment recommendations</li>
        </ul>
        All analyses are for research and pattern exploration purposes only.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Analysis:",
        [
            "📊 Data Overview",
            "🔍 Resistance Heatmap",
            "🌳 Cluster Analysis",
            "🎯 Pattern Differentiation",
            "🗺️ Regional & Environmental Analysis",
            "📈 PCA Visualization"
        ]
    )
    
    # Load data
    feature_matrix, metadata = load_default_data()
    
    if feature_matrix is None or metadata is None:
        st.error("Failed to load data. Please check CSV files.")
        return
    
    # Page routing
    if page == "📊 Data Overview":
        show_data_overview(feature_matrix, metadata)
    elif page == "🔍 Resistance Heatmap":
        show_resistance_heatmap(feature_matrix, metadata)
    elif page == "🌳 Cluster Analysis":
        show_cluster_analysis(feature_matrix, metadata)
    elif page == "🎯 Pattern Differentiation":
        show_pattern_differentiation(feature_matrix, metadata)
    elif page == "🗺️ Regional & Environmental Analysis":
        show_regional_analysis(feature_matrix, metadata)
    elif page == "📈 PCA Visualization":
        show_pca_visualization(feature_matrix, metadata)


def show_data_overview(feature_matrix, metadata):
    """Display data overview"""
    st.markdown('<p class="sub-header">📊 Data Overview</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Clean data
    valid_rows = feature_matrix.notna().any(axis=1)
    clean_feature = feature_matrix[valid_rows]
    clean_metadata = metadata[valid_rows]
    
    with col1:
        st.metric("Total Isolates", len(clean_feature))
    
    with col2:
        st.metric("Total Antibiotics", len(feature_matrix.columns))
    
    with col3:
        if 'MDR' in clean_metadata.columns:
            mdr_count = clean_metadata['MDR'].sum()
            st.metric("MDR Isolates", mdr_count)
        else:
            st.metric("MDR Isolates", "N/A")
    
    with col4:
        if 'Region' in clean_metadata.columns:
            n_regions = clean_metadata['Region'].nunique()
            st.metric("Regions", n_regions)
        else:
            st.metric("Regions", "N/A")
    
    # Display data summary
    st.subheader("Resistance Data Summary")
    
    # Calculate resistance statistics
    resistance_counts = (clean_feature == 2).sum()
    susceptible_counts = (clean_feature == 0).sum()
    intermediate_counts = (clean_feature == 1).sum()
    
    stats_df = pd.DataFrame({
        'Antibiotic': clean_feature.columns,
        'Resistant': resistance_counts.values,
        'Intermediate': intermediate_counts.values,
        'Susceptible': susceptible_counts.values,
        'Total Tested': (clean_feature.notna().sum()).values
    })
    stats_df['Resistance %'] = (stats_df['Resistant'] / stats_df['Total Tested'] * 100).round(2)
    stats_df = stats_df.sort_values('Resistance %', ascending=False)
    
    st.dataframe(stats_df, use_container_width=True)
    
    # MDR distribution
    if 'MDR' in clean_metadata.columns:
        st.subheader("MDR Distribution")
        mdr_dist = clean_metadata['MDR'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(mdr_dist)
        with col2:
            st.write("MDR Status Counts:")
            st.write(f"- Non-MDR (0): {mdr_dist.get(0, 0)}")
            st.write(f"- MDR (1): {mdr_dist.get(1, 0)}")


def show_resistance_heatmap(feature_matrix, metadata):
    """Display resistance heatmap"""
    st.markdown('<p class="sub-header">🔍 Resistance Heatmap</p>', unsafe_allow_html=True)
    
    st.write("Visualize resistance patterns across isolates and antibiotics")
    
    # Options
    max_isolates = st.slider("Maximum isolates to display", 50, 500, 200, 50)
    
    # Create visualizer
    visualizer = ResistanceVisualizer(feature_matrix, metadata)
    
    # Generate heatmap
    with st.spinner("Generating heatmap..."):
        fig = visualizer.create_heatmap(max_isolates=max_isolates)
        st.pyplot(fig)
        plt.close()


def show_cluster_analysis(feature_matrix, metadata):
    """Display cluster analysis"""
    st.markdown('<p class="sub-header">🌳 Cluster Analysis</p>', unsafe_allow_html=True)
    
    st.write("Hierarchical clustering reveals recurring AMR patterns")
    
    # Clustering parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        method = st.selectbox("Linkage Method", ['ward', 'average', 'complete', 'single'])
    
    with col2:
        metric = st.selectbox("Distance Metric", ['euclidean', 'manhattan'])
        if method == 'ward':
            metric = 'euclidean'  # Ward requires euclidean
            st.info("Ward linkage requires Euclidean distance")
    
    with col3:
        n_clusters = st.slider("Number of Clusters", 2, 10, 5)
    
    # Perform clustering
    if st.button("Perform Clustering"):
        with st.spinner("Performing clustering..."):
            clusterer = ResistancePatternClusterer(feature_matrix, metadata)
            clusters = clusterer.perform_clustering(
                method=method, 
                metric=metric, 
                n_clusters=n_clusters
            )
            
            # Get summaries
            summaries = clusterer.get_cluster_summaries()
            
            # Display summaries
            st.subheader("Cluster Summaries")
            st.dataframe(summaries, use_container_width=True)
            
            # Visualizations
            st.subheader("Dendrogram")
            visualizer = ResistanceVisualizer(feature_matrix, metadata)
            fig_dend = visualizer.create_dendrogram(clusterer.linkage_matrix)
            st.pyplot(fig_dend)
            plt.close()
            
            st.subheader("Cluster Distribution")
            fig_dist = visualizer.create_cluster_distribution_plot(summaries)
            st.pyplot(fig_dist)
            plt.close()


def show_pattern_differentiation(feature_matrix, metadata):
    """Display supervised pattern recognition"""
    st.markdown('<p class="sub-header">🎯 Pattern Differentiation Analysis</p>', unsafe_allow_html=True)
    
    st.write("""
    Assess how well resistance fingerprints differentiate known groups.
    **Not for prediction** - for understanding pattern separability.
    """)
    
    # Select analysis type
    analysis_type = st.radio(
        "Select Analysis Type:",
        ["Species-Informed Pattern Recognition", "MDR-Informed Pattern Recognition"]
    )
    
    test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
    
    if st.button("Run Pattern Differentiation Analysis"):
        with st.spinner("Analyzing patterns..."):
            recognizer = SupervisedPatternRecognizer(feature_matrix, metadata)
            
            if analysis_type == "Species-Informed Pattern Recognition":
                results = recognizer.species_pattern_recognition(test_size=test_size)
                label_type = "Species"
            else:
                results = recognizer.mdr_pattern_recognition(test_size=test_size)
                label_type = "MDR"
            
            if results is None:
                st.warning("Insufficient data for this analysis")
                return
            
            # Display metrics
            st.subheader("Pattern Differentiation Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{results['accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{results['precision']:.3f}")
            with col3:
                st.metric("Recall", f"{results['recall']:.3f}")
            with col4:
                st.metric("F1-Score", f"{results['f1_score']:.3f}")
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            fig_cm = recognizer.plot_confusion_matrix(results, label_type)
            st.pyplot(fig_cm)
            plt.close()
            
            # Feature importance
            st.subheader("Feature Importance")
            fig_fi = recognizer.plot_feature_importance(results, label_type, top_n=15)
            st.pyplot(fig_fi)
            plt.close()
            
            # Top features table
            st.subheader("Top Contributing Antibiotics")
            st.dataframe(results['feature_importance'].head(15), use_container_width=True)


def show_regional_analysis(feature_matrix, metadata):
    """Display regional and environmental analysis"""
    st.markdown('<p class="sub-header">🗺️ Regional & Environmental Analysis</p>', unsafe_allow_html=True)
    
    st.write("Analyze resistance pattern distribution across regions and environments")
    
    # First perform clustering
    if st.button("Analyze Regional Patterns"):
        with st.spinner("Performing analysis..."):
            # Cluster first
            clusterer = ResistancePatternClusterer(feature_matrix, metadata)
            clusters = clusterer.perform_clustering(n_clusters=5)
            
            # Update metadata
            updated_metadata = clusterer.metadata
            
            # Regional analysis
            analyzer = RegionalPatternAnalyzer(feature_matrix, updated_metadata)
            regional_dist, env_dist = analyzer.analyze_cluster_distribution()
            
            # Display distributions
            if regional_dist is not None:
                st.subheader("Cluster Distribution by Region")
                st.dataframe(regional_dist, use_container_width=True)
            
            if env_dist is not None:
                st.subheader("Cluster Distribution by Environment")
                st.dataframe(env_dist, use_container_width=True)


def show_pca_visualization(feature_matrix, metadata):
    """Display PCA visualizations"""
    st.markdown('<p class="sub-header">📈 PCA Visualization</p>', unsafe_allow_html=True)
    
    st.write("Principal Component Analysis reveals pattern structure in resistance data")
    
    n_components = st.slider("Number of Components", 2, 5, 3)
    
    if st.button("Perform PCA"):
        with st.spinner("Performing PCA..."):
            analyzer = RegionalPatternAnalyzer(feature_matrix, metadata)
            pca_components = analyzer.perform_pca(n_components=n_components)
            
            # Display explained variance
            st.subheader("Explained Variance")
            variance_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(n_components)],
                'Variance Ratio': analyzer.pca.explained_variance_ratio_,
                'Variance %': analyzer.pca.explained_variance_ratio_ * 100
            })
            st.dataframe(variance_df, use_container_width=True)
            
            # PCA plots
            st.subheader("PCA Visualizations")
            
            tab1, tab2, tab3, tab4 = st.tabs(["By Region", "By Environment", "By Species", "By MDR"])
            
            with tab1:
                fig = analyzer.plot_pca_by_region()
                if fig:
                    st.pyplot(fig)
                    plt.close()
            
            with tab2:
                fig = analyzer.plot_pca_by_environment()
                if fig:
                    st.pyplot(fig)
                    plt.close()
            
            with tab3:
                fig = analyzer.plot_pca_by_species()
                if fig:
                    st.pyplot(fig)
                    plt.close()
            
            with tab4:
                fig = analyzer.plot_pca_by_mdr()
                if fig:
                    st.pyplot(fig)
                    plt.close()


if __name__ == "__main__":
    main()
