# Quick Start Guide - AMR Resistance Fingerprint Analysis System

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Running the Analysis

### Option 1: Automated Pipeline (Recommended)

Run the complete analysis pipeline:

```bash
python3 main.py
```

This will:
- Load and process all CSV files
- Perform hierarchical clustering
- Run pattern differentiation analysis
- Generate all visualizations
- Save results to `outputs/` directory

### Option 2: Interactive Dashboard

Launch the Streamlit app:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Option 3: Custom Analysis

Use individual modules in your own scripts:

```python
from data_ingestion import AMRDataLoader, AMRDataHarmonizer
from resistance_encoding import ResistanceFingerprintEncoder
from clustering import ResistancePatternClusterer, ResistanceVisualizer

# Load data
loader = AMRDataLoader(".")
data = loader.load_all_csv_files()

# Harmonize
harmonizer = AMRDataHarmonizer(data)
harmonized = harmonizer.harmonize()

# Encode
encoder = ResistanceFingerprintEncoder(harmonized)
feature_matrix, metadata = encoder.create_resistance_fingerprints()

# Cluster
clusterer = ResistancePatternClusterer(feature_matrix, metadata)
clusters = clusterer.perform_clustering(n_clusters=5)
```

## Understanding the Outputs

### Data Files (CSV)
- `feature_matrix.csv` - Encoded resistance fingerprints (0=S, 1=I, 2=R)
- `metadata.csv` - Isolate metadata with MDR status
- `cluster_summaries.csv` - Cluster characteristics
- `feature_importance_*.csv` - Key antibiotics driving patterns
- `cluster_distribution_*.csv` - Regional/environmental distributions

### Visualizations (PNG)
- `heatmap.png` - Resistance pattern heatmap
- `dendrogram.png` - Hierarchical clustering tree
- `cluster_distribution.png` - Cluster characteristics plots
- `confusion_matrix_*.png` - Pattern overlap analysis
- `feature_importance_*.png` - Key antibiotics plots
- `pca_*.png` - Principal component analysis plots

## Key Concepts

### Resistance Encoding
- **Susceptible (S) = 0**: No resistance
- **Intermediate (I) = 1**: Partial resistance
- **Resistant (R) = 2**: Full resistance
- **Not Tested = NaN**: Missing data

### MDR Status
- **MDR = 1**: Resistant to 3 or more antibiotics
- **Non-MDR = 0**: Resistant to fewer than 3 antibiotics

### Pattern Differentiation Metrics
- **Accuracy**: Overall pattern separability
- **Precision**: Specificity of pattern grouping
- **Recall**: Sensitivity to group patterns
- **F1-score**: Balance of pattern consistency

### Clustering
- **Ward Linkage**: Minimizes within-cluster variance
- **Euclidean Distance**: Standard distance metric
- **Dendrogram**: Shows hierarchical relationships

### PCA
- **PC1, PC2**: Principal components capturing pattern variance
- **Explained Variance**: How much pattern variation is captured

## Tips

1. **Large Datasets**: The heatmap automatically samples 200 isolates for visualization
2. **Missing Data**: Empty/invalid rows are automatically filtered
3. **Custom Clusters**: Adjust `n_clusters` parameter in clustering functions
4. **Output Location**: All results saved to `outputs/` directory

## Troubleshooting

### "No CSV files found"
- Ensure CSV files are in the current directory
- Files should match pattern `1NET_P2-AMR*.csv`

### "Insufficient data"
- Check that CSV files have proper headers and data rows
- Verify at least some isolates have resistance data

### "No species information"
- Species data may not be present in some CSV files
- System will skip species-based analyses

### Memory Issues
- Reduce `max_isolates` parameter in heatmap generation
- Process fewer CSV files at once

## Next Steps

1. Review outputs in `outputs/` directory
2. Explore interactive dashboard: `streamlit run app.py`
3. Customize analysis parameters in `main.py`
4. Generate additional visualizations using individual modules

## Important Reminder

This system is for **exploratory research and pattern analysis only**. It is not designed for:
- Clinical decision-making
- Treatment recommendations
- Predictive forecasting

Always consult with healthcare professionals for clinical decisions.
