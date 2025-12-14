# AMR Resistance Fingerprint Analysis System

An interactive system for exploring antimicrobial resistance (AMR) patterns using unsupervised and supervised machine learning techniques.

## 🎯 Purpose

This system supports **exploratory pattern recognition** and **AMR surveillance research**. It is designed to:

- ✅ Identify recurring AMR patterns through clustering
- ✅ Assess how resistance fingerprints differentiate known groups
- ✅ Visualize resistance patterns across regions and environments
- ✅ Support One Health AMR research

**This system is NOT designed for:**
- ❌ Clinical decision-making
- ❌ Predictive forecasting
- ❌ Treatment recommendations

## 📋 Features

### Phase 2: Data Preparation & Resistance Fingerprint Construction
- **Data Ingestion**: Loads all Project 2 CSV files while preserving original data
- **Data Harmonization**: Standardizes antibiotic names, species labels, regions, and environments
- **Resistance Encoding**: Encodes susceptibility results as numerical fingerprints
  - Susceptible (S) → 0
  - Intermediate (I) → 1
  - Resistant (R) → 2
- **Feature Matrix**: Constructs isolate × antibiotic matrices with metadata preservation

### Phase 3: Unsupervised Pattern Recognition
- **Hierarchical Clustering**: Ward linkage with Euclidean/Manhattan distance
- **Visualizations**:
  - Heatmaps (Isolates × Antibiotics)
  - Dendrograms (cluster similarity)
  - Cluster distribution plots
- **Cluster Annotation**: Summarizes species composition, MDR proportion, regional distribution

### Phase 4: Supervised Learning for Pattern Differentiation
- **Species-Informed Pattern Recognition**: Assesses how resistance patterns separate species
- **MDR-Informed Pattern Recognition**: Evaluates MDR pattern sensitivity
- **Evaluation Metrics**:
  - Confusion matrices
  - Accuracy, precision, recall, F1-score
  - Feature importance (key antibiotics)
- **80-20 Train-Test Split**: Ensures pattern stability assessment

### Phase 5: Regional & Environmental Pattern Analysis
- **Cluster Distribution Analysis**: Compares patterns across regions and environments
- **PCA Visualization**: Reveals pattern structure and separability

### Phase 6: Interactive Streamlit Dashboard
- File upload and processing
- Interactive resistance heatmaps
- Cluster exploration
- Pattern differentiation analysis
- Regional/environmental comparisons
- PCA visualizations

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/csreynaldopahay-sketch/thesis.git
cd thesis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Usage

### Command Line Analysis

#### 1. Data Ingestion and Encoding
```python
from data_ingestion import AMRDataLoader, AMRDataHarmonizer
from resistance_encoding import ResistanceFingerprintEncoder

# Load data
loader = AMRDataLoader(".")
data = loader.load_all_csv_files()

# Harmonize
harmonizer = AMRDataHarmonizer(data)
harmonized = harmonizer.harmonize()

# Encode resistance fingerprints
encoder = ResistanceFingerprintEncoder(harmonized)
feature_matrix, metadata = encoder.create_resistance_fingerprints()
```

#### 2. Clustering Analysis
```python
from clustering import ResistancePatternClusterer, ResistanceVisualizer

# Perform clustering
clusterer = ResistancePatternClusterer(feature_matrix, metadata)
clusters = clusterer.perform_clustering(method='ward', n_clusters=5)

# Get cluster summaries
summaries = clusterer.get_cluster_summaries()

# Create visualizations
visualizer = ResistanceVisualizer(feature_matrix, metadata)
fig_heatmap = visualizer.create_heatmap(linkage_matrix=clusterer.linkage_matrix)
fig_dendrogram = visualizer.create_dendrogram(clusterer.linkage_matrix)
```

#### 3. Pattern Differentiation
```python
from supervised_learning import SupervisedPatternRecognizer

# Initialize recognizer
recognizer = SupervisedPatternRecognizer(feature_matrix, metadata)

# Species pattern recognition
species_results = recognizer.species_pattern_recognition()

# MDR pattern recognition
mdr_results = recognizer.mdr_pattern_recognition()

# Visualize results
recognizer.plot_confusion_matrix(species_results, 'Species')
recognizer.plot_feature_importance(species_results, 'Species')
```

#### 4. Regional Analysis
```python
from regional_analysis import RegionalPatternAnalyzer

# Initialize analyzer
analyzer = RegionalPatternAnalyzer(feature_matrix, metadata)

# Perform PCA
pca_components = analyzer.perform_pca(n_components=3)

# Analyze cluster distribution
regional_dist, env_dist = analyzer.analyze_cluster_distribution()

# Create PCA visualizations
analyzer.plot_pca_by_region()
analyzer.plot_pca_by_environment()
analyzer.plot_pca_by_mdr()
```

### Interactive Dashboard

Launch the Streamlit app:
```bash
streamlit run app.py
```

Then navigate to `http://localhost:8501` in your web browser.

The dashboard provides:
- **Data Overview**: Summary statistics and resistance distribution
- **Resistance Heatmap**: Interactive visualization of resistance patterns
- **Cluster Analysis**: Hierarchical clustering with dendrograms
- **Pattern Differentiation**: Species and MDR pattern recognition
- **Regional Analysis**: Distribution across regions and environments
- **PCA Visualization**: Principal component analysis plots

## 📁 Project Structure

```
thesis/
├── app.py                      # Streamlit dashboard
├── data_ingestion.py          # Data loading and harmonization
├── resistance_encoding.py     # Resistance fingerprint encoding
├── clustering.py              # Hierarchical clustering and visualization
├── supervised_learning.py     # Pattern differentiation analysis
├── regional_analysis.py       # Regional/environmental analysis and PCA
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── *.csv                      # AMR data files
```

## 📈 Outputs

The system generates the following visualizations and reports:

1. **Heatmaps**: Resistance patterns (isolates × antibiotics)
2. **Dendrograms**: Hierarchical clustering trees
3. **Cluster Summaries**: Species composition, MDR proportion, regional distribution
4. **Confusion Matrices**: Pattern overlap analysis
5. **Feature Importance Plots**: Key antibiotics driving pattern separation
6. **PCA Projections**: Pattern structure visualization

All outputs are suitable for inclusion in research papers and thesis documents.

## 🔬 Methodology

### Resistance Encoding
- Each isolate is represented as a numerical vector
- Encoding enables distance-based pattern recognition
- Supports MDR and MAR index analysis
- Matches biological interpretation of resistance

### Unsupervised Learning
- **Purpose**: Identify natural groupings in resistance patterns
- **Method**: Hierarchical clustering with Ward linkage
- **Interpretation**: Reveals recurring AMR patterns without labels

### Supervised Learning (Pattern Differentiation)
- **Purpose**: Assess how well patterns differentiate known groups
- **Not Used For**: Prediction or forecasting
- **Interpretation**: 
  - High accuracy → distinct resistance patterns between groups
  - Low accuracy → overlapping resistance patterns
  - Feature importance → antibiotics driving separation

### Evaluation Metrics Interpretation

| Metric | Pattern-Focused Interpretation |
|--------|-------------------------------|
| Accuracy | Overall separability of patterns |
| Precision | Specificity of resistance-pattern grouping |
| Recall | Sensitivity to group-specific patterns |
| F1-score | Balance of pattern consistency |
| Confusion Matrix | Overlap between resistance profiles |

## 🔒 Data Privacy

- All data processing occurs locally
- No data is transmitted to external servers
- Original CSV files are preserved (read-only)

## 📚 Citation

If you use this system in your research, please cite:

```
[Your thesis citation here]
```

## 👥 Contributors

- [Your name]
- Supervised by: [Supervisor name]

## 📄 License

[Specify license]

## 🙏 Acknowledgments

This work is part of the One Health Antimicrobial Resistance (AMR) surveillance research initiative.

## 📧 Contact

For questions or support, please contact: [your email]

---

**Reminder**: This system is for research and pattern exploration only. It is not designed for clinical decision-making, predictive forecasting, or treatment recommendations.
