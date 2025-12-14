"""
Supervised Learning Module for Pattern Differentiation
Assesses how well resistance fingerprints differentiate known groups
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, 
                            accuracy_score, precision_score, recall_score, f1_score)
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class SupervisedPatternRecognizer:
    """
    Use supervised learning to assess pattern differentiation
    NOT for prediction, but for understanding pattern separability
    """
    
    def __init__(self, feature_matrix: pd.DataFrame, metadata: pd.DataFrame):
        """
        Initialize pattern recognizer
        
        Args:
            feature_matrix: Resistance fingerprints
            metadata: Isolate metadata with labels
        """
        self.feature_matrix = feature_matrix.copy()
        self.metadata = metadata.copy()
        
        # Remove rows with all NaN
        self._clean_data()
        
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def _clean_data(self):
        """Remove rows with no valid data"""
        valid_rows = self.feature_matrix.notna().any(axis=1)
        self.feature_matrix = self.feature_matrix[valid_rows]
        self.metadata = self.metadata[valid_rows]
        
        self.feature_matrix.reset_index(drop=True, inplace=True)
        self.metadata.reset_index(drop=True, inplace=True)
    
    def species_pattern_recognition(self, test_size=0.2, random_state=42):
        """
        Assess how well resistance patterns differentiate bacterial species
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with results
        """
        print("\n" + "="*60)
        print("SPECIES-INFORMED PATTERN RECOGNITION")
        print("="*60)
        print("\nPurpose: Assess how distinctly resistance patterns separate species")
        print("Not for prediction - for pattern analysis only")
        
        # Check if species information is available
        if 'Species_Standardized' not in self.metadata.columns:
            print("Warning: No species information available")
            return None
        
        # Filter out rows with unknown/missing species
        valid_species = self.metadata['Species_Standardized'].notna()
        valid_species &= (self.metadata['Species_Standardized'] != 'Unknown')
        
        if valid_species.sum() < 10:
            print("Insufficient data with species labels")
            return None
        
        X = self.feature_matrix[valid_species].fillna(-1)  # -1 for not tested
        y = self.metadata.loc[valid_species, 'Species_Standardized']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) > 1 else None
        )
        
        print(f"\nDataset partitioned to evaluate stability of patterns:")
        print(f"Training set: {len(self.X_train)} isolates")
        print(f"Test set: {len(self.X_test)} isolates")
        print(f"\nSpecies in dataset: {sorted(y.unique())}")
        
        # Train Random Forest
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state, 
                                           max_depth=10, min_samples_split=5)
        self.model.fit(self.X_train, self.y_train)
        
        # Predict
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        results = self._calculate_metrics(y_pred, 'Species')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Antibiotic': X.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        results['feature_importance'] = feature_importance
        
        print(f"\nTop 10 antibiotics driving species pattern separation:")
        print(feature_importance.head(10).to_string(index=False))
        
        return results
    
    def mdr_pattern_recognition(self, test_size=0.2, random_state=42):
        """
        Assess how well resistance patterns differentiate MDR status
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with results
        """
        print("\n" + "="*60)
        print("MDR-INFORMED PATTERN RECOGNITION")
        print("="*60)
        print("\nPurpose: Evaluate sensitivity to MDR-associated patterns")
        print("Not for prediction - for pattern analysis only")
        
        # Check if MDR information is available
        if 'MDR' not in self.metadata.columns:
            print("Warning: No MDR information available")
            return None
        
        # Filter valid data
        valid_mdr = self.metadata['MDR'].notna()
        
        if valid_mdr.sum() < 10:
            print("Insufficient data with MDR labels")
            return None
        
        X = self.feature_matrix[valid_mdr].fillna(-1)
        y = self.metadata.loc[valid_mdr, 'MDR']
        
        # Check class balance
        mdr_counts = y.value_counts()
        print(f"\nMDR distribution:")
        print(f"Non-MDR (0): {mdr_counts.get(0, 0)}")
        print(f"MDR (1): {mdr_counts.get(1, 0)}")
        
        if len(mdr_counts) < 2:
            print("Only one class present - cannot perform pattern recognition")
            return None
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nDataset partitioned to evaluate stability of MDR patterns:")
        print(f"Training set: {len(self.X_train)} isolates")
        print(f"Test set: {len(self.X_test)} isolates")
        
        # Train Random Forest
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state,
                                           max_depth=10, min_samples_split=5,
                                           class_weight='balanced')
        self.model.fit(self.X_train, self.y_train)
        
        # Predict
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        results = self._calculate_metrics(y_pred, 'MDR')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Antibiotic': X.columns,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        results['feature_importance'] = feature_importance
        
        print(f"\nTop 10 antibiotics contributing to MDR pattern differentiation:")
        print(feature_importance.head(10).to_string(index=False))
        
        return results
    
    def _calculate_metrics(self, y_pred, label_type):
        """
        Calculate pattern differentiation metrics
        
        Args:
            y_pred: Predictions
            label_type: Type of label ('Species' or 'MDR')
            
        Returns:
            Dictionary with metrics
        """
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Classification report
        report = classification_report(self.y_test, y_pred, output_dict=True)
        
        # Overall metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Calculate weighted averages for multi-class
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\n{label_type} Pattern Differentiation Metrics:")
        print(f"{'Metric':<25} {'Value':<10} {'Interpretation'}")
        print("-" * 70)
        print(f"{'Accuracy':<25} {accuracy:.3f}      Overall separability of patterns")
        print(f"{'Precision (weighted)':<25} {precision:.3f}      Specificity of pattern grouping")
        print(f"{'Recall (weighted)':<25} {recall:.3f}      Sensitivity to group patterns")
        print(f"{'F1-score (weighted)':<25} {f1:.3f}      Balance of pattern consistency")
        
        print(f"\nDetailed Classification Report:")
        print("-" * 70)
        for label, metrics in report.items():
            if isinstance(metrics, dict) and label not in ['accuracy', 'macro avg', 'weighted avg']:
                print(f"\n{label}:")
                print(f"  Precision: {metrics['precision']:.3f}")
                print(f"  Recall: {metrics['recall']:.3f}")
                print(f"  F1-score: {metrics['f1-score']:.3f}")
                print(f"  Support: {metrics['support']}")
        
        return {
            'confusion_matrix': cm,
            'classification_report': report,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_true': self.y_test,
            'y_pred': y_pred
        }
    
    def plot_confusion_matrix(self, results, label_type, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            results: Results dictionary from pattern recognition
            label_type: Type of label
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        cm = results['confusion_matrix']
        labels = sorted(results['y_true'].unique())
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax)
        
        ax.set_xlabel('Pattern-Based Assignment', fontsize=12)
        ax.set_ylabel('Actual Label', fontsize=12)
        ax.set_title(f'{label_type} Pattern Overlap Analysis\n(Confusion Matrix)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_feature_importance(self, results, label_type, top_n=15, save_path=None):
        """
        Plot feature importance
        
        Args:
            results: Results dictionary
            label_type: Type of label
            top_n: Number of top features to plot
            save_path: Path to save figure
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        feature_imp = results['feature_importance'].head(top_n)
        
        ax.barh(range(len(feature_imp)), feature_imp['Importance'], color='steelblue')
        ax.set_yticks(range(len(feature_imp)))
        ax.set_yticklabels(feature_imp['Antibiotic'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Antibiotic', fontsize=12)
        ax.set_title(f'Key Antibiotics Driving {label_type} Pattern Separation', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        return fig


if __name__ == "__main__":
    # Test the supervised learning
    from data_ingestion import AMRDataLoader, AMRDataHarmonizer
    from resistance_encoding import ResistanceFingerprintEncoder
    
    # Load and process data
    loader = AMRDataLoader(".")
    data = loader.load_all_csv_files()
    
    harmonizer = AMRDataHarmonizer(data)
    harmonized = harmonizer.harmonize()
    
    encoder = ResistanceFingerprintEncoder(harmonized)
    feature_matrix, metadata = encoder.create_resistance_fingerprints()
    
    # Perform pattern recognition
    recognizer = SupervisedPatternRecognizer(feature_matrix, metadata)
    
    # Species-informed pattern recognition
    species_results = recognizer.species_pattern_recognition()
    
    if species_results:
        fig_cm_species = recognizer.plot_confusion_matrix(
            species_results, 'Species', 
            save_path='confusion_matrix_species.png'
        )
        
        fig_fi_species = recognizer.plot_feature_importance(
            species_results, 'Species',
            save_path='feature_importance_species.png'
        )
    
    # MDR-informed pattern recognition
    mdr_results = recognizer.mdr_pattern_recognition()
    
    if mdr_results:
        fig_cm_mdr = recognizer.plot_confusion_matrix(
            mdr_results, 'MDR',
            save_path='confusion_matrix_mdr.png'
        )
        
        fig_fi_mdr = recognizer.plot_feature_importance(
            mdr_results, 'MDR',
            save_path='feature_importance_mdr.png'
        )
    
    plt.show()
    
    print("\nSupervised pattern recognition complete!")
