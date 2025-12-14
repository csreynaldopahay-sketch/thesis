"""
Resistance Fingerprint Encoding Module
Encodes antibiotic susceptibility data as numerical vectors
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List


class ResistanceFingerprintEncoder:
    """
    Encode antibiotic susceptibility results as resistance fingerprints
    
    Encoding scheme:
    - Susceptible (S) -> 0
    - Intermediate (I) -> 1
    - Resistant (R) -> 2
    """
    
    # Resistance encoding map
    RESISTANCE_ENCODING = {
        'S': 0,
        'I': 1,
        'R': 2
    }
    
    def __init__(self, harmonized_data: pd.DataFrame):
        """
        Initialize encoder with harmonized data
        
        Args:
            harmonized_data: DataFrame from AMRDataHarmonizer
        """
        self.data = harmonized_data.copy()
        self.feature_matrix = None
        self.metadata = None
        self.antibiotic_columns = []
        
    def create_resistance_fingerprints(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create resistance fingerprint vectors for all isolates
        
        Returns:
            Tuple of (feature_matrix, metadata)
            - feature_matrix: DataFrame with isolates × antibiotics
            - metadata: DataFrame with isolate metadata
        """
        print("\nCreating resistance fingerprints...")
        
        # Identify antibiotic interpretation columns
        self._identify_antibiotic_columns()
        
        # Extract resistance patterns
        self._extract_resistance_patterns()
        
        # Separate metadata
        self._separate_metadata()
        
        # Calculate MDR status
        self._calculate_mdr_status()
        
        # Calculate MAR index if not present
        self._calculate_mar_index()
        
        print(f"Feature matrix shape: {self.feature_matrix.shape}")
        print(f"Number of antibiotics: {len(self.antibiotic_columns)}")
        print(f"Number of isolates: {len(self.feature_matrix)}")
        
        return self.feature_matrix, self.metadata
    
    def _identify_antibiotic_columns(self):
        """Identify columns containing antibiotic interpretations"""
        # Look for columns ending with '_INT' (interpretation)
        int_columns = [col for col in self.data.columns if col.endswith('_INT')]
        
        self.antibiotic_columns = int_columns
        
        if not self.antibiotic_columns:
            print("Warning: No interpretation columns found")
        else:
            print(f"Found {len(self.antibiotic_columns)} antibiotic interpretation columns")
    
    def _extract_resistance_patterns(self):
        """Extract resistance patterns and encode as fingerprints"""
        if not self.antibiotic_columns:
            print("Error: No antibiotic columns identified")
            self.feature_matrix = pd.DataFrame()
            return
        
        # Initialize feature matrix dictionary
        resistance_data = {}
        
        # Process each antibiotic interpretation column
        for col in self.antibiotic_columns:
            # Extract antibiotic name (remove _INT suffix)
            ab_name = col.replace('_INT', '')
            
            # Encode values
            encoded_values = [self._encode_value(val) for val in self.data[col]]
            resistance_data[ab_name] = encoded_values
        
        # Create DataFrame
        self.feature_matrix = pd.DataFrame(resistance_data)
        
        # Reset index to avoid duplicate issues
        self.feature_matrix.reset_index(drop=True, inplace=True)
    
    def _encode_value(self, value) -> int:
        """
        Encode a single susceptibility value
        
        Args:
            value: Susceptibility value (S/I/R or variant)
            
        Returns:
            Encoded value (0/1/2) or NaN
        """
        if pd.isna(value):
            return np.nan
        
        val_str = str(value).strip().upper()
        
        # Handle variations like *R, *S
        if 'R' in val_str:
            return 2
        elif 'I' in val_str:
            return 1
        elif 'S' in val_str:
            return 0
        else:
            return np.nan
    
    def _separate_metadata(self):
        """Separate and preserve metadata"""
        metadata_columns = ['Isolate_ID', 'Code', 'Species', 'ESBL', 
                           'Region', 'Site', 'Environment', 
                           'Scored_Resistance', 'Num_Antibiotics_Tested', 'MAR_Index']
        
        # Select existing metadata columns
        available_metadata = [col for col in metadata_columns if col in self.data.columns]
        
        if available_metadata:
            self.metadata = self.data[available_metadata].copy()
            self.metadata.reset_index(drop=True, inplace=True)
        else:
            # Create basic metadata
            self.metadata = pd.DataFrame({
                'Isolate_ID': range(len(self.data))
            })
        
        # Add species standardization
        if 'Species' in self.data.columns:
            self.metadata['Species_Standardized'] = self.data['Species'].apply(
                lambda x: self._standardize_species_name(x)
            )
    
    def _standardize_species_name(self, species_name):
        """Standardize species name"""
        if pd.isna(species_name):
            return 'Unknown'
        
        species_str = str(species_name).strip()
        
        # Apply mapping
        species_mapping = {
            'Escherichia coli': 'E. coli',
            'Klebsiella pneumoniae ssp pneumoniae': 'K. pneumoniae',
            'Klebsiella pneumoniae': 'K. pneumoniae',
            'Salmonella enterica': 'Salmonella spp.',
            'Enterobacter cloacae': 'E. cloacae',
            'Citrobacter freundii': 'C. freundii'
        }
        
        return species_mapping.get(species_str, species_str)
    
    def _calculate_mdr_status(self):
        """
        Calculate MDR (Multi-Drug Resistant) status
        MDR is typically defined as resistance to 3 or more antibiotic classes
        """
        if self.feature_matrix is not None and not self.feature_matrix.empty:
            # Count resistances (value = 2) per isolate
            resistance_count = (self.feature_matrix == 2).sum(axis=1)
            resistance_count.reset_index(drop=True, inplace=True)
            
            # MDR = resistant to 3 or more antibiotics
            # This is a simplified definition
            self.metadata['MDR'] = (resistance_count >= 3).astype(int)
            self.metadata['Resistance_Count'] = resistance_count
    
    def _calculate_mar_index(self):
        """
        Calculate MAR (Multiple Antibiotic Resistance) Index
        MAR Index = Number of resistances / Number of antibiotics tested
        """
        if self.feature_matrix is not None and not self.feature_matrix.empty:
            # Count resistances (value = 2)
            resistances = (self.feature_matrix == 2).sum(axis=1)
            resistances.reset_index(drop=True, inplace=True)
            
            # Count tested antibiotics (non-NaN values)
            tested = self.feature_matrix.notna().sum(axis=1)
            tested.reset_index(drop=True, inplace=True)
            
            # Calculate MAR index
            mar_calculated = (resistances / tested).fillna(0)
            
            # Store both calculated and original (if exists)
            self.metadata['MAR_Index_Calculated'] = mar_calculated
            
            # Keep original MAR_Index if it exists
            if 'MAR_Index' not in self.metadata.columns and 'MAR_Index' in self.data.columns:
                self.metadata['MAR_Index_Original'] = self.data['MAR_Index'].reset_index(drop=True)


if __name__ == "__main__":
    # Test the encoder
    from data_ingestion import AMRDataLoader, AMRDataHarmonizer
    
    loader = AMRDataLoader(".")
    data = loader.load_all_csv_files()
    
    harmonizer = AMRDataHarmonizer(data)
    harmonized = harmonizer.harmonize()
    
    encoder = ResistanceFingerprintEncoder(harmonized)
    feature_matrix, metadata = encoder.create_resistance_fingerprints()
    
    print("\nFeature Matrix:")
    print(feature_matrix.head())
    print("\nMetadata:")
    print(metadata.head())
    print("\nMDR Distribution:")
    print(metadata['MDR'].value_counts())
