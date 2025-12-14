"""
Data Ingestion Module for AMR Resistance Fingerprint Analysis
Loads and harmonizes Project 2 CSV files
"""

import pandas as pd
import numpy as np
import glob
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class AMRDataLoader:
    """Load and preprocess AMR CSV files"""
    
    def __init__(self, data_dir: str = "."):
        """
        Initialize the data loader
        
        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = data_dir
        self.raw_data = []
        self.combined_data = None
        
    def load_all_csv_files(self) -> pd.DataFrame:
        """
        Load all Project 2 CSV files while preserving original data
        
        Returns:
            Combined DataFrame with all isolates
        """
        # Find all CSV files matching the pattern
        csv_files = glob.glob(os.path.join(self.data_dir, "1NET_P2-AMR*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        print(f"Found {len(csv_files)} CSV files")
        
        all_data = []
        
        for csv_file in csv_files:
            print(f"Loading {os.path.basename(csv_file)}...")
            
            # Extract metadata from filename
            filename = os.path.basename(csv_file)
            metadata = self._extract_metadata_from_filename(filename)
            
            # Read CSV, skipping header rows
            df = self._read_csv_file(csv_file)
            
            if df is not None and not df.empty:
                # Add metadata columns
                for key, value in metadata.items():
                    df[key] = value
                
                # Store raw data (preserving original)
                self.raw_data.append(df.copy())
                all_data.append(df)
        
        # Combine all data
        if all_data:
            self.combined_data = pd.concat(all_data, ignore_index=True)
            print(f"\nTotal isolates loaded: {len(self.combined_data)}")
            return self.combined_data
        else:
            raise ValueError("No data could be loaded from CSV files")
    
    def _read_csv_file(self, filepath: str) -> pd.DataFrame:
        """
        Read a single CSV file with proper parsing
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Parsed DataFrame
        """
        try:
            # CSV has multi-level headers starting at row 5 (0-indexed)
            # Row 5: Main headers (CODE, ISOLATE ID, etc.)
            # Row 6: Antibiotic codes (AM, AMC, etc.)
            # Row 7: MIC/INT indicators
            
            # First check if file has enough rows
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 8:  # Not enough rows for multi-level header
                print(f"Skipping {filepath} - insufficient rows")
                return None
            
            # Read with multi-level header
            df = pd.read_csv(filepath, skiprows=5, header=[0, 1, 2])
            
            # Flatten the multi-level columns
            df = self._flatten_columns(df)
            
            # Remove empty rows and columns
            df = df.dropna(how='all', axis=0)
            df = df.dropna(how='all', axis=1)
            
            # Remove rows where Isolate_ID is empty
            if 'Isolate_ID' in df.columns:
                df = df[df['Isolate_ID'].notna()]
                df = df[df['Isolate_ID'] != '']
            
            return df
            
        except Exception as e:
            print(f"Error reading {filepath}: {str(e)}")
            return None
    
    def _flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten multi-level column headers
        
        Args:
            df: DataFrame with multi-level columns
            
        Returns:
            DataFrame with flattened columns
        """
        new_columns = []
        column_counts = {}  # Track duplicate column names
        last_antibiotic = None  # Track the last antibiotic code seen
        
        for col in df.columns:
            if isinstance(col, tuple):
                # Extract meaningful parts
                level0, level1, level2 = col
                
                # Handle specific columns
                if 'CODE' in str(level0):
                    col_name = 'Code'
                elif 'ISOLATE ID' in str(level0):
                    col_name = 'Isolate_ID'
                elif 'LEVEL OF RESISTANCE' in str(level0):
                    if 'ESBL' in str(level1):
                        col_name = 'ESBL'
                    else:
                        col_name = 'Species'
                elif 'Scored Resistance' in str(level0):
                    col_name = 'Scored_Resistance'
                elif 'No. of Antibiotic' in str(level0):
                    col_name = 'Num_Antibiotics_Tested'
                elif 'MAR INDEX' in str(level0):
                    col_name = 'MAR_Index'
                elif level2 == 'MIC' and level1 and not level1.startswith('Unnamed'):
                    # MIC column with antibiotic name
                    last_antibiotic = level1
                    col_name = f'{level1}_MIC'
                elif level2 == 'INT.':
                    # INT. column (interpretation) - use last antibiotic code
                    if last_antibiotic:
                        col_name = f'{last_antibiotic}_INT'
                    else:
                        col_name = 'Unknown_INT'
                else:
                    # Default naming
                    col_name = f'Unnamed_{df.columns.get_loc(col)}'
            else:
                col_name = col
            
            # Handle duplicate column names
            if col_name in column_counts:
                column_counts[col_name] += 1
                col_name = f'{col_name}_{column_counts[col_name]}'
            else:
                column_counts[col_name] = 0
            
            new_columns.append(col_name)
        
        df.columns = new_columns
        return df
    
    def _extract_metadata_from_filename(self, filename: str) -> Dict[str, str]:
        """
        Extract region, site information from filename
        
        Args:
            filename: CSV filename
            
        Returns:
            Dictionary with metadata
        """
        metadata = {}
        
        # Extract region
        if "BARMM Region" in filename:
            metadata['Region'] = "BARMM"
        elif "Region III-Central Luzon" in filename:
            metadata['Region'] = "Region III - Central Luzon"
        elif "Region VIII-Eastern Visayas" in filename:
            metadata['Region'] = "Region VIII - Eastern Visayas"
        else:
            metadata['Region'] = "Unknown"
        
        # Extract site name (after "LOR-")
        if "LOR-" in filename:
            site_part = filename.split("LOR-")[1]
            site_name = site_part.replace(".csv", "").strip()
            metadata['Site'] = site_name
        else:
            metadata['Site'] = "Unknown"
        
        return metadata
    
    def get_raw_data(self) -> List[pd.DataFrame]:
        """
        Get the raw, unmodified data
        
        Returns:
            List of original DataFrames
        """
        return self.raw_data


class AMRDataHarmonizer:
    """Harmonize and standardize AMR data"""
    
    # Standard antibiotic name mappings
    ANTIBIOTIC_MAPPING = {
        'AM': 'Ampicillin',
        'AMC': 'Amoxicillin-Clavulanate',
        'CPT': 'Cefpodoxime',
        'CN': 'Gentamicin',
        'CF': 'Cephalothin',
        'CPD': 'Cefpodoxime',
        'CTX': 'Cefotaxime',
        'CFO': 'Cefoxitin',
        'CFT': 'Ceftriaxone',
        'CZA': 'Ceftazidime-Avibactam',
        'IPM': 'Imipenem',
        'AN': 'Amikacin',
        'GM': 'Gentamicin',
        'N': 'Neomycin',
        'NAL': 'Nalidixic Acid',
        'ENR': 'Enrofloxacin',
        'MRB': 'Meropenem',
        'PRA': 'Piperacillin',
        'DO': 'Doxycycline',
        'TE': 'Tetracycline',
        'FT': 'Nitrofurantoin',
        'C': 'Chloramphenicol',
        'SXT': 'Trimethoprim-Sulfamethoxazole'
    }
    
    # Standard species name mappings
    SPECIES_MAPPING = {
        'Escherichia coli': 'E. coli',
        'Klebsiella pneumoniae ssp pneumoniae': 'K. pneumoniae',
        'Klebsiella pneumoniae': 'K. pneumoniae',
        'Salmonella enterica': 'Salmonella spp.',
        'Enterobacter cloacae': 'E. cloacae',
        'Citrobacter freundii': 'C. freundii'
    }
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize harmonizer with data
        
        Args:
            data: Combined DataFrame from loader
        """
        self.data = data.copy()
        self.harmonized_data = None
        
    def harmonize(self) -> pd.DataFrame:
        """
        Perform complete data harmonization
        
        Returns:
            Harmonized DataFrame
        """
        print("\nHarmonizing data...")
        
        # Standardize column names
        self._standardize_columns()
        
        # Standardize species labels
        self._standardize_species()
        
        # Standardize regions
        self._standardize_regions()
        
        # Standardize environment categories (infer from site/context)
        self._infer_environments()
        
        # Extract and standardize susceptibility data
        self._extract_susceptibility_data()
        
        self.harmonized_data = self.data
        print("Harmonization complete!")
        
        return self.harmonized_data
    
    def _standardize_columns(self):
        """Standardize column names - already done in flattening"""
        # Columns are already standardized in the flattening process
        pass
    
    def _standardize_species(self):
        """Standardize species labels"""
        if 'Species' in self.data.columns:
            # Apply species mapping
            self.data['Species_Standardized'] = self.data['Species'].apply(
                lambda x: self.SPECIES_MAPPING.get(x, x) if pd.notna(x) else 'Unknown'
            )
    
    def _standardize_regions(self):
        """Ensure consistent region naming"""
        if 'Region' in self.data.columns:
            region_mapping = {
                'BARMM': 'BARMM Region',
                'Region III - Central Luzon': 'Central Luzon',
                'Region VIII - Eastern Visayas': 'Eastern Visayas'
            }
            self.data['Region'] = self.data['Region'].map(
                lambda x: region_mapping.get(x, x) if pd.notna(x) else 'Unknown'
            )
    
    def _infer_environments(self):
        """Infer environment categories from site names"""
        def categorize_environment(site):
            if pd.isna(site):
                return 'Unknown'
            
            site_lower = str(site).lower()
            
            # Categorize based on keywords
            if 'hospital' in site_lower or 'clinic' in site_lower:
                return 'Clinical'
            elif 'water' in site_lower or 'river' in site_lower or 'lake' in site_lower:
                return 'Water'
            elif 'farm' in site_lower or 'livestock' in site_lower:
                return 'Animal'
            elif 'market' in site_lower or 'food' in site_lower:
                return 'Food'
            else:
                return 'Environmental'
        
        if 'Site' in self.data.columns:
            self.data['Environment'] = self.data['Site'].apply(categorize_environment)
    
    def _extract_susceptibility_data(self):
        """Extract susceptibility interpretations (S/I/R) from columns"""
        # This will be handled in the resistance encoder
        pass


if __name__ == "__main__":
    # Test the data loading
    loader = AMRDataLoader(".")
    data = loader.load_all_csv_files()
    print("\nData shape:", data.shape)
    print("\nColumns:", data.columns.tolist())
    
    # Test harmonization
    harmonizer = AMRDataHarmonizer(data)
    harmonized = harmonizer.harmonize()
    print("\nHarmonized data shape:", harmonized.shape)
    print("\nFirst few rows:")
    print(harmonized.head())
