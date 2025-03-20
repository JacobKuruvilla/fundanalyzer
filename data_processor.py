import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

class DataProcessor:
    def __init__(self):
        self.df = None
        self.debug = True  # Enable debug logging

    def log(self, message: str) -> None:
        """Debug logging function"""
        if self.debug:
            print(f"DataProcessor: {message}")

    def format_date(self, date_value) -> str:
        """Safely format a date value, handling NaT"""
        if pd.isna(date_value):
            return "N/A"
        try:
            return date_value.strftime('%Y-%m-%d')
        except:
            return "N/A"

    def load_data(self, file_path: str) -> None:
        """Load and preprocess the Excel file."""
        try:
            # Load the Excel file
            self.log("Loading Excel file...")
            self.df = pd.read_excel(file_path)

            # Print column names for debugging
            self.log(f"Found columns: {', '.join(self.df.columns.tolist())}")

            # Map expected column names to actual column names
            column_mapping = {
                'Funding Opportunity Number': 'Opportunity Number',
                'Funding Opportunity Title': 'Opportunity Title',
                'Description': 'Description',
                'Posted Date': 'Posted Date',
                'Closing Date': 'Close Date',  # Changed from 'CLOSE DATE' to match actual column
                'Funding Instrument Type': 'Funding Instrument Type',
                'Agency Name': 'Agency Name',
                'Eligible Applicants': 'Eligibility',
                'Additional Information on Eligibility': 'Additional Eligibility',
                'Assistance Listings': 'Assistance Listings',
                'HHS Divison': 'HHS Division'  # Fixed typo in 'Division'
            }

            # Verify columns exist and create mapping
            actual_mapping = {}
            missing_cols = []
            for actual, expected in column_mapping.items():
                if actual in self.df.columns:
                    actual_mapping[actual] = expected
                    self.log(f"Mapped column {actual} to {expected}")
                else:
                    missing_cols.append(actual)

            if missing_cols:
                self.log(f"Missing optional columns: {', '.join(missing_cols)}")

            # Rename columns
            self.df = self.df.rename(columns=actual_mapping)

            # Clean and preprocess data
            self.df = self.df.dropna(subset=['Description'])

            # Add empty columns if they don't exist
            for col in ['Eligibility', 'Additional Eligibility', 'Assistance Listings', 'HHS Division']:
                if col not in self.df.columns:
                    self.df[col] = ''
                    self.log(f"Added empty {col} column")

            # Convert date columns with better error handling
            for date_col in ['Posted Date', 'Close Date']:
                try:
                    if date_col in self.df.columns:
                        self.log(f"Converting {date_col} to datetime...")
                        self.df[date_col] = pd.to_datetime(self.df[date_col], errors='coerce')
                        self.log(f"Successfully converted {date_col}")
                    else:
                        self.log(f"Warning: {date_col} column not found")
                except Exception as e:
                    self.log(f"Error converting {date_col}: {str(e)}")
                    self.df[date_col] = pd.NaT

            if len(self.df) == 0:
                raise ValueError("No valid data found in the file after preprocessing")

            self.log(f"Successfully loaded {len(self.df)} records")

        except Exception as e:
            self.df = None
            raise Exception(f"Error loading data: {str(e)}")

    def get_hhs_divisions(self) -> List[str]:
        """Get list of all HHS divisions"""
        if self.df is None:
            return []
        return sorted(self.df['HHS Division'].unique().tolist())

    def get_agencies(self) -> List[str]:
        """Get list of all agencies"""
        if self.df is None:
            return []
        return sorted(self.df['Agency Name'].unique().tolist())

    def get_division_opportunities(self, division_name: str) -> pd.DataFrame:
        """Get all opportunities for a specific HHS division"""
        if self.df is None:
            return pd.DataFrame()
        return self.df[self.df['HHS Division'] == division_name]

    def get_basic_stats(self, division_name: str = None) -> dict:
        """Calculate basic statistics about the dataset."""
        if self.df is None:
            return {
                'total_opportunities': 0,
                'funding_types': {},
                'avg_days_open': 0,
                'active_opportunities': 0
            }

        try:
            # Filter by division if specified
            df = self.df if division_name is None else self.df[self.df['HHS Division'] == division_name]

            # Calculate average days open, handling NaT values
            days_open = (df['Close Date'] - df['Posted Date']).dropna()
            avg_days = days_open.mean().days if not days_open.empty else 0

            # Count active opportunities (future close dates)
            now = pd.Timestamp.now()
            active_mask = df['Close Date'].notna() & (df['Close Date'] > now)

            stats = {
                'total_opportunities': len(df),
                'funding_types': df['Funding Instrument Type'].value_counts().to_dict(),
                'avg_days_open': avg_days,
                'active_opportunities': active_mask.sum()
            }
            return stats
        except Exception as e:
            self.log(f"Error calculating stats: {str(e)}")
            return {}

    def get_opportunity_details(self, opp_number: str) -> dict:
        """Get details for a specific opportunity."""
        if self.df is None or opp_number not in self.df['Opportunity Number'].values:
            return {}

        try:
            opp = self.df[self.df['Opportunity Number'] == opp_number].iloc[0]
            return {
                'number': opp['Opportunity Number'],
                'title': opp['Opportunity Title'],
                'description': opp['Description'],
                'eligibility': opp.get('Eligibility', ''),
                'additional_eligibility': opp.get('Additional Eligibility', ''),
                'assistance_listings': opp.get('Assistance Listings', ''),
                'posted_date': self.format_date(opp['Posted Date']),
                'close_date': self.format_date(opp['Close Date']),
                'type': opp['Funding Instrument Type'],
                'agency': opp['Agency Name'],
                'hhs_division': opp.get('HHS Division', '')
            }
        except Exception as e:
            self.log(f"Error getting opportunity details: {str(e)}")
            return {}
