"""
Data Validation Service
Validates OHLCV data for quality issues: missing values, duplicates, outliers
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


def validate_ohlcv_data(ohlcv_data: List[Dict]) -> Tuple[bool, str]:
    """
    Validate OHLCV data for quality issues
    
    Checks:
    1. Missing values (null or NaN)
    2. Duplicate timestamps
    3. Outliers in price data
    
    Args:
        ohlcv_data: List of OHLCV dicts with timestamp, open, high, low, close, volume
    
    Returns:
        Tuple of (is_valid: bool, reason: str)
        - is_valid: True if all checks pass, False otherwise
        - reason: Empty string if valid, otherwise description of failed checks
    """
    if not ohlcv_data or len(ohlcv_data) == 0:
        return False, "No data provided"
    
    try:
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(ohlcv_data)
        
        # Required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        reasons = []
        
        # 1. Check for missing values (null or NaN)
        missing_checks = []
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                missing_count = df[col].isna().sum() + (df[col] == None).sum()
                if missing_count > 0:
                    missing_checks.append(f"{col}: {missing_count} missing")
        
        if missing_checks:
            reasons.append(f"Missing values - {', '.join(missing_checks)}")
        
        # Check timestamp column
        if 'timestamp' in df.columns:
            missing_timestamps = df['timestamp'].isna().sum() + (df['timestamp'] == None).sum()
            if missing_timestamps > 0:
                reasons.append(f"timestamp: {missing_timestamps} missing")
        
        # 2. Check for duplicate timestamps
        if 'timestamp' in df.columns:
            # Convert timestamps to comparable format
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            duplicate_count = df['timestamp'].duplicated().sum()
            if duplicate_count > 0:
                reasons.append(f"Duplicate timestamps: {duplicate_count} duplicates found")
        
        # 3. Check for outliers in price data
        # Outliers: values that are more than 3 standard deviations from the mean
        # or values that are clearly wrong (negative prices, high > low violations, etc.)
        outlier_checks = []
        
        # Check for negative or zero prices (except volume can be zero)
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                negative_count = (df[col] <= 0).sum()
                if negative_count > 0:
                    outlier_checks.append(f"{col}: {negative_count} non-positive values")
        
        # Check for high < low violations
        if 'high' in df.columns and 'low' in df.columns:
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            violation_count = (df['high'] < df['low']).sum()
            if violation_count > 0:
                outlier_checks.append(f"high < low violations: {violation_count}")
        
        # Check for extreme outliers using IQR method (more robust than z-score)
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Remove NaN values for outlier detection
                values = df[col].dropna()
                if len(values) > 0:
                    Q1 = values.quantile(0.25)
                    Q3 = values.quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:  # Only check if there's variation
                        lower_bound = Q1 - 3 * IQR  # 3x IQR for extreme outliers
                        upper_bound = Q3 + 3 * IQR
                        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                        if outliers > 0:
                            # Only flag if outliers are significant (> 1% of data)
                            outlier_percentage = (outliers / len(df)) * 100
                            if outlier_percentage > 1.0:
                                outlier_checks.append(f"{col}: {outliers} extreme outliers ({outlier_percentage:.1f}%)")
        
        if outlier_checks:
            reasons.append(f"Outliers detected - {', '.join(outlier_checks)}")
        
        # Determine if data is valid
        is_valid = len(reasons) == 0
        reason = "; ".join(reasons) if reasons else ""
        
        return is_valid, reason
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


