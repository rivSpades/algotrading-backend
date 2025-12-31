"""
Data Validation Service
Validates OHLCV data for quality issues: missing values, duplicates, outliers,
OHLCV logical constraints, large time gaps, extreme price jumps, and suspicious patterns
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
    4. Complete OHLCV logical constraints (High >= all, Low <= all)
    5. Large time gaps (>90 days)
    6. Extreme single-day price jumps (>50%, or >100% after >1 year gap)
    7. Suspicious patterns (3+ consecutive days with identical OHLC values)
    
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
        
        # Check complete OHLCV logical constraints
        # High should be >= all prices (open, high, low, close)
        # Low should be <= all prices (open, high, low, close)
        logical_violations = []
        
        # Ensure all price columns are numeric
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Check High >= all prices
        if all(col in df.columns for col in ['high', 'open', 'low', 'close']):
            high_vs_open = (df['high'] < df['open']).sum()
            high_vs_close = (df['high'] < df['close']).sum()
            if high_vs_open > 0:
                logical_violations.append(f"high < open: {high_vs_open} violations")
            if high_vs_close > 0:
                logical_violations.append(f"high < close: {high_vs_close} violations")
        
        # Check Low <= all prices
        if all(col in df.columns for col in ['low', 'open', 'high', 'close']):
            low_vs_open = (df['low'] > df['open']).sum()
            low_vs_close = (df['low'] > df['close']).sum()
            if low_vs_open > 0:
                logical_violations.append(f"low > open: {low_vs_open} violations")
            if low_vs_close > 0:
                logical_violations.append(f"low > close: {low_vs_close} violations")
        
        if logical_violations:
            outlier_checks.append(f"OHLCV logical violations - {', '.join(logical_violations)}")
        
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
        
        # 4. Check for large time gaps and suspicious price jumps after gaps
        # Large gaps (>90 days) indicate missing data and should be flagged
        large_gap_threshold_days = 90  # Define threshold for large gaps (>90 days = ~3 months)
        if len(df) > 1 and 'timestamp' in df.columns:
            # Sort by timestamp to ensure chronological order
            df_sorted = df.sort_values('timestamp').reset_index(drop=True)
            df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'], errors='coerce')
            
            # Calculate time gaps between consecutive records
            time_gaps = df_sorted['timestamp'].diff().dt.days
            
            # Check for large time gaps
            large_gaps = time_gaps[time_gaps > large_gap_threshold_days]
            
            if len(large_gaps) > 0:
                gap_details = []
                for idx in large_gaps.index:
                    if idx > 0 and pd.notna(time_gaps.iloc[idx]):
                        gap_days = int(time_gaps.iloc[idx])
                        gap_years = gap_days / 365.25
                        prev_date = df_sorted.iloc[idx - 1]['timestamp']
                        curr_date = df_sorted.iloc[idx]['timestamp']
                        
                        if pd.notna(prev_date) and pd.notna(curr_date):
                            prev_date_str = pd.to_datetime(prev_date).strftime('%Y-%m-%d')
                            curr_date_str = pd.to_datetime(curr_date).strftime('%Y-%m-%d')
                            gap_details.append(
                                f"{prev_date_str}->{curr_date_str} ({gap_days} days, {gap_years:.1f} years)"
                            )
                
                if gap_details:
                    reasons.append(
                        f"Large time gaps detected ({len(large_gaps)}): " + 
                        "; ".join(gap_details[:3])  # Limit to first 3
                    )
        
        # 5. Check for extreme single-day price jumps (including after large gaps)
        # Price jumps >50% are not normal and indicate data quality issues
        if len(df) > 1 and 'timestamp' in df.columns and 'close' in df.columns:
            # Sort by timestamp to ensure chronological order (reuse from previous check if available)
            if 'df_sorted' not in locals():
                df_sorted = df.sort_values('timestamp').reset_index(drop=True)
            df_sorted['close'] = pd.to_numeric(df_sorted['close'], errors='coerce')
            df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'], errors='coerce')
            
            # Calculate time gaps and price changes
            time_gaps = df_sorted['timestamp'].diff().dt.days
            price_changes = df_sorted['close'].pct_change().abs()
            
            # Flag extreme jumps (>50% change)
            extreme_jump_threshold = 0.50  # 50%
            extreme_jumps = price_changes[price_changes > extreme_jump_threshold]
            
            if len(extreme_jumps) > 0:
                jump_details = []
                for idx in extreme_jumps.index:
                    if idx > 0:
                        prev_close = df_sorted.iloc[idx - 1]['close']
                        curr_close = df_sorted.iloc[idx]['close']
                        change_pct = price_changes.iloc[idx] * 100
                        jump_date = df_sorted.iloc[idx]['timestamp']
                        
                        # Check if this jump occurs after a large gap
                        gap_days = time_gaps.iloc[idx] if idx < len(time_gaps) else None
                        is_after_large_gap = gap_days is not None and gap_days > large_gap_threshold_days
                        
                        # Special handling for jumps after large gaps: flag if >100% change after >1 year gap
                        if is_after_large_gap and gap_days > 365 and change_pct > 100:
                            gap_info = f" (after {int(gap_days)} day gap)"
                        else:
                            gap_info = ""
                        
                        if pd.notna(prev_close) and pd.notna(curr_close):
                            date_str = pd.to_datetime(jump_date).strftime('%Y-%m-%d') if pd.notna(jump_date) else 'N/A'
                            jump_details.append(
                                f"{date_str}: {prev_close:.4f}->{curr_close:.4f} ({change_pct:.2f}%){gap_info}"
                            )
                
                if jump_details:
                    reasons.append(
                        f"Extreme price jumps detected ({len(extreme_jumps)}): " + 
                        "; ".join(jump_details[:5])  # Limit to first 5 to avoid very long messages
                    )
        
        # 6. Check for suspicious patterns: consecutive days with identical OHLC values
        # This can indicate data quality issues or stale data
        if len(df) > 2 and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Sort by timestamp if not already sorted
            if 'df_sorted' not in locals():
                df_sorted = df.sort_values('timestamp').reset_index(drop=True)
            
            # Check for identical OHLC values (all prices the same)
            df_sorted['identical_ohlc'] = (
                (df_sorted['open'] == df_sorted['high']) &
                (df_sorted['high'] == df_sorted['low']) &
                (df_sorted['low'] == df_sorted['close'])
            )
            
            # Check for 3 or more consecutive days with identical OHLC
            consecutive_identical = df_sorted['identical_ohlc'].rolling(window=3, min_periods=3).sum()
            suspicious_patterns = consecutive_identical[consecutive_identical >= 3]
            
            if len(suspicious_patterns) > 0:
                pattern_details = []
                pattern_indices = suspicious_patterns.index
                for idx in pattern_indices[:5]:  # Limit to first 5
                    if idx >= 2:  # Need at least 3 consecutive records
                        pattern_date = df_sorted.iloc[idx]['timestamp']
                        if pd.notna(pattern_date):
                            date_str = pd.to_datetime(pattern_date).strftime('%Y-%m-%d')
                            price = df_sorted.iloc[idx]['close']
                            pattern_details.append(f"{date_str} (price: {price:.4f})")
                
                if pattern_details:
                    reasons.append(
                        f"Suspicious patterns detected ({len(suspicious_patterns)} instances of 3+ consecutive identical OHLC): " +
                        "; ".join(pattern_details)
                    )
        
        # Determine if data is valid
        is_valid = len(reasons) == 0
        reason = "; ".join(reasons) if reasons else ""
        
        return is_valid, reason
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"


