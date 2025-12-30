"""
Polygon.io Provider
Handles fetching OHLCV data from Polygon.io via S3 flat files
Uses the same structure and interface as YahooFinanceProvider
"""

import boto3
import pandas as pd
import json
from typing import List, Dict, Optional
from datetime import datetime, date
from django.utils import timezone
from io import BytesIO, StringIO
import pytz


class PolygonProvider:
    """Provider for Polygon.io flat files via S3 - same interface as YahooFinanceProvider"""
    
    # Class-level variables to store credentials (set via factory)
    _access_key_id = None
    _secret_access_key = None
    _endpoint_url = None
    _bucket_name = None
    _s3_client = None
    
    @classmethod
    def initialize(cls, access_key_id: str, secret_access_key: str, endpoint_url: str, bucket_name: str):
        """
        Initialize Polygon provider with S3 credentials (called by factory)
        
        Args:
            access_key_id: S3 Access Key ID
            secret_access_key: S3 Secret Access Key
            endpoint_url: S3 endpoint URL (e.g., https://files.massive.com)
            bucket_name: S3 bucket name (e.g., flatfiles)
        """
        cls._access_key_id = access_key_id
        cls._secret_access_key = secret_access_key
        cls._endpoint_url = endpoint_url
        cls._bucket_name = bucket_name
        
        # Initialize S3 client with proper configuration (signature_version='s3v4' is required)
        from botocore.config import Config
        session = boto3.Session(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
        )
        cls._s3_client = session.client(
            's3',
            endpoint_url=endpoint_url,
            config=Config(signature_version='s3v4')
        )
    
    @staticmethod
    def get_historical_data(
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: Optional[str] = None,
        interval: str = '1d'
    ) -> List[Dict]:
        """
        Get historical OHLCV data for a symbol (same interface as YahooFinanceProvider)
        
        Args:
            ticker: Symbol ticker (e.g., 'AAPL', 'MSFT')
            start_date: Start date for data (datetime object)
            end_date: End date for data (datetime object)
            period: Period string (not commonly used with flat files)
            interval: Data interval ('1d' for daily, '1h' for hourly, '1m' for minute)
        
        Returns:
            List of dictionaries with OHLCV data (same format as YahooFinanceProvider):
            [
                {
                    'timestamp': datetime,
                    'open': float,
                    'high': float,
                    'low': float,
                    'close': float,
                    'volume': int
                },
                ...
            ]
        """
        # Use bulk method for single symbol
        bulk_data = PolygonProvider.get_multiple_symbols_data(
            tickers=[ticker],
            start_date=start_date,
            end_date=end_date,
            period=period,
            interval=interval
        )
        return bulk_data.get(ticker, [])
    
    @staticmethod
    def get_multiple_symbols_data(
        tickers: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: Optional[str] = None,
        interval: str = '1d'
    ) -> Dict[str, List[Dict]]:
        """
        Get historical data for multiple symbols (same interface as YahooFinanceProvider)
        
        This is the primary method for Polygon - it downloads flat files from S3
        and extracts data for the requested symbols.
        
        Args:
            tickers: List of symbol tickers
            start_date: Start date for data
            end_date: End date for data
            period: Period string (overrides start_date/end_date) - not commonly used with flat files
            interval: Data interval
        
        Returns:
            Dictionary mapping tickers to their OHLCV data (same format as YahooFinanceProvider)
        """
        # TODO: Implement based on actual Polygon flat file structure
        # This is a placeholder implementation that needs file format details
        
        result = {ticker: [] for ticker in tickers}
        
        try:
            # List files in the bucket (adjust path structure based on actual organization)
            # Common patterns might be:
            # - Date-based: YYYY-MM-DD/ohlcv.csv
            # - Symbol-based: ticker/YYYY-MM-DD.csv
            # - Combined: YYYY-MM-DD/ticker.csv
            
            # Determine date range
            if start_date and end_date:
                # Generate list of dates in range
                current_date = start_date.date() if isinstance(start_date, datetime) else start_date
                end_d = end_date.date() if isinstance(end_date, datetime) else end_date
                
                date_list = []
                while current_date <= end_d:
                    date_list.append(current_date)
                    current_date += pd.Timedelta(days=1)
            else:
                # If no date range, fetch most recent file(s)
                # This logic needs to be adjusted based on actual file structure
                date_list = [date.today()]
            
                # For each date, download and parse the file
                for file_date in date_list:
                    # Try to discover file path first, fallback to default pattern
                    file_key = PolygonProvider._discover_file_path(file_date, interval)
                    if not file_key:
                        file_key = PolygonProvider._get_file_path(file_date, interval)
                    
                    if not file_key:
                        print(f"Could not determine file path for date {file_date} and interval {interval}, skipping...")
                        continue
                    
                    try:
                        # Download file from S3 using download_file to temp file then read
                        # (Polygon's example uses download_file, which works better with their endpoint)
                        import tempfile
                        import os
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.gz') as tmp_file:
                            tmp_path = tmp_file.name
                        
                        try:
                            PolygonProvider._s3_client.download_file(
                                Bucket=PolygonProvider._bucket_name,
                                Key=file_key,
                                Filename=tmp_path
                            )
                            # Read the downloaded file
                            with open(tmp_path, 'rb') as f:
                                file_content = f.read()
                        finally:
                            # Clean up temp file
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
                        
                        # Check if file is gzipped
                        is_gzipped = file_key.endswith('.gz')
                        
                        # Parse file (supports JSON, CSV, gzipped files, etc.)
                        df = PolygonProvider._parse_flat_file(file_content, file_date, is_gzipped=is_gzipped)
                        
                        # Normalize ticker column to uppercase for comparison
                        if 'ticker' in df.columns:
                            df['ticker'] = df['ticker'].astype(str).str.upper()
                        
                        # Filter by requested tickers and extract data
                        requested_tickers_upper = [t.upper() for t in tickers]
                        filtered_df = df[df['ticker'].isin(requested_tickers_upper)]
                        
                        # Convert to dictionary format (same as YahooFinanceProvider)
                        for ticker in tickers:
                            ticker_upper = ticker.upper()
                            ticker_df = filtered_df[filtered_df['ticker'] == ticker_upper]
                            if not ticker_df.empty:
                                for _, row in ticker_df.iterrows():
                                    result[ticker].append({
                                        'timestamp': PolygonProvider._parse_timestamp(row, file_date),
                                        'open': float(row['open']),
                                        'high': float(row['high']),
                                        'low': float(row['low']),
                                        'close': float(row['close']),
                                        'volume': int(row['volume']) if pd.notna(row.get('volume', 0)) else 0
                                    })
                    
                    except PolygonProvider._s3_client.exceptions.NoSuchKey:
                        # File doesn't exist for this date, skip
                        print(f"File not found in S3: {file_key} for date {file_date}")
                        continue
                    except Exception as e:
                        error_msg = str(e)
                        # Check if it's an authentication/permission error
                        if '403' in error_msg or 'Forbidden' in error_msg:
                            print(f"⚠️ Permission denied (403) accessing {file_key}. This could mean:")
                            print(f"  1. Incorrect S3 credentials (access_key_id, secret_access_key)")
                            print(f"  2. The file doesn't exist at this path")
                            print(f"  3. The bucket name is incorrect")
                            print(f"  4. The endpoint URL is incorrect")
                            print(f"  Please verify Polygon provider credentials in the database.")
                        else:
                            print(f"Error processing file {file_key}: {error_msg}")
                        # Don't print full traceback for 403 errors, but continue trying other files
                        if '403' not in error_msg and 'Forbidden' not in error_msg:
                            import traceback
                            traceback.print_exc()
                        continue
            
            # Sort each ticker's data by timestamp
            for ticker in result:
                result[ticker].sort(key=lambda x: x['timestamp'])
            
            return result
            
        except Exception as e:
            print(f"Error fetching Polygon flat file data: {str(e)}")
            raise
    
    @staticmethod
    def _get_file_path(file_date: date, interval: str) -> str:
        """
        Construct S3 file path based on date and interval
        
        Tries common file naming patterns. Will be adjusted based on actual structure discovered.
        
        Args:
            file_date: Date for the file
            interval: Data interval ('1d', '1h', etc.)
        
        Returns:
            S3 object key (file path)
        """
        year = file_date.strftime('%Y')
        month = file_date.strftime('%m')
        date_str = file_date.strftime('%Y-%m-%d')
        
        # Polygon structure: us_stocks_sip/day_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz
        if interval == '1d':
            return f"us_stocks_sip/day_aggs_v1/{year}/{month}/{date_str}.csv.gz"
        elif interval == '1h':
            # For hourly, might be different path structure
            return f"us_stocks_sip/hour_aggs_v1/{year}/{month}/{date_str}.csv.gz"
        else:
            # Default to daily structure
            return f"us_stocks_sip/day_aggs_v1/{year}/{month}/{date_str}.csv.gz"
    
    @staticmethod
    def _discover_file_path(file_date: date, interval: str = '1d') -> Optional[str]:
        """
        Try to discover the actual file path by listing S3 bucket
        
        Args:
            file_date: Date for the file
            interval: Data interval
        
        Returns:
            S3 object key if found, None otherwise
        """
        if not PolygonProvider._s3_client:
            return None
        
        date_str = file_date.strftime('%Y-%m-%d')
        date_patterns = [
            date_str,  # YYYY-MM-DD
            date_str.replace('-', ''),  # YYYYMMDD
            date_str.replace('-', '_'),  # YYYY_MM_DD
        ]
        
        try:
            year = file_date.strftime('%Y')
            month = file_date.strftime('%m')
            
            # Try Polygon stock data structure: us_stocks_sip/day_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz
            prefixes_to_try = [
                f"us_stocks_sip/day_aggs_v1/{year}/{month}/{date_str}",  # Full path
                f"us_stocks_sip/day_aggs_v1/{year}/{month}/",  # Month folder
                f"us_stocks_sip/day_aggs_v1/{year}/",  # Year folder
                f"us_stocks_sip/",  # Base folder
            ]
            
            for prefix in prefixes_to_try:
                try:
                    response = PolygonProvider._s3_client.list_objects_v2(
                        Bucket=PolygonProvider._bucket_name,
                        Prefix=prefix,
                        MaxKeys=100
                    )
                    
                    if 'Contents' in response:
                        for obj in response['Contents']:
                            key = obj['Key']
                            # Check if it matches our date pattern and is a data file
                            if date_str in key and (key.endswith('.csv.gz') or key.endswith('.csv') or key.endswith('.json')):
                                return key
                except Exception as e:
                    # Try next prefix
                    continue
            
        except Exception as e:
            print(f"Error discovering file path: {str(e)}")
        
        return None
    
    @staticmethod
    def _parse_flat_file(file_content: bytes, file_date: date, is_gzipped: bool = False) -> pd.DataFrame:
        """
        Parse flat file content into DataFrame
        
        Supports multiple JSON formats:
        - JSON Lines (one JSON object per line)
        - JSON array of objects
        - Nested JSON structures
        
        Args:
            file_content: Raw file content from S3
            file_date: Date associated with the file
        
        Returns:
            DataFrame with columns: ticker, open, high, low, close, volume, (date/timestamp)
        """
        try:
            # Handle gzipped files
            if is_gzipped:
                import gzip
                try:
                    file_content = gzip.decompress(file_content)
                except Exception as e:
                    print(f"Error decompressing gzip file: {str(e)}")
                    raise
            
            # Decode bytes to string
            try:
                content_str = file_content.decode('utf-8')
            except UnicodeDecodeError:
                # Try latin-1 as fallback
                content_str = file_content.decode('latin-1')
            
            # Try to parse as JSON
            try:
                # First, try parsing as JSON Lines (one object per line)
                json_lines = []
                for line in content_str.strip().split('\n'):
                    line = line.strip()
                    if line:
                        try:
                            json_lines.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                
                if json_lines:
                    # Successfully parsed as JSON Lines
                    df = pd.DataFrame(json_lines)
                else:
                    # Try parsing as single JSON (array or object)
                    parsed_json = json.loads(content_str)
                    
                    if isinstance(parsed_json, list):
                        # Array of objects
                        df = pd.DataFrame(parsed_json)
                    elif isinstance(parsed_json, dict):
                        # Single object or nested structure
                        # Check if it has a 'data' key or similar
                        if 'data' in parsed_json and isinstance(parsed_json['data'], list):
                            df = pd.DataFrame(parsed_json['data'])
                        elif 'results' in parsed_json and isinstance(parsed_json['results'], list):
                            df = pd.DataFrame(parsed_json['results'])
                        else:
                            # Single object, wrap in list
                            df = pd.DataFrame([parsed_json])
                    else:
                        raise ValueError(f"Unexpected JSON structure: {type(parsed_json)}")
            
            except json.JSONDecodeError as e:
                # Not JSON, try CSV as fallback
                print(f"Failed to parse as JSON, trying CSV: {str(e)}")
                df = pd.read_csv(StringIO(content_str))
            
            # Normalize column names (case-insensitive)
            df.columns = df.columns.str.lower().str.strip()
            
            # Map common column name variations to standard names
            column_mapping = {
                'symbol': 'ticker',
                'ticker_symbol': 'ticker',
                's': 'ticker',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'vw': 'volume',
                'n': 'volume',  # trades count sometimes used as volume
            }
            
            # Apply column mapping
            df.rename(columns=column_mapping, inplace=True)
            
            # Ensure we have required columns - check for common variations
            required_columns = {
                'ticker': ['ticker', 'symbol', 'ticker_symbol'],
                'open': ['open', 'o'],
                'high': ['high', 'h'],
                'low': ['low', 'l'],
                'close': ['close', 'c'],
                'volume': ['volume', 'v', 'vw']
            }
            
            # Check and map required columns
            for std_col, variations in required_columns.items():
                if std_col not in df.columns:
                    # Try to find a variation
                    found = False
                    for var in variations:
                        if var in df.columns:
                            df.rename(columns={var: std_col}, inplace=True)
                            found = True
                            break
                    if not found:
                        raise ValueError(f"Missing required column: {std_col} (tried variations: {variations}). Available columns: {list(df.columns)}")
            
            # Ensure numeric columns are numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Filter out rows with missing required data
            df = df.dropna(subset=['ticker', 'open', 'high', 'low', 'close'])
            
            return df
            
        except Exception as e:
            print(f"Error parsing flat file: {str(e)}")
            print(f"File date: {file_date}")
            # Print first 500 chars of content for debugging
            try:
                preview = file_content.decode('utf-8')[:500]
                print(f"File content preview:\n{preview}")
            except:
                pass
            raise
    
    @staticmethod
    def _parse_timestamp(row: pd.Series, file_date: date) -> datetime:
        """
        Parse timestamp from row data
        
        Tries multiple common timestamp/date field names and formats
        
        Args:
            row: DataFrame row
            file_date: Date associated with the file
        
        Returns:
            Timezone-aware datetime
        """
        # Try common timestamp/date column names
        timestamp_fields = ['timestamp', 'date', 'time', 't', 'datetime', 'ticker_date']
        
        ts = None
        for field in timestamp_fields:
            if field in row.index and pd.notna(row[field]):
                try:
                    ts = pd.to_datetime(row[field])
                    break
                except (ValueError, TypeError):
                    continue
        
        # If no timestamp field found, use file_date
        if ts is None:
            # Use file_date and set time to market close (16:00 ET = 20:00 UTC)
            ts = datetime.combine(file_date, datetime.min.time().replace(hour=20))
        else:
            # Ensure we have a date component
            if isinstance(ts, pd.Timestamp):
                ts = ts.to_pydatetime()
            
            # If timestamp is time-only, combine with file_date
            if isinstance(ts, datetime) and ts.date() == datetime(1900, 1, 1).date():
                ts = datetime.combine(file_date, ts.time())
        
        # Convert to timezone-aware
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        
        if timezone.is_naive(ts):
            ts = timezone.make_aware(ts, pytz.UTC)
        
        return ts
    
    @staticmethod
    def list_available_dates(start_date: Optional[date] = None, end_date: Optional[date] = None) -> List[date]:
        """
        List available dates in the S3 bucket
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
        
        Returns:
            List of available dates
        """
        try:
            # List objects in bucket
            response = PolygonProvider._s3_client.list_objects_v2(Bucket=PolygonProvider._bucket_name)
            
            dates = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Extract date from key (adjust pattern based on actual structure)
                    key = obj['Key']
                    # Example: "2024-01-15/stocks_daily.csv" -> "2024-01-15"
                    try:
                        date_str = key.split('/')[0]
                        file_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        
                        if start_date and file_date < start_date:
                            continue
                        if end_date and file_date > end_date:
                            continue
                        
                        dates.append(file_date)
                    except (ValueError, IndexError):
                        continue
            
            return sorted(set(dates))
            
        except Exception as e:
            print(f"Error listing available dates: {str(e)}")
            return []

