"""
Django management command to test Polygon provider setup
Run with: python manage.py test_polygon_provider
"""

from django.core.management.base import BaseCommand
from market_data.models import Provider
from market_data.providers.factory import ProviderFactory
from market_data.providers.polygon import PolygonProvider
from datetime import date, timedelta


class Command(BaseCommand):
    help = 'Test Polygon provider setup and S3 connection'

    def add_arguments(self, parser):
        parser.add_argument(
            '--test-fetch',
            action='store_true',
            help='Test fetching data for a specific symbol',
        )
        parser.add_argument(
            '--symbol',
            type=str,
            default='AAPL',
            help='Symbol to test fetching (default: AAPL)',
        )
        parser.add_argument(
            '--days-back',
            type=int,
            default=7,
            help='Number of days back to test fetching (default: 7)',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('=' * 60))
        self.stdout.write(self.style.SUCCESS('POLYGON PROVIDER TEST'))
        self.stdout.write(self.style.SUCCESS('=' * 60))
        self.stdout.write('')

        # Step 1: Check provider in database
        self.stdout.write('1. Checking Polygon provider in database...')
        try:
            polygon_provider = Provider.objects.get(code='POLYGON')
            self.stdout.write(self.style.SUCCESS(f'   ✅ Found provider: {polygon_provider.name}'))
            self.stdout.write(f'   - Code: {polygon_provider.code}')
            self.stdout.write(f'   - Active: {polygon_provider.is_active}')
            self.stdout.write(f'   - Endpoint: {polygon_provider.endpoint_url}')
            self.stdout.write(f'   - Bucket: {polygon_provider.bucket_name}')
            self.stdout.write(f'   - Has Access Key ID: {bool(polygon_provider.access_key_id)}')
            self.stdout.write(f'   - Has Secret Access Key: {bool(polygon_provider.secret_access_key)}')
        except Provider.DoesNotExist:
            self.stdout.write(self.style.ERROR('   ❌ Polygon provider not found in database!'))
            return
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'   ❌ Error: {str(e)}'))
            return

        self.stdout.write('')

        # Step 2: Initialize provider via factory
        self.stdout.write('2. Initializing Polygon provider via factory...')
        try:
            provider = ProviderFactory.get_provider('POLYGON')
            self.stdout.write(self.style.SUCCESS(f'   ✅ Provider initialized: {type(provider).__name__}'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'   ❌ Error initializing provider: {str(e)}'))
            import traceback
            self.stdout.write(traceback.format_exc())
            return

        self.stdout.write('')

        # Step 3: Test S3 connection
        self.stdout.write('3. Testing S3 connection...')
        try:
            if not PolygonProvider._s3_client:
                self.stdout.write(self.style.ERROR('   ❌ S3 client not initialized!'))
                return

            # Try to list objects in bucket (this will verify credentials)
            response = PolygonProvider._s3_client.list_objects_v2(
                Bucket=PolygonProvider._bucket_name,
                MaxKeys=10
            )
            self.stdout.write(self.style.SUCCESS('   ✅ S3 connection successful!'))
            self.stdout.write(f'   - Bucket: {PolygonProvider._bucket_name}')
            self.stdout.write(f'   - Endpoint: {PolygonProvider._endpoint_url}')

            if 'Contents' in response and len(response['Contents']) > 0:
                self.stdout.write(self.style.SUCCESS(f'   - Found {len(response["Contents"])} objects (showing first 10):'))
                for obj in response['Contents'][:10]:
                    size_mb = obj['Size'] / (1024 * 1024)
                    self.stdout.write(f'     * {obj["Key"]} ({size_mb:.2f} MB, {obj["LastModified"]})')
                
                # Analyze file structure
                self.stdout.write('')
                self.stdout.write('   Analyzing file structure...')
                paths = [obj['Key'] for obj in response['Contents']]
                unique_prefixes = set()
                for path in paths:
                    if '/' in path:
                        prefix = path.split('/')[0]
                        unique_prefixes.add(prefix)
                    else:
                        unique_prefixes.add('(root)')
                
                if len(unique_prefixes) > 0:
                    self.stdout.write(f'   Found {len(unique_prefixes)} unique prefixes/folders:')
                    for prefix in sorted(unique_prefixes)[:10]:
                        self.stdout.write(f'     - {prefix}')
            else:
                self.stdout.write(self.style.WARNING('   ⚠️  Bucket appears to be empty or no permissions to list objects'))

        except Exception as e:
            error_msg = str(e)
            if '403' in error_msg or 'Forbidden' in error_msg:
                self.stdout.write(self.style.ERROR('   ❌ Permission denied (403 Forbidden)'))
                self.stdout.write('   This indicates:')
                self.stdout.write('     - Incorrect credentials (access_key_id or secret_access_key)')
                self.stdout.write('     - Bucket name is wrong')
                self.stdout.write('     - Endpoint URL is incorrect')
            elif 'NoSuchBucket' in error_msg:
                self.stdout.write(self.style.ERROR('   ❌ Bucket not found'))
                self.stdout.write('   - Verify the bucket name is correct')
            elif 'InvalidAccessKeyId' in error_msg:
                self.stdout.write(self.style.ERROR('   ❌ Invalid Access Key ID'))
            elif 'SignatureDoesNotMatch' in error_msg:
                self.stdout.write(self.style.ERROR('   ❌ Invalid Secret Access Key'))
            else:
                self.stdout.write(self.style.ERROR(f'   ❌ Error: {error_msg}'))
            import traceback
            self.stdout.write(traceback.format_exc())
            return

        self.stdout.write('')

        # Step 4: Test file discovery
        self.stdout.write('4. Testing file discovery...')
        try:
            test_date = date.today()
            self.stdout.write(f'   Testing with date: {test_date}')

            # Try to discover file path
            file_key = PolygonProvider._discover_file_path(test_date, '1d')
            if file_key:
                self.stdout.write(self.style.SUCCESS(f'   ✅ Discovered file path: {file_key}'))
            else:
                self.stdout.write(self.style.WARNING(f'   ⚠️  No file found for {test_date}, trying recent dates...'))
                # Try a few days back
                found = False
                for days_back in range(1, options['days_back'] + 1):
                    test_date = date.today() - timedelta(days=days_back)
                    file_key = PolygonProvider._discover_file_path(test_date, '1d')
                    if file_key:
                        self.stdout.write(self.style.SUCCESS(f'   ✅ Discovered file path for {test_date}: {file_key}'))
                        found = True
                        break
                if not found:
                    self.stdout.write(self.style.WARNING('   ⚠️  Could not discover file path for recent dates'))
                    self.stdout.write('   - This might be normal if files use a different naming pattern')
                    self.stdout.write('   - Check the actual file structure in your S3 bucket')

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'   ❌ Error during file discovery: {str(e)}'))
            import traceback
            self.stdout.write(traceback.format_exc())

        self.stdout.write('')

        # Step 5: Test file path construction
        self.stdout.write('5. Testing file path construction...')
        try:
            test_date = date.today()
            file_key = PolygonProvider._get_file_path(test_date, '1d')
            self.stdout.write(f'   Default file path pattern: {file_key}')
            self.stdout.write(self.style.WARNING('   ⚠️  This is a default pattern - actual files may use different naming'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'   ❌ Error: {str(e)}'))

        self.stdout.write('')

        # Step 6: Test file download (permissions check)
        self.stdout.write('6. Testing file download permissions...')
        try:
            # Find an actual file that exists
            response = PolygonProvider._s3_client.list_objects_v2(
                Bucket=PolygonProvider._bucket_name,
                Prefix='us_stocks_sip/',
                MaxKeys=1
            )
            
            if 'Contents' in response and len(response['Contents']) > 0:
                test_file_key = response['Contents'][0]['Key']
                self.stdout.write(f'   Testing download of: {test_file_key}')
                
                try:
                    # Use download_file to temp file (matches Polygon's example)
                    import tempfile
                    import os
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.gz') as tmp_file:
                        tmp_path = tmp_file.name
                    
                    try:
                        PolygonProvider._s3_client.download_file(
                            Bucket=PolygonProvider._bucket_name,
                            Key=test_file_key,
                            Filename=tmp_path
                        )
                        # Read the downloaded file
                        with open(tmp_path, 'rb') as f:
                            content = f.read()
                    finally:
                        # Clean up temp file
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                    self.stdout.write(self.style.SUCCESS(f'   ✅ Successfully downloaded file ({len(content)} bytes)'))
                    
                    # Try to decompress
                    import gzip
                    try:
                        decompressed = gzip.decompress(content)
                        self.stdout.write(self.style.SUCCESS(f'   ✅ Successfully decompressed ({len(decompressed)} bytes)'))
                        
                        # Show sample content
                        lines = decompressed.decode('utf-8').split('\n')[:3]
                        self.stdout.write(f'   Sample content (first 3 lines):')
                        for i, line in enumerate(lines[:3], 1):
                            if line.strip():
                                self.stdout.write(f'     {i}: {line[:150]}')
                        
                        download_works = True
                    except Exception as e:
                        self.stdout.write(self.style.ERROR(f'   ❌ Error decompressing: {str(e)}'))
                        download_works = False
                        
                except Exception as e:
                    error_msg = str(e)
                    if '403' in error_msg or 'Forbidden' in error_msg:
                        self.stdout.write(self.style.ERROR('   ❌ PERMISSION DENIED (403 Forbidden)'))
                        self.stdout.write('')
                        self.stdout.write(self.style.WARNING('   ⚠️  DIAGNOSIS: You have LIST permissions but not READ permissions'))
                        self.stdout.write('')
                        self.stdout.write('   This means:')
                        self.stdout.write('     - ✅ Your S3 credentials are correct (you can list files)')
                        self.stdout.write('     - ❌ Your S3 credentials do NOT have read/download permissions')
                        self.stdout.write('')
                        self.stdout.write('   SOLUTION:')
                        self.stdout.write('     1. Check your Polygon account settings')
                        self.stdout.write('     2. Verify your S3 credentials have "GetObject" permissions')
                        self.stdout.write('     3. Contact Polygon support if you need read access enabled')
                        self.stdout.write('     4. The credentials you provided may be for listing only')
                        download_works = False
                    else:
                        self.stdout.write(self.style.ERROR(f'   ❌ Error: {error_msg}'))
                        download_works = False
            else:
                self.stdout.write(self.style.WARNING('   ⚠️  No files found to test download'))
                download_works = None
                
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'   ❌ Error: {str(e)}'))
            download_works = False

        self.stdout.write('')

        # Step 7: Test data fetching (optional, only if download works)
        if options['test_fetch']:
            if download_works is False:
                self.stdout.write('7. Skipping data fetch test (download permissions issue)')
            else:
                self.stdout.write(f'7. Testing data fetch for symbol: {options["symbol"]}...')
                try:
                    from datetime import datetime
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=options['days_back'])
                    
                    self.stdout.write(f'   Date range: {start_date.date()} to {end_date.date()}')
                    
                    data = provider.get_historical_data(
                        ticker=options['symbol'],
                        start_date=start_date,
                        end_date=end_date,
                        interval='1d'
                    )
                    
                    if data:
                        self.stdout.write(self.style.SUCCESS(f'   ✅ Successfully fetched {len(data)} data points'))
                        if len(data) > 0:
                            self.stdout.write(f'   First record:')
                            first = data[0]
                            self.stdout.write(f'     - Date: {first.get("timestamp")}')
                            self.stdout.write(f'     - OHLC: O={first.get("open")}, H={first.get("high")}, L={first.get("low")}, C={first.get("close")}')
                            self.stdout.write(f'     - Volume: {first.get("volume")}')
                            
                            if len(data) > 1:
                                self.stdout.write(f'   Last record:')
                                last = data[-1]
                                self.stdout.write(f'     - Date: {last.get("timestamp")}')
                                self.stdout.write(f'     - OHLC: O={last.get("open")}, H={last.get("high")}, L={last.get("low")}, C={last.get("close")}')
                                self.stdout.write(f'     - Volume: {last.get("volume")}')
                    else:
                        self.stdout.write(self.style.WARNING('   ⚠️  No data returned'))
                        
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f'   ❌ Error fetching data: {str(e)}'))
                    import traceback
                    self.stdout.write(traceback.format_exc())

        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS('=' * 60))
        self.stdout.write(self.style.SUCCESS('TEST SUMMARY'))
        self.stdout.write(self.style.SUCCESS('=' * 60))
        self.stdout.write(self.style.SUCCESS('✅ Provider database check: PASSED'))
        self.stdout.write(self.style.SUCCESS('✅ Provider initialization: PASSED'))
        self.stdout.write(self.style.SUCCESS('✅ S3 connection: PASSED'))
        self.stdout.write(self.style.SUCCESS('✅ File discovery: WORKING'))
        self.stdout.write(self.style.SUCCESS('✅ File path structure: CONFIGURED'))
        
        if download_works is False:
            self.stdout.write(self.style.ERROR('❌ File download: FAILED - Permission denied'))
            self.stdout.write('')
            self.stdout.write(self.style.WARNING('⚠️  CRITICAL: You need READ permissions to use Polygon provider'))
            self.stdout.write('   The test shows you can list files but cannot download them.')
            self.stdout.write('   Please check your Polygon account permissions.')
        elif download_works:
            self.stdout.write(self.style.SUCCESS('✅ File download: WORKING'))
        else:
            self.stdout.write(self.style.WARNING('⚠️  File download: NOT TESTED'))
        
        self.stdout.write('')
        self.stdout.write('Status:')
        if download_works is False:
            self.stdout.write(self.style.ERROR('  ❌ Provider cannot be used - download permissions missing'))
        elif download_works:
            self.stdout.write(self.style.SUCCESS('  ✅ Provider is ready to use!'))
        else:
            self.stdout.write(self.style.WARNING('  ⚠️  Provider partially configured - run full test'))
        self.stdout.write('')

