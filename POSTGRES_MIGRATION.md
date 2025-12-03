# PostgreSQL Migration Guide

## Prerequisites

1. **Install PostgreSQL** (if not already installed):
   ```bash
   sudo apt-get update
   sudo apt-get install postgresql postgresql-contrib
   ```

2. **Start PostgreSQL service**:
   ```bash
   sudo systemctl start postgresql
   sudo systemctl enable postgresql  # Enable on boot
   ```

## Setup Database

### Option 1: Use the setup script (Recommended)

```bash
cd algo_trading_backend
./setup_postgres.sh
```

### Option 2: Manual setup

1. **Create database and user**:
   ```bash
   sudo -u postgres psql
   ```

2. **In PostgreSQL prompt, run**:
   ```sql
   CREATE DATABASE trading_db;
   CREATE USER trading_user WITH PASSWORD 'trading_password';
   ALTER USER trading_user CREATEDB;
   GRANT ALL PRIVILEGES ON DATABASE trading_db TO trading_user;
   \q
   ```

## Run Migrations

After setting up the database:

```bash
conda activate trading
cd algo_trading_backend
python manage.py migrate
```

## Old SQLite File

The old SQLite database file (`db.sqlite3`) has been deleted. 
All data will be fresh in PostgreSQL after running migrations.

## Custom Configuration

To use different database credentials, set environment variables:

```bash
export DB_NAME=your_db_name
export DB_USER=your_db_user
export DB_PASSWORD=your_password
export DB_HOST=localhost
export DB_PORT=5432
```

Or create a `.env` file and use `python-decouple` or `django-environ`.

## Verify Connection

Test the connection:

```bash
python manage.py dbshell
```

You should see the PostgreSQL prompt if connection is successful.

## Benefits of PostgreSQL over SQLite

- ✅ Supports concurrent writes (no "database is locked" errors)
- ✅ Better performance for large datasets
- ✅ Production-ready
- ✅ Supports advanced features (full-text search, JSON fields, etc.)
- ✅ Better for Celery tasks with multiple workers

