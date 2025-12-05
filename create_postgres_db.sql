-- PostgreSQL Database Setup Script
-- Execute with: sudo -u postgres psql -f create_postgres_db.sql

-- Drop existing database and user if they exist
DROP DATABASE IF EXISTS trading_db;
DROP USER IF EXISTS trading_user;

-- Create user
CREATE USER trading_user WITH PASSWORD 'trading_password' CREATEDB;

-- Create database
CREATE DATABASE trading_db OWNER trading_user;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE trading_db TO trading_user;

-- Connect to the database and grant schema privileges
\c trading_db
GRANT ALL ON SCHEMA public TO trading_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO trading_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO trading_user;

-- Verify
\du trading_user
\l trading_db




