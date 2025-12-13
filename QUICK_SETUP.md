# Quick PostgreSQL Setup

Se você está tendo erro de autenticação, execute estes comandos:

## 1. Criar usuário e banco de dados

```bash
sudo -u postgres psql
```

No prompt do PostgreSQL, execute:

```sql
DROP DATABASE IF EXISTS trading_db;
DROP USER IF EXISTS trading_user;
CREATE USER trading_user WITH PASSWORD 'trading_password' CREATEDB;
CREATE DATABASE trading_db OWNER trading_user;
GRANT ALL PRIVILEGES ON DATABASE trading_db TO trading_user;
\c trading_db
GRANT ALL ON SCHEMA public TO trading_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO trading_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO trading_user;
\q
```

## 2. Ou use o arquivo SQL

```bash
sudo -u postgres psql -f create_postgres_db.sql
```

## 3. Executar migrações

```bash
conda activate trading
cd algo_trading_backend
python manage.py migrate
```

## 4. Verificar conexão

```bash
python manage.py dbshell
```

Se conseguir entrar no shell do PostgreSQL, está tudo funcionando!

## Troubleshooting

Se ainda tiver erro de autenticação:

1. Verifique se o PostgreSQL está rodando:
   ```bash
   sudo service postgresql status
   ```

2. Verifique se o usuário existe:
   ```bash
   sudo -u postgres psql -c "\du"
   ```

3. Teste a conexão manualmente:
   ```bash
   psql -U trading_user -d trading_db -h localhost
   # Password: trading_password
   ```








