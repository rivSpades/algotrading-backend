# 🔧 Fix PostgreSQL Authentication Error

## Problema
```
password authentication failed for user "trading_user"
```

## Solução Rápida

### Opção 1: Usar usuário postgres padrão (Mais fácil)

Edite `algo_trading_backend/settings.py` e mude para:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'trading_db',
        'USER': 'postgres',
        'PASSWORD': 'postgres',  # Ou sua senha do postgres
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

Depois execute:
```bash
sudo -u postgres psql -c "CREATE DATABASE trading_db;"
conda activate trading
cd algo_trading_backend
python manage.py migrate
```

### Opção 2: Criar usuário trading_user (Recomendado para produção)

Execute no terminal:

```bash
sudo -u postgres psql
```

No prompt do PostgreSQL, execute:

```sql
-- Criar usuário
CREATE USER trading_user WITH PASSWORD 'trading_password' CREATEDB;

-- Criar banco de dados
CREATE DATABASE trading_db OWNER trading_user;

-- Dar permissões
GRANT ALL PRIVILEGES ON DATABASE trading_db TO trading_user;

-- Conectar ao banco e dar permissões no schema
\c trading_db
GRANT ALL ON SCHEMA public TO trading_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO trading_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO trading_user;

-- Sair
\q
```

Depois execute as migrações:
```bash
conda activate trading
cd algo_trading_backend
python manage.py migrate
```

### Opção 3: Usar arquivo SQL

```bash
sudo -u postgres psql -f create_postgres_db.sql
conda activate trading
cd algo_trading_backend
python manage.py migrate
```

## Verificar se funcionou

```bash
python manage.py dbshell
```

Se conseguir entrar no shell do PostgreSQL, está funcionando! Digite `\q` para sair.




