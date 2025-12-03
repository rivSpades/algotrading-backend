#!/bin/bash
# PostgreSQL Setup Script for Trading Platform

echo "=== PostgreSQL Setup for Trading Platform ==="
echo ""

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo "❌ PostgreSQL não está instalado."
    echo ""
    echo "Para instalar no WSL/Ubuntu:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install postgresql postgresql-contrib"
    echo ""
    echo "Após instalar, execute este script novamente."
    exit 1
fi

# Check if PostgreSQL is running
if ! pg_isready -h localhost -p 5432 &> /dev/null; then
    echo "❌ PostgreSQL não está rodando."
    echo ""
    echo "Para iniciar:"
    echo "  sudo service postgresql start"
    echo ""
    echo "Para habilitar no boot:"
    echo "  sudo systemctl enable postgresql"
    echo ""
    echo "Após iniciar, execute este script novamente."
    exit 1
fi

echo "✅ PostgreSQL está instalado e rodando."
echo ""

# Create database
echo "📦 Criando banco de dados 'trading_db'..."
sudo -u postgres psql -c "CREATE DATABASE trading_db;" 2>&1 | grep -v "already exists" || echo "✅ Banco de dados criado ou já existe."

# Create user
echo "👤 Criando usuário 'trading_user'..."
sudo -u postgres psql -c "CREATE USER trading_user WITH PASSWORD 'trading_password';" 2>&1 | grep -v "already exists" || echo "✅ Usuário criado ou já existe."

# Grant privileges
echo "🔐 Concedendo privilégios..."
sudo -u postgres psql -c "ALTER USER trading_user CREATEDB;" 2>&1 | grep -v "ERROR" || true
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE trading_db TO trading_user;" 2>&1 | grep -v "ERROR" || true

echo ""
echo "=== ✅ Setup concluído! ==="
echo ""
echo "Configurações do banco de dados:"
echo "  📊 Database: trading_db"
echo "  👤 User: trading_user"
echo "  🔑 Password: trading_password"
echo "  🌐 Host: localhost"
echo "  🔌 Port: 5432"
echo ""
echo "Para usar credenciais diferentes, defina variáveis de ambiente:"
echo "  export DB_NAME=seu_db"
echo "  export DB_USER=seu_user"
echo "  export DB_PASSWORD=sua_senha"
echo "  export DB_HOST=localhost"
echo "  export DB_PORT=5432"
echo ""
echo "📝 Próximos passos:"
echo "  1. conda activate trading"
echo "  2. cd algo_trading_backend"
echo "  3. python manage.py migrate"
echo ""
echo "Para migrar dados do SQLite (se necessário):"
echo "  python manage.py dumpdata > data_backup.json  # No SQLite"
echo "  python manage.py migrate                       # No PostgreSQL"
echo "  python manage.py loaddata data_backup.json     # No PostgreSQL"

