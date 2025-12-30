# 🔄 Restart Celery Workers After Database Migration

## Problema
Se você está vendo o erro `no such table: market_data_provider` após migrar para PostgreSQL, o Celery worker provavelmente ainda está usando a conexão antiga do SQLite.

## Solução

### 1. Parar todos os workers Celery

```bash
# Encontrar processos Celery
ps aux | grep celery

# Parar workers (substitua PID pelos números dos processos)
kill -9 <PID>

# Ou parar todos de uma vez
pkill -9 -f celery
```

### 2. Parar Celery Beat (se estiver rodando)

```bash
pkill -9 -f celery-beat
```

### 3. Reiniciar Redis (opcional, mas recomendado)

```bash
sudo service redis-server restart
# ou
redis-cli shutdown
redis-server
```

### 4. Reiniciar Celery Worker

```bash
conda activate trading
cd algo_trading_backend
celery -A algo_trading_backend worker --loglevel=info
```

### 5. Reiniciar Celery Beat (se necessário)

```bash
conda activate trading
cd algo_trading_backend
celery -A algo_trading_backend beat --loglevel=info
```

## Verificar se está funcionando

Teste criando uma tarefa simples e veja se o erro desapareceu.

## Nota Importante

Sempre reinicie os workers Celery após:
- Migrar banco de dados
- Alterar configurações do Django
- Atualizar modelos
- Mudar configurações do Celery















