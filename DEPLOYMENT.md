# File: DEPLOYMENT.md

# 🚀 PRODUCTION DEPLOYMENT GUIDE

Complete guide for deploying the Crypto Sentiment Intelligence System to production.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Production Deployment](#production-deployment)
4. [Monitoring & Maintenance](#monitoring--maintenance)
5. [Troubleshooting](#troubleshooting)

---

## 🎯 Prerequisites

### System Requirements

```bash
# Hardware
- CPU: 4+ cores recommended
- RAM: 8GB minimum, 16GB recommended
- Storage: 20GB free space
- GPU: Optional (CUDA-capable for faster inference)

# Software
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.9+
- Git

Required Accounts

Reddit API credentials (for data collection)
Docker Hub account (optional, for custom images)



🔧 Local Development Setup

1. Clone Repository

git clone <repository-url>
cd crypto-sentiment-mlops

2. Environment Configuration

# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env

# Required variables:
# - REDDIT_CLIENT_ID
# - REDDIT_CLIENT_SECRET
# - POSTGRES_PASSWORD

3. Start Development Services

# Start database and MLflow
docker compose up -d postgres redis mlflow

# Verify services
docker compose ps

# Check logs
docker compose logs -f mlflow

4. Install Python Dependencies

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

5. Verify Model

# Validate production model
python scripts/validate_model.py \
    --model_path models/lightgbm_production.pkl \
    --min_accuracy 0.80

# Expected output:
# ✅ MODEL VALIDATION PASSED! Model is ready for deployment.

6. Test API Locally

# Start API server
uvicorn src.api.main:app --reload --port 8000

# In another terminal, test endpoint
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "X-API-Key: demo-key-789" \
     -H "Content-Type: application/json" \
     -d '{"text": "Bitcoin is going to the moon!"}'

7. Run Dashboard Locally

# Start Streamlit dashboard
streamlit run src/monitoring/dashboard/streamlit_app.py

# Open browser: http://localhost:8501



🚀 Production Deployment

Option 1: Docker Compose (Recommended)

Step 1: Build Images

# Build all images
docker compose -f docker-compose.prod.yml build

# Verify images
docker images | grep crypto-sentiment

Step 2: Start Production Stack

# Start all services
docker compose -f docker-compose.prod.yml up -d

# Verify all services are running
docker compose -f docker-compose.prod.yml ps

# Expected output:
# NAME                    STATUS
# crypto_postgres_prod    Up (healthy)
# crypto_redis_prod       Up (healthy)
# crypto_mlflow_prod      Up
# crypto_api_prod         Up (healthy)
# crypto_dashboard_prod   Up
# crypto_nginx_prod       Up

Step 3: Verify Deployment

# Check API health
curl http://localhost/api/v1/health

# Test prediction
curl -X POST "http://localhost/api/v1/predict" \
     -H "X-API-Key: demo-key-789" \
     -H "Content-Type: application/json" \
     -d '{"text": "Ethereum is pumping!"}'

# Access dashboard
# Open browser: http://localhost:8501

# Access MLflow
# Open browser: http://localhost:5000

Step 4: Monitor Logs

# View all logs
docker compose -f docker-compose.prod.yml logs -f

# View specific service
docker compose -f docker-compose.prod.yml logs -f api

# Check for errors
docker compose -f docker-compose.prod.yml logs api | grep ERROR

Option 2: Kubernetes (Advanced)

Prerequisites

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

Deploy to Kubernetes

# Create namespace
kubectl create namespace crypto-sentiment

# Deploy PostgreSQL
helm install postgres bitnami/postgresql \
  --namespace crypto-sentiment \
  --set auth.username=crypto_user \
  --set auth.password=crypto_password_2024 \
  --set auth.database=crypto_sentiment_db

# Deploy Redis
helm install redis bitnami/redis \
  --namespace crypto-sentiment \
  --set auth.enabled=false

# Apply application manifests
kubectl apply -f k8s/ -n crypto-sentiment

# Verify deployment
kubectl get pods -n crypto-sentiment
kubectl get services -n crypto-sentiment



📊 Monitoring & Maintenance

Health Checks

# API Health
curl http://localhost/api/v1/health

# Expected response:
# {
#   "status": "healthy",
#   "model_status": "healthy",
#   "redis_status": "healthy",
#   "uptime_seconds": 3600.5
# }

# Liveness probe
curl http://localhost/api/v1/health/live

# Readiness probe
curl http://localhost/api/v1/health/ready

Performance Monitoring

# Get metrics
curl http://localhost/api/v1/metrics \
     -H "X-API-Key: demo-key-789"

# Get detailed metrics
curl http://localhost/api/v1/metrics/detailed \
     -H "X-API-Key: demo-key-789"

# Get hourly trends
curl "http://localhost/api/v1/metrics/hourly?hours=24" \
     -H "X-API-Key: demo-key-789"

Grafana Dashboards

# Access Grafana
# Open browser: http://localhost:3000
# Default credentials: admin / admin123

# Import dashboard
# 1. Go to Dashboards > Import
# 2. Upload docker/grafana/dashboards/crypto-sentiment-dashboard.json
# 3. Select Prometheus datasource

Log Management

# View application logs
docker compose -f docker-compose.prod.yml logs -f api

# Export logs
docker compose -f docker-compose.prod.yml logs api > api_logs.txt

# Search logs for errors
docker compose -f docker-compose.prod.yml logs api | grep -i error

# Real-time error monitoring
docker compose -f docker-compose.prod.yml logs -f api | grep -i error

Backup & Recovery

Database Backup

# Backup PostgreSQL
docker exec crypto_postgres_prod pg_dump \
  -U crypto_user crypto_sentiment_db > backup_$(date +%Y%m%d).sql

# Restore from backup
docker exec -i crypto_postgres_prod psql \
  -U crypto_user crypto_sentiment_db < backup_20250111.sql

Model Backup

# Backup models directory
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Restore models
tar -xzf models_backup_20250111.tar.gz

Automated Retraining

# Manual retraining
python scripts/collect_more_data.py
python scripts/auto_label_data.py --filter_confidence
python scripts/train_production_model.py --augment_data

# Validate new model
python scripts/validate_model.py \
    --model_path models/lightgbm_production.pkl \
    --min_accuracy 0.80

# If validation passes, restart API
docker compose -f docker-compose.prod.yml restart api



🐛 Troubleshooting

Common Issues

Issue 1: Model Not Loading

# Symptoms
# - API returns 503 errors
# - Health check shows "model_status": "unhealthy"

# Solution
# 1. Check if model file exists
ls -lh models/lightgbm_production.pkl

# 2. Validate model
python scripts/validate_model.py --model_path models/lightgbm_production.pkl

# 3. Check API logs
docker compose logs api | grep -i "model"

# 4. Restart API
docker compose restart api

Issue 2: High Response Time

# Symptoms
# - Response time > 500ms
# - Dashboard shows slow performance

# Solution
# 1. Check system resources
docker stats

# 2. Scale API horizontally
docker compose -f docker-compose.prod.yml up -d --scale api=3

# 3. Enable Redis caching
# Verify Redis is running
docker compose ps redis

# 4. Optimize model
# Use smaller model or quantization

Issue 3: Database Connection Errors

# Symptoms
# - "Connection refused" errors
# - MLflow not tracking experiments

# Solution
# 1. Check PostgreSQL is running
docker compose ps postgres

# 2. Test connection
docker exec crypto_postgres_prod psql \
  -U crypto_user -d crypto_sentiment_db -c "SELECT 1"

# 3. Check credentials in .env
cat .env | grep POSTGRES

# 4. Restart database
docker compose restart postgres

Issue 4: Redis Connection Errors

# Symptoms
# - Caching not working
# - "Redis unavailable" warnings

# Solution
# 1. Check Redis is running
docker compose ps redis

# 2. Test connection
docker exec crypto_redis_prod redis-cli ping

# 3. Restart Redis
docker compose restart redis

Performance Optimization

1. Enable GPU Acceleration

# Install NVIDIA Docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Update docker-compose.prod.yml
# Add to api service:
#   deploy:
#     resources:
#       reservations:
#         devices:
#           - driver: nvidia
#             count: 1
#             capabilities: [gpu]

2. Horizontal Scaling

# Scale API instances
docker compose -f docker-compose.prod.yml up -d --scale api=3

# Nginx will automatically load balance across instances

3. Database Optimization

-- Connect to PostgreSQL
docker exec -it crypto_postgres_prod psql -U crypto_user -d crypto_sentiment_db

-- Create indexes
CREATE INDEX idx_created_at ON experiments(created_at);
CREATE INDEX idx_status ON runs(status);

-- Analyze tables
ANALYZE;

Security Hardening

1. Change Default Passwords

# Update .env
POSTGRES_PASSWORD=<strong_password>
GRAFANA_ADMIN_PASSWORD=<strong_password>

# Restart services
docker compose -f docker-compose.prod.yml restart

2. Enable HTTPS

# Generate SSL certificate
sudo apt-get install certbot
sudo certbot certonly --standalone -d your-domain.com

# Update nginx.conf
# Add SSL configuration
server {
    listen 443 ssl http2;
    ssl_certificate /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    ...
}

# Restart Nginx
docker compose restart nginx

3. API Key Management

# Update src/api/dependencies.py
# Use environment variables for API keys
import os

VALID_API_KEYS = {
    os.getenv("API_KEY_DEV"): "development",
    os.getenv("API_KEY_PROD"): "production",
}



📞 Support & Maintenance

Regular Maintenance Tasks

Daily
Check health endpoints
Monitor error logs
Review performance metrics

Weekly
Backup database
Backup models
Review system resources
Update dependencies

Monthly
Retrain model with new data
Security updates
Performance optimization review
Cost analysis

Getting Help

Documentation: See README.md and code comments
Issues: Check GitHub Issues
Logs: docker compose logs -f
Metrics: http://localhost:3000 (Grafana)



🎉 Success Checklist

All services running (postgres, redis, mlflow, api, dashboard, nginx)
API health check returns "healthy"
Model validation passes (accuracy > 80%)
Prediction endpoint works
Dashboard accessible
MLflow tracking works
Monitoring dashboards configured
Backups scheduled
Documentation updated

Congratulations! Your Crypto Sentiment Intelligence System is now in production! 🚀



Last Updated: January 11, 2025  
Version: 2.0.0  
Deployment Status: ✅ Production Ready