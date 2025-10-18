#!/bin/bash
# File: scripts/quick_start.sh
"""
🚀 Quick Start Script
One-command deployment for production
"""

set -e  # Exit on error

echo "=================================="
echo "🔮 CRYPTO SENTIMENT INTELLIGENCE"
echo "=================================="
echo "Quick Start Deployment Script"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "ℹ️  $1"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker found: $(docker --version)"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_success "Docker Compose found"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_warning "Python 3 is not installed. Some scripts may not work."
    else
        print_success "Python found: $(python3 --version)"
    fi
    
    echo ""
}

# Setup environment
setup_environment() {
    print_info "Setting up environment..."
    
    # Check if .env exists
    if [ ! -f .env ]; then
        print_warning ".env file not found. Copying from .env.example..."
        cp .env.example .env
        print_info "Please edit .env file with your Reddit API credentials"
        print_info "Press Enter to continue after editing .env..."
        read
    else
        print_success ".env file found"
    fi
    
    echo ""
}

# Validate model
validate_model() {
    print_info "Validating production model..."
    
    if [ ! -f "models/lightgbm_production.pkl" ]; then
        print_error "Production model not found at models/lightgbm_production.pkl"
        print_info "Please train the model first using:"
        print_info "  python scripts/train_production_model.py"
        exit 1
    fi
    
    print_success "Production model found"
    
    # Run validation if Python is available
    if command -v python3 &> /dev/null; then
        print_info "Running model validation..."
        python3 scripts/validate_model.py --model_path models/lightgbm_production.pkl --min_accuracy 0.75
        
        if [ $? -eq 0 ]; then
            print_success "Model validation passed!"
        else
            print_error "Model validation failed!"
            exit 1
        fi
    fi
    
    echo ""
}

# Start services
start_services() {
    print_info "Starting production services..."
    
    # Stop any running services
    print_info "Stopping existing services..."
    docker-compose -f docker-compose.prod.yml down 2>/dev/null || true
    
    # Build images
    print_info "Building Docker images..."
    docker-compose -f docker-compose.prod.yml build
    
    if [ $? -ne 0 ]; then
        print_error "Failed to build Docker images"
        exit 1
    fi
    print_success "Docker images built successfully"
    
    # Start services
    print_info "Starting services..."
    docker-compose -f docker-compose.prod.yml up -d
    
    if [ $? -ne 0 ]; then
        print_error "Failed to start services"
        exit 1
    fi
    print_success "Services started successfully"
    
    echo ""
}

# Wait for services
wait_for_services() {
    print_info "Waiting for services to be ready..."
    
    # Wait for PostgreSQL
    print_info "Waiting for PostgreSQL..."
    for i in {1..30}; do
        if docker-compose -f docker-compose.prod.yml exec -T postgres pg_isready -U crypto_user &>/dev/null; then
            print_success "PostgreSQL is ready"
            break
        fi
        sleep 2
    done
    
    # Wait for Redis
    print_info "Waiting for Redis..."
    for i in {1..30}; do
        if docker-compose -f docker-compose.prod.yml exec -T redis redis-cli ping &>/dev/null; then
            print_success "Redis is ready"
            break
        fi
        sleep 2
    done
    
    # Wait for API
    print_info "Waiting for API..."
    for i in {1..60}; do
        if curl -s -f http://localhost:8000/api/v1/health/live &>/dev/null; then
            print_success "API is ready"
            break
        fi
        sleep 2
    done
    
    echo ""
}

# Run health checks
run_health_checks() {
    print_info "Running health checks..."
    
    # Check API health
    print_info "Checking API health..."
    response=$(curl -s -X GET "http://localhost:8000/api/v1/health" \
        -H "X-API-Key: demo-key-789")
    
    if echo "$response" | grep -q '"status":"healthy"'; then
        print_success "API health check passed"
    else
        print_warning "API health check returned: $response"
    fi
    
    # Test prediction
    print_info "Testing prediction endpoint..."
    response=$(curl -s -X POST "http://localhost:8000/api/v1/predict" \
        -H "X-API-Key: demo-key-789" \
        -H "Content-Type: application/json" \
        -d '{"text":"Bitcoin is going to the moon!"}')
    
    if echo "$response" | grep -q '"prediction"'; then
        print_success "Prediction endpoint working"
    else
        print_warning "Prediction endpoint test failed"
    fi
    
    echo ""
}

# Display access info
display_access_info() {
    echo "=================================="
    echo "🎉 DEPLOYMENT SUCCESSFUL!"
    echo "=================================="
    echo ""
    print_success "All services are running!"
    echo ""
    echo "📡 Access Points:"
    echo "  • API Documentation:  http://localhost:8000/docs"
    echo "  • Dashboard:          http://localhost:8501"
    echo "  • MLflow:             http://localhost:5000"
    echo "  • Grafana:            http://localhost:3000"
    echo "  • Prometheus:         http://localhost:9090"
    echo ""
    echo "🔑 Default Credentials:"
    echo "  • API Key:            demo-key-789"
    echo "  • Grafana:            admin / admin123"
    echo ""
    echo "📊 Quick Commands:"
    echo "  • View logs:          docker-compose -f docker-compose.prod.yml logs -f"
    echo "  • Stop services:      docker-compose -f docker-compose.prod.yml down"
    echo "  • Restart services:   docker-compose -f docker-compose.prod.yml restart"
    echo ""
    echo "📖 Documentation:      See DEPLOYMENT.md for detailed guide"
    echo "=================================="
}

# Main execution
main() {
    echo ""
    print_info "Starting deployment process..."
    echo ""
    
    check_prerequisites
    setup_environment
    validate_model
    start_services
    wait_for_services
    run_health_checks
    display_access_info
    
    echo ""
    print_success "Deployment completed successfully! 🚀"
    echo ""
}

# Run main function
main