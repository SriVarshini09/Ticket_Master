#!/bin/bash

# Support Ticket Router - Deployment Script
# This script helps deploy the application to various platforms

set -e

echo "🎫 Support Ticket Router - Deployment Script"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to create necessary directories
setup_directories() {
    print_status "Setting up directories..."
    mkdir -p templates
    mkdir -p logs
    print_success "Directories created"
}

# Function for local development setup
setup_local() {
    print_status "Setting up local development environment..."
    
    # Check Python
    if ! command_exists python3; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Install dependencies
    print_status "Installing dependencies..."
    pip install -r requirements.txt
    
    # Setup directories
    setup_directories
    
    print_success "Local development environment ready!"
    print_status "Run the application with: python app.py"
}

# Function for Docker deployment
setup_docker() {
    print_status "Setting up Docker deployment..."
    
    # Check Docker
    if ! command_exists docker; then
        print_error "Docker is required but not installed"
        exit 1
    fi
    
    # Build Docker image
    print_status "Building Docker image..."
    docker build -t support-ticket-router .
    
    # Check if docker-compose exists
    if command_exists docker-compose; then
        print_status "Starting with docker-compose..."
        docker-compose up -d --build
        print_success "Application started with docker-compose"
        print_status "Access the application at: http://localhost:5000"
    else
        print_status "Starting Docker container..."
        docker run -d -p 5000:5000 --name ticket-router support-ticket-router
        print_success "Docker container started"
        print_status "Access the application at: http://localhost:5000"
    fi
}

# Function for Heroku deployment
setup_heroku() {
    print_status "Setting up Heroku deployment..."
    
    # Check Heroku CLI
    if ! command_exists heroku; then
        print_error "Heroku CLI is required but not installed"
        print_status "Install from: https://devcenter.heroku.com/articles/heroku-cli"
        exit 1
    fi
    
    # Check if already in a git repo
    if [ ! -d ".git" ]; then
        print_status "Initializing git repository..."
        git init
        git add .
        git commit -m "Initial commit"
    fi
    
    # Login to Heroku
    print_status "Please login to Heroku..."
    heroku login
    
    # Create Heroku app
    read -p "Enter Heroku app name (or press Enter for auto-generated): " app_name
    if [ -z "$app_name" ]; then
        heroku create
    else
        heroku create "$app_name"
    fi
    
    # Deploy to Heroku
    print_status "Deploying to Heroku..."
    git push heroku main
    
    print_success "Deployed to Heroku!"
    heroku open
}

# Function to run tests
run_tests() {
    print_status "Running API tests..."
    
    # Check if server is running
    if curl -s http://localhost:5000/health > /dev/null; then
        print_status "Server is running, executing tests..."
        python test_api.py
    else
        print_warning "Server is not running. Starting server first..."
        python app.py &
        SERVER_PID=$!
        
        # Wait for server to start
        sleep 5
        
        # Run tests
        python test_api.py
        
        # Kill server
        kill $SERVER_PID
    fi
}

# Function to generate sample data
generate_sample_data() {
    print_status "Generating sample data..."
    
    python -c "
import requests
import json

# Sample tickets
samples = [
    {'subject': 'Login issues', 'message': 'Cannot login to my account', 'customer_email': 'user1@test.com'},
    {'subject': 'Billing question', 'message': 'Wrong amount charged', 'customer_email': 'user2@test.com'},
    {'subject': 'URGENT: System down', 'message': 'Critical system failure', 'customer_email': 'admin@test.com'},
    {'subject': 'Feature request', 'message': 'Please add export functionality', 'customer_email': 'user3@test.com'}
]

for sample in samples:
    try:
        response = requests.post('http://localhost:5000/api/tickets', json=sample)
        if response.status_code == 201:
            print(f'Created ticket: {sample[\"subject\"]}')
        else:
            print(f'Failed to create: {sample[\"subject\"]}')
    except:
        print('Server not running. Start the app first.')
        break
"
    
    print_success "Sample data generated!"
}

# Function to show deployment status
show_status() {
    print_status "Checking deployment status..."
    
    # Check if running locally
    if curl -s http://localhost:5000/health > /dev/null; then
        print_success "✅ Application running locally at http://localhost:5000"
    else
        print_warning "❌ Application not running locally"
    fi
    
    # Check Docker
    if command_exists docker && docker ps | grep -q "ticket-router"; then
        print_success "✅ Docker container running"
    else
        print_warning "❌ Docker container not running"
    fi
    
    # Check git status
    if [ -d ".git" ]; then
        print_status "Git status:"
        git status --porcelain
    fi
}

# Function to clean up
cleanup() {
    print_status "Cleaning up..."
    
    # Stop Docker containers
    if command_exists docker; then
        docker stop ticket-router 2>/dev/null || true
        docker rm ticket-router 2>/dev/null || true
    fi
    
    # Stop docker-compose
    if command_exists docker-compose && [ -f "docker-compose.yml" ]; then
        docker-compose down
    fi
    
    # Clean Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  local     Setup local development environment"
    echo "  docker    Deploy with Docker"
    echo "  heroku    Deploy to Heroku"
    echo "  test      Run API tests"
    echo "  sample    Generate sample data"
    echo "  status    Show deployment status"
    echo "  cleanup   Clean up containers and cache"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 local          # Setup for local development"
    echo "  $0 docker         # Deploy with Docker"
    echo "  $0 test           # Run tests"
}

# Main script logic
case "${1:-help}" in
    "local")
        setup_local
        ;;
    "docker")
        setup_docker
        ;;
    "heroku")
        setup_heroku
        ;;
    "test")
        run_tests
        ;;
    "sample")
        generate_sample_data
        ;;
    "status")
        show_status
        ;;
    "cleanup")
        cleanup
        ;;
    "help"|*)
        show_usage
        ;;
esac