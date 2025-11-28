#!/bin/bash

# ============================================
# NEXUS AI - Startup Script
# ============================================

echo "=========================================="
echo "🔷 NEXUS AI - Starting Platform"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating from .env.example...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✅ .env file created. Please edit it with your API keys.${NC}"
    echo ""
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python 3 found${NC}"

if ! command_exists docker; then
    echo -e "${YELLOW}⚠️  Docker not found. Some features will be unavailable.${NC}"
else
    echo -e "${GREEN}✅ Docker found${NC}"
fi

if ! command_exists node; then
    echo -e "${YELLOW}⚠️  Node.js not found. Dashboard will not be available.${NC}"
else
    echo -e "${GREEN}✅ Node.js found${NC}"
fi

echo ""

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo -e "${BLUE}🔨 Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

echo -e "${BLUE}🔄 Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"
echo ""

# Install Python dependencies
echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo -e "${GREEN}✅ Python dependencies installed${NC}"
echo ""

# Start Docker services (optional)
if command_exists docker; then
    echo -e "${BLUE}🐳 Starting Docker services...${NC}"
    echo -e "${YELLOW}This will start: Kafka, Neo4j, ClickHouse, MLflow, Prometheus, Grafana${NC}"
    read -p "Start Docker services? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose up -d
        echo -e "${GREEN}✅ Docker services started${NC}"
        echo -e "${BLUE}Waiting for services to be ready...${NC}"
        sleep 10
    else
        echo -e "${YELLOW}⚠️  Docker services not started${NC}"
    fi
    echo ""
fi

# Start API Backend
echo -e "${BLUE}🚀 Starting NEXUS AI API...${NC}"
echo -e "${YELLOW}API will be available at: http://localhost:8000${NC}"
echo -e "${YELLOW}API Docs: http://localhost:8000/docs${NC}"
echo ""

# Open in new terminal or background
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    osascript -e 'tell app "Terminal" to do script "cd '$(pwd)' && source venv/bin/activate && python scripts/run_api.py"'
else
    # Linux
    gnome-terminal -- bash -c "cd $(pwd) && source venv/bin/activate && python scripts/run_api.py; exec bash"
fi

echo -e "${GREEN}✅ API started in new terminal${NC}"
echo ""

# Start Dashboard (if Node.js available)
if command_exists node; then
    echo -e "${BLUE}🎨 Starting Dashboard...${NC}"
    
    # Install dashboard dependencies if needed
    if [ ! -d "dashboard/node_modules" ]; then
        echo -e "${BLUE}📦 Installing Dashboard dependencies...${NC}"
        cd dashboard
        npm install
        cd ..
        echo -e "${GREEN}✅ Dashboard dependencies installed${NC}"
    fi
    
    echo -e "${YELLOW}Dashboard will be available at: http://localhost:3000${NC}"
    echo ""
    
    # Start dashboard in new terminal
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        osascript -e 'tell app "Terminal" to do script "cd '$(pwd)'/dashboard && npm run dev"'
    else
        # Linux
        gnome-terminal -- bash -c "cd $(pwd)/dashboard && npm run dev; exec bash"
    fi
    
    echo -e "${GREEN}✅ Dashboard started in new terminal${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}🎉 NEXUS AI Platform Started!${NC}"
echo "=========================================="
echo ""
echo "📊 Access Points:"
echo "  🎨 Dashboard:    http://localhost:3000"
echo "  📚 API Docs:     http://localhost:8000/docs"
echo "  📈 Grafana:      http://localhost:3001 (admin/admin)"
echo "  🗄️  Neo4j:        http://localhost:7474 (neo4j/nexuspassword)"
echo "  📦 MLflow:       http://localhost:5000"
echo "  📊 Prometheus:   http://localhost:9090"
echo "  🔍 Kafka UI:     http://localhost:8080"
echo ""
echo "🛑 To stop all services:"
echo "  docker-compose down"
echo ""
echo "📖 Documentation: README.md"
echo "🐛 Issues: https://github.com/yourusername/nexus-ai/issues"
echo ""
echo "=========================================="

