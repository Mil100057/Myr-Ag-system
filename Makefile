# Myr-Ag RAG System Makefile
# Provides convenient shortcuts for common operations

.PHONY: help start stop restart restart-ui restart-api hard-restart-ui hard-restart-api status logs clean install setup

# Default target
help:
	@echo "🚀 Myr-Ag RAG System Management"
	@echo "================================"
	@echo ""
	@echo "Available commands:"
	@echo "  start        - Start all services (API + UI)"
	@echo "  stop         - Stop all services"
	@echo "  restart      - Restart all services"
	@echo "  restart-ui   - Restart UI only"
	@echo "  restart-api  - Restart API only"
	@echo "  hard-restart-ui  - Hard restart UI (clear cache)"
	@echo "  hard-restart-api - Hard restart API (clear cache)"
	@echo "  status       - Show service status"
	@echo "  logs         - Show service logs"
	@echo "  clean        - Clean up temporary files"
	@echo "  install      - Install dependencies"
	@echo "  setup        - Initial setup"
	@echo ""
	@echo "MPS & Device Management:"
	@echo "  mps-info     - Show MPS device information"
	@echo "  mps-cleanup  - Clean up MPS memory"
	@echo "  device-test  - Test device configuration"
	@echo "  mps-monitor  - Monitor MPS performance"
	@echo "  start-api-mps - Start API with MPS monitoring"
	@echo ""
	@echo "Examples:"
	@echo "  make start"
	@echo "  make stop"
	@echo "  make restart-ui"
	@echo "  make restart-api"
	@echo "  make hard-restart-ui"
	@echo "  make hard-restart-api"
	@echo "  make status"
	@echo "  make logs"
	@echo "  make mps-info"
	@echo "  make device-test"
	@echo "  make mps-cleanup"
	@echo ""

# Start all services
start:
	@echo "🚀 Starting Myr-Ag RAG System..."
	@./myr-ag.sh start

# Stop all services
stop:
	@echo "🛑 Stopping Myr-Ag RAG System..."
	@./myr-ag.sh stop

# Restart all services
restart:
	@echo "🔄 Restarting Myr-Ag RAG System..."
	@./myr-ag.sh restart

# Restart UI only
restart-ui:
	@echo "🎨 Restarting UI only..."
	@./myr-ag.sh stop-ui
	@./myr-ag.sh start-ui

# Restart API only
restart-api:
	@echo "🔧 Restarting API only..."
	@./myr-ag.sh stop-api
	@./myr-ag.sh start-api

# Hard restart UI (clear Python cache)
hard-restart-ui:
	@echo "🎨 Hard restarting UI (clearing cache)..."
	@./myr-ag.sh hard-restart-ui

# Hard restart API (clear Python cache)
hard-restart-api:
	@echo "🔧 Hard restarting API (clearing cache)..."
	@./myr-ag.sh hard-restart-api

# Show service status
status:
	@./myr-ag.sh status

# Show service logs
logs:
	@./myr-ag.sh logs

# Clean up temporary files
clean:
	@echo "🧹 Cleaning up..."
	@./myr-ag.sh clean


# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	@bash -c "source venv/bin/activate && python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt"

# Initial setup
setup:
	@echo "🔧 Setting up Myr-Ag RAG System..."
	@echo "1. Creating virtual environment with Python 3.11..."
	@python3.11 -m venv venv
	@echo "2. Activating virtual environment and installing dependencies..."
	@bash -c "source venv/bin/activate && python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt"
	@echo "3. Creating necessary directories..."
	@mkdir -p data/uploads data/processed data/vector_db logs
	@echo "✅ Setup completed!"
	@echo ""
	@echo "Next steps:"
	@echo "  make start    - Start the application"
	@echo "  make status   - Check service status"


# Quick start (alias for start)
run: start

# Quick stop (alias for stop)
kill: stop

# Show help (alias for help)
h: help

# MPS & Device Management Commands
.PHONY: mps-info mps-cleanup device-test mps-monitor

# Show MPS device information
mps-info:
	@echo "📊 MPS Device Information"
	@echo "========================="
	@bash -c "source venv/bin/activate && python3 -c \"from src.utils.device_manager import get_mps_info; import json; print(json.dumps(get_mps_info(), indent=2))\""

# Clean up MPS memory
mps-cleanup:
	@echo "🧹 Cleaning up MPS memory..."
	@bash -c "source venv/bin/activate && python3 -c \"from src.utils.device_manager import device_manager; device_manager.cleanup_memory('mps'); print('✅ MPS memory cleaned successfully')\""

# Test device configuration
device-test:
	@echo "🧪 Testing Device Configuration"
	@echo "==============================="
	@bash -c "source venv/bin/activate && python3 -c \"from src.utils.device_manager import get_device; print('General device:', get_device('general')); print('Embedding device:', get_device('embedding')); print('Docling device:', get_device('docling')); print('OCR device:', get_device('ocr'))\""

# Monitor MPS performance (requires API to be running)
mps-monitor:
	@echo "📈 MPS Performance Monitor"
	@echo "=========================="
	@echo "Checking API MPS status..."
	@curl -s http://localhost:8199/system/mps-info 2>/dev/null | python3 -m json.tool || echo "❌ API not running or MPS info not available"

# Start API with MPS monitoring
start-api-mps:
	@echo "🚀 Starting API with MPS monitoring..."
	@echo "MPS configuration will be logged during startup"
	@bash -c "source venv/bin/activate && python3 run_api.py"
