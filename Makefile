# Myr-Ag RAG System Makefile
# Provides convenient shortcuts for common operations

.PHONY: help start stop restart status logs clean install setup

# Default target
help:
	@echo "🚀 Myr-Ag RAG System Management"
	@echo "================================"
	@echo ""
	@echo "Available commands:"
	@echo "  start        - Start all services (API + UI)"
	@echo "  stop         - Stop all services"
	@echo "  restart      - Restart all services"
	@echo "  status       - Show service status"
	@echo "  logs         - Show service logs"
	@echo "  clean        - Clean up temporary files"
	@echo "  install      - Install dependencies"
	@echo "  setup        - Initial setup"
	@echo ""
	@echo "Examples:"
	@echo "  make start"
	@echo "  make stop"
	@echo "  make status"
	@echo "  make logs"
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
