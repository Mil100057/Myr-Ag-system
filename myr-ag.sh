#!/bin/bash

# Myr-Ag RAG System Management Script
# Simple shell script for managing the application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Functions
print_header() {
    echo -e "${BLUE}🚀 Myr-Ag RAG System Management${NC}"
    echo "=========================================="
    echo
}

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
    echo -e "${BLUE}ℹ️  $1${NC}"
}

check_venv() {
    if [[ ! -d "venv" ]]; then
        print_error "Virtual environment not found!"
        echo "Please run: make setup"
        exit 1
    fi
}

start_services() {
    print_header
    print_info "Starting Myr-Ag RAG System..."
    
    check_venv
    
    # Start API
    print_info "Starting API backend..."
    source venv/bin/activate
    nohup python run_api.py > logs/api.log 2>&1 &
    API_PID=$!
    echo $API_PID > .api.pid
    
    # Wait for API to start
    sleep 5
    
    # Start UI
    print_info "Starting UI frontend..."
    nohup python run_ui.py > logs/ui.log 2>&1 &
    UI_PID=$!
    echo $UI_PID > .ui.pid
    
    # Wait for UI to start
    sleep 3
    
    print_success "All services started!"
    echo
    echo "🌐 API: http://localhost:8199"
    echo "🎨 UI: http://localhost:7860"
    echo "📚 Health: http://localhost:8199/health"
    echo
    echo "📝 Logs:"
    echo "  API: logs/api.log"
    echo "  UI:  logs/ui.log"
    echo
    echo "To stop: ./myr-ag.sh stop"
}

stop_services() {
    print_header
    print_info "Stopping Myr-Ag RAG System..."
    
    # Stop API
    if [[ -f ".api.pid" ]]; then
        API_PID=$(cat .api.pid)
        if kill -0 $API_PID 2>/dev/null; then
            print_info "Stopping API (PID: $API_PID)..."
            kill $API_PID
            sleep 2
            if kill -0 $API_PID 2>/dev/null; then
                print_warning "API still running, force killing..."
                kill -9 $API_PID
            fi
            print_success "API stopped"
        else
            print_warning "API PID file exists but process not running"
        fi
        rm -f .api.pid
    else
        print_info "No API PID file found"
    fi
    
    # Stop UI
    if [[ -f ".ui.pid" ]]; then
        UI_PID=$(cat .ui.pid)
        if kill -0 $UI_PID 2>/dev/null; then
            print_info "Stopping UI (PID: $UI_PID)..."
            kill $UI_PID
            sleep 2
            if kill -0 $UI_PID 2>/dev/null; then
                print_warning "UI still running, force killing..."
                kill -9 $UI_PID
            fi
            print_success "UI stopped"
        else
            print_warning "UI PID file exists but process not running"
        fi
        rm -f .ui.pid
    else
        print_info "No UI PID file found"
    fi
    
    # Kill any remaining processes
    pkill -f "python run_api.py" 2>/dev/null || true
    pkill -f "python run_ui.py" 2>/dev/null || true
    
    print_success "All services stopped!"
}

restart_services() {
    print_header
    print_info "Restarting Myr-Ag RAG System..."
    
    stop_services
    sleep 2
    start_services
}

show_status() {
    print_header
    print_info "Service Status:"
    echo
    
    # Check API
    if [[ -f ".api.pid" ]]; then
        API_PID=$(cat .api.pid)
        if kill -0 $API_PID 2>/dev/null; then
            echo -e "🔧 API Backend: ${GREEN}🟢 Running${NC} (PID: $API_PID)"
            echo "   URL: http://localhost:8199"
        else
            echo -e "🔧 API Backend: ${RED}🔴 Stopped${NC}"
        fi
    else
        echo -e "🔧 API Backend: ${RED}🔴 Stopped${NC}"
    fi
    
    # Check UI
    if [[ -f ".ui.pid" ]]; then
        UI_PID=$(cat .ui.pid)
        if kill -0 $UI_PID 2>/dev/null; then
            echo -e "🎨 UI Frontend: ${GREEN}🟢 Running${NC} (PID: $UI_PID)"
            echo "   URL: http://localhost:7860"
        else
            echo -e "🎨 UI Frontend: ${RED}🔴 Stopped${NC}"
        fi
    else
        echo -e "🎨 UI Frontend: ${RED}🔴 Stopped${NC}"
    fi
    
    echo
}

show_logs() {
    print_header
    print_info "Service Logs:"
    echo
    
    if [[ -f "logs/api.log" ]]; then
        echo "📝 API Logs (last 20 lines):"
        echo "----------------------------------------"
        tail -20 logs/api.log
        echo
    else
        print_warning "No API logs found"
    fi
    
    if [[ -f "logs/ui.log" ]]; then
        echo "📝 UI Logs (last 20 lines):"
        echo "----------------------------------------"
        tail -20 logs/ui.log
        echo
    else
        print_warning "No UI logs found"
    fi
}

cleanup() {
    print_header
    print_info "Cleaning up..."
    
    # Remove PID files
    rm -f .api.pid .ui.pid
    
    # Clean logs (keep last 1000 lines)
    if [[ -f "logs/api.log" ]]; then
        tail -1000 logs/api.log > logs/api.log.tmp && mv logs/api.log.tmp logs/api.log
        print_success "API logs cleaned"
    fi
    
    if [[ -f "logs/ui.log" ]]; then
        tail -1000 logs/ui.log > logs/ui.log.tmp && mv logs/ui.log.tmp logs/ui.log
        print_success "UI logs cleaned"
    fi
    
    print_success "Cleanup completed!"
}

show_help() {
    print_header
    echo "Available commands:"
    echo "  start     - Start all services"
    echo "  stop      - Stop all services"
    echo "  restart   - Restart all services"
    echo "  status    - Show service status"
    echo "  logs      - Show service logs"
    echo "  clean     - Clean up temporary files"
    echo "  help      - Show this help message"
    echo
    echo "Examples:"
    echo "  ./myr-ag.sh start"
    echo "  ./myr-ag.sh stop"
    echo "  ./myr-ag.sh status"
    echo
    echo "Alternative: use the Makefile:"
    echo "  make start"
    echo "  make stop"
    echo "  make status"
}

# Main script logic
case "${1:-help}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    clean)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo
        show_help
        exit 1
        ;;
esac
