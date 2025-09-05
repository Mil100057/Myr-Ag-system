#!/usr/bin/env python3
"""
Launcher script for Myr-Ag RAG System FastAPI backend.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.api.fastapi_app import api_backend
from config.settings import settings


def main():
    """Launch the FastAPI backend server."""
    print("🚀 Starting Myr-Ag RAG System FastAPI Backend")
    print("=" * 50)
    print(f"Host: {settings.API_HOST}")
    print(f"Port: {settings.API_PORT}")
    print(f"Data Directory: {settings.DATA_DIR}")
    print("=" * 50)
    
    try:
        api_backend.run(
            host=settings.API_HOST,
            port=settings.API_PORT
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
