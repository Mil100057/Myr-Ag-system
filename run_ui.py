#!/usr/bin/env python3
"""
Launcher script for Myr-Ag RAG System Gradio frontend.
"""
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.ui.gradio_app import frontend
from config.settings import settings


def main():
    """Launch the Gradio frontend interface."""
    parser = argparse.ArgumentParser(description='Launch Myr-Ag RAG System Gradio Frontend')
    parser.add_argument('--share', action='store_true', help='Enable public sharing (creates shareable link)')
    parser.add_argument('--port', type=int, default=settings.UI_PORT, help=f'Port to use (default: {settings.UI_PORT})')
    args = parser.parse_args()
    
    print("🎨 Starting Myr-Ag RAG System Gradio Frontend")
    print("=" * 50)
    print(f"Host: {settings.UI_HOST}")
    print(f"Port: {args.port}")
    print(f"Share: {args.share}")
    print(f"API URL: http://{settings.API_HOST}:{settings.API_PORT}")
    print("=" * 50)
    
    try:
        frontend.launch(
            server_name=settings.UI_HOST,
            server_port=args.port,
            share=args.share
        )
    except Exception as e:
        if "localhost is not accessible" in str(e) or "Address already in use" in str(e) or "Port already in use" in str(e) or "Cannot find empty port" in str(e):
            print(f"⚠️  Port {args.port} not accessible. Trying alternative port without sharing...")
            try:
                # Try a different port range
                import random
                alternative_port = random.randint(7861, 7890)
                frontend.launch(
                    server_name="0.0.0.0",
                    server_port=alternative_port,
                    share=False
                )
            except Exception as share_error:
                print(f"❌ Error starting frontend on port {alternative_port}: {share_error}")
                print("💡 Trying with automatic port selection...")
                try:
                    # Let Gradio choose any available port
                    frontend.launch(
                        server_name="0.0.0.0",
                        server_port=0,  # Let Gradio choose
                        share=False
                    )
                except Exception as auto_error:
                    print(f"❌ Error starting frontend with automatic port: {auto_error}")
                    print("💡 Try running: python run_ui.py --share")
                    sys.exit(1)
        else:
            print(f"❌ Error starting frontend: {e}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Frontend stopped by user")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
