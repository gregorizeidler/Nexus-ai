"""
Script to run the FastAPI server.
"""
import uvicorn
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from loguru import logger

def main():
    """Run the API server"""
    logger.info("Starting AML-FORENSIC AI API Server...")
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()

