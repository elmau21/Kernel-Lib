#!/usr/bin/env python
"""
Script para iniciar la API REST de Kernel ML Engine.
"""

import uvicorn
import sys
import os

# Añade el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload en desarrollo
        log_level="info"
    )

