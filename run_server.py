"""Launch the Harvestify Flask app from the repository root.

This script ensures the repo root is on `sys.path`, imports the Flask
`app` object from `app.app` and runs the server. Use this to start the
application from the project root (recommended) instead of running
`app/app.py` directly.

Usage:
    python run_server.py
    # or set PORT and DEBUG env vars
    PORT=8000 DEBUG=1 python run_server.py
"""

import os
import sys

# Ensure repo root is on sys.path so `import app` works
ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.app import app

def main():
    port = int(os.environ.get("PORT", 5000))
    debug = bool(int(os.environ.get("DEBUG", 0)))
    host = os.environ.get("HOST", "127.0.0.1")

    print(f"Starting Harvestify app on http://{host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    main()
import os
import sys

# Ensure repository root is on path so `import app.app` resolves package imports
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.app import app

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
