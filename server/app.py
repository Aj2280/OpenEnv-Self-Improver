# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import os

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

# Import all three environments
try:
    from .math_environment import MathEscalationEnvironment
    from .negotiation_environment import NegotiationEnvironment
    from .coding_environment import CodingEnvironment
except ImportError:
    from server.math_environment import MathEscalationEnvironment
    from server.negotiation_environment import NegotiationEnvironment
    from server.coding_environment import CodingEnvironment

max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))

# Create sub-apps
math_app = create_app(MathEscalationEnvironment, CallToolAction, CallToolObservation, env_name="math_env")
negotiate_app = create_app(NegotiationEnvironment, CallToolAction, CallToolObservation, env_name="neg_env")
coding_app = create_app(CodingEnvironment, CallToolAction, CallToolObservation, env_name="code_env")

app = FastAPI(title="Math Escalation Suite")

# Mount Negotiation and Coding on prefixes
app.mount("/negotiate", negotiate_app)
app.mount("/code", coding_app)

# Forward math_app routes to the root (best compatibility with openenv-core)
for route in math_app.routes:
    if route.path != "/":
        app.routes.append(route)

@app.get("/api/landing")
def landing():
    return {"status": "ok", "environments": ["/", "/negotiate", "/code"]}

# UI logic
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(frontend_path):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_path, "assets")), name="assets")

    @app.get("/")
    async def root_redirect():
        return FileResponse(os.path.join(frontend_path, "index.html"))

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Only serve UI if not a known API route
        if full_path in ["reset", "step", "state", "health", "metadata", "negotiate", "code"]:
             return JSONResponse({"error": "Use correct method/prefix"}, status_code=405)
        
        index_file = os.path.join(frontend_path, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return JSONResponse({"error": "Frontend not built."}, status_code=404)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
