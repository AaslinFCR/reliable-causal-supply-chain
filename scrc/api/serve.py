"""Single-process service for persistent SQLite deployments."""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "scrc.api.app:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        workers=1,
        proxy_headers=False,
    )
