"""
Minimal FastAPI server so .tp.serve() works from a Jupyter cell or script.
Optional dependency — only needed for live serving.
"""
from __future__ import annotations

import json
import threading
import webbrowser
from typing import Any


def serve(
    payload: dict[str, Any],
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
) -> None:
    """
    Serve a single field payload over HTTP and WebSocket.

    GET  /field          → JSON payload (for fetch() in JS)
    GET  /health         → {"status": "ok"}
    WS   /ws             → push updated payload when serve() is called again

    Blocks the calling thread. Run in a background thread for Jupyter use:
        t = threading.Thread(target=da.tp.serve, kwargs={"port": 8765}, daemon=True)
        t.start()
    """
    try:
        import uvicorn
        from fastapi import FastAPI, WebSocket
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as e:
        raise ImportError(
            "pyterraplot[serve] required: pip install pyterraplot[serve]"
        ) from e

    app = FastAPI(title="pyterraplot")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    _payload_ref = {"data": payload}
    _ws_clients: list[WebSocket] = []

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/field")
    async def field():
        from fastapi.responses import JSONResponse
        return JSONResponse(_payload_ref["data"])

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        _ws_clients.append(websocket)
        # send current payload immediately on connect
        await websocket.send_text(json.dumps(_payload_ref["data"]))
        try:
            while True:
                await websocket.receive_text()  # keep alive
        except Exception:
            _ws_clients.remove(websocket)

    if open_browser:
        url = f"http://{host}:{port}"
        threading.Timer(1.0, webbrowser.open, args=[url]).start()

    uvicorn.run(app, host=host, port=port, log_level="warning")
