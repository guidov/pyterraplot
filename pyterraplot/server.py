"""
Minimal FastAPI server so .tp.serve() works from a Jupyter cell or script,
plus an interactive 3D Globe / 2D Map NetCDF visual viewer dashboard.
"""
from __future__ import annotations

import json
import threading
import webbrowser
from typing import Any
from pathlib import Path


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


def start_viewer(
    dataset_or_path: str | Path | xr.Dataset,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    """
    Start an interactive visual NetCDF viewer server for the given dataset.
    """
    try:
        import uvicorn
        import xarray as xr
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError as e:
        raise ImportError(
            "xarray, fastapi, and uvicorn are required for the viewer."
        ) from e

    # Handle dataset or file path
    if isinstance(dataset_or_path, (str, Path)):
        ds = xr.open_dataset(dataset_or_path)
        filename_title = Path(dataset_or_path).name
    else:
        ds = dataset_or_path
        filename_title = "In-Memory Dataset"

    app = FastAPI(title=f"pyterraplot NetCDF Viewer - {filename_title}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metadata")
    async def get_metadata():
        import numpy as np
        variables_info = {}
        
        for name in ds.data_vars:
            da = ds[name]
            # Ensure the DataArray has latitude and longitude dimensions
            dims_lower = [d.lower() for d in da.dims]
            has_lat = any(d in dims_lower for d in ["lat", "latitude", "y", "rlat"])
            has_lon = any(d in dims_lower for d in ["lon", "longitude", "x", "rlon"])
            if not (has_lat and has_lon):
                continue

            levels = []
            if "level" in da.dims:
                levels = [float(l) for l in da["level"].values]
            elif "vertical" in da.dims:
                levels = [float(l) for l in da["vertical"].values]

            times = []
            if "time" in da.dims:
                t_vals = da["time"].values
                if np.issubdtype(t_vals.dtype, np.datetime64) or np.issubdtype(t_vals.dtype, np.dtype('<M8[ns]')):
                    times = [str(t)[:19].replace('T', ' ') for t in t_vals]
                else:
                    times = [float(t) for t in t_vals]

            variables_info[name] = {
                "name": name,
                "long_name": da.attrs.get("long_name", name),
                "units": da.attrs.get("units", ""),
                "levels": levels,
                "times": times,
            }
            
        return {"filename": filename_title, "variables": variables_info}

    @app.get("/data")
    async def get_data(var: str, time_idx: int = 0, level: float | None = None):
        if var not in ds.data_vars:
            return JSONResponse({"error": f"Variable {var} not found"}, status_code=400)
            
        da = ds[var]
        
        # Subset time dimension if it exists
        if "time" in da.dims:
            da = da.isel(time=time_idx)
            
        # Subset level dimension if it exists
        if level is not None:
            if "level" in da.dims:
                da = da.sel(level=level, method="nearest")
            elif "vertical" in da.dims:
                da = da.sel(vertical=level, method="nearest")
        else:
            if "level" in da.dims:
                da = da.isel(level=0)
            elif "vertical" in da.dims:
                da = da.isel(vertical=0)
                
        # Reduce any other remaining dimensions
        keep = {"lat", "latitude", "y", "rlat", "lon", "longitude", "x", "rlon"}
        extra = [d for d in da.dims if d.lower() not in keep]
        for d in extra:
            da = da.isel({d: 0})

        # Serialize utilizing the registered tp accessor
        payload = da.tp.to_dict()
        return JSONResponse(payload)

    @app.get("/", response_class=HTMLResponse)
    async def get_viewer_page():
        from .accessor import _load_terraplot_bundle, _SHARED_CSS, _IMPORTMAP
        bundle_js = _load_terraplot_bundle(None)
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>pyterraplot NetCDF Viewer</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700&display=swap" rel="stylesheet">
<style>
{_SHARED_CSS}

body {{
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background-color: #0b0f19;
    color: #e2e8f0;
    display: flex;
    width: 100vw;
    height: 100vh;
    overflow: hidden;
}}

#map {{
    flex: 1;
    height: 100%;
    position: relative;
    z-index: 1;
}}

/* Sidebar styling with premium glassmorphism */
#sidebar {{
    width: 320px;
    height: 100%;
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
    display: flex;
    flex-direction: column;
    z-index: 10;
    padding: 1.5rem;
    box-shadow: 10px 0 30px rgba(0, 0, 0, 0.5);
}}

h1 {{
    font-size: 1.25rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
}}

.filename-badge {{
    font-size: 0.7rem;
    background: rgba(99, 102, 241, 0.2);
    color: #818cf8;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    border: 1px solid rgba(99, 102, 241, 0.3);
    margin-bottom: 1.5rem;
    text-overflow: ellipsis;
    white-space: nowrap;
    overflow: hidden;
}}

.control-group {{
    margin-bottom: 1.25rem;
}}

label {{
    display: block;
    font-size: 0.75rem;
    font-weight: 500;
    color: #94a3b8;
    margin-bottom: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

select, input[type="range"] {{
    width: 100%;
}}

select {{
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 8px;
    padding: 0.6rem;
    color: #fff;
    font-family: inherit;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s ease;
}}

select:focus {{
    outline: none;
    border-color: #6366f1;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
}}

/* Slider styling */
input[type="range"] {{
    -webkit-appearance: none;
    background: transparent;
    cursor: pointer;
    margin: 0.5rem 0;
}}

input[type="range"]::-webkit-slider-runnable-track {{
    background: rgba(255, 255, 255, 0.1);
    height: 4px;
    border-radius: 2px;
}}

input[type="range"]::-webkit-slider-thumb {{
    -webkit-appearance: none;
    background: #6366f1;
    height: 14px;
    width: 14px;
    border-radius: 50%;
    margin-top: -5px;
    transition: background 0.15s ease;
}}

input[type="range"]::-webkit-slider-thumb:hover {{
    background: #818cf8;
}}

/* Colorbar and Legend overlay */
#colorbar-container {{
    position: absolute;
    bottom: 2rem;
    right: 2rem;
    background: rgba(15, 23, 42, 0.8);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 0.8rem;
    z-index: 5;
    pointer-events: none;
    display: flex;
    flex-direction: column;
    align-items: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}}

#cbar-label {{
    font-size: 0.75rem;
    color: #fff;
    margin-bottom: 0.4rem;
    font-weight: 500;
}}

#cbar {{
    border-radius: 3px;
}}

#cbar-ticks {{
    width: 220px;
    display: flex;
    justify-content: space-between;
    font-size: 0.65rem;
    color: #cbd5e1;
    margin-top: 0.3rem;
}}

/* Time Controller Overlay */
#time-controller {{
    position: absolute;
    bottom: 2rem;
    left: 340px;
    right: 260px;
    background: rgba(15, 23, 42, 0.8);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    z-index: 5;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}}

#play-btn {{
    background: #6366f1;
    border: none;
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    color: #fff;
    transition: all 0.15s ease;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    flex-shrink: 0;
}}

#play-btn:hover {{
    background: #818cf8;
    transform: scale(1.05);
}}

#play-btn svg {{
    fill: currentColor;
    width: 16px;
    height: 16px;
}}

.time-slider-wrapper {{
    flex: 1;
    display: flex;
    flex-direction: column;
}}

#time-display {{
    font-size: 0.75rem;
    font-weight: 500;
    color: #94a3b8;
    margin-bottom: 0.1rem;
}}

.tooltip-custom {{
    position: absolute;
    background: rgba(15, 23, 42, 0.9);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 6px 10px;
    border-radius: 6px;
    color: #fff;
    font-size: 0.75rem;
    pointer-events: none;
    z-index: 100;
    display: none;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
}}

/* Custom Styling for original Tooltip elements */
#terraplot-tooltip {{
    background: rgba(15, 23, 42, 0.9) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 6px !important;
    color: #fff !important;
    font-size: 0.75rem !important;
    padding: 6px 10px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5) !important;
}}
</style>
</head>
<body>

<div id="sidebar">
    <h1>terraplot</h1>
    <div class="filename-badge" id="filename-badge">Loading...</div>

    <div class="control-group">
        <label for="var-select">Variable</label>
        <select id="var-select"></select>
    </div>

    <div class="control-group" id="level-group" style="display: none;">
        <label for="level-select">Pressure Level</label>
        <select id="level-select"></select>
    </div>

    <div class="control-group">
        <label for="proj-select">Projection</label>
        <select id="proj-select">
            <option value="3d">3D Globe (GeoSphere)</option>
            <option value="equirectangular">Equirectangular (2D)</option>
            <option value="mercator">Mercator (2D)</option>
            <option value="orthographic">Orthographic (2D)</option>
            <option value="naturalEarth">Natural Earth (2D)</option>
        </select>
    </div>

    <div class="control-group">
        <label for="cmap-select">Colormap</label>
        <select id="cmap-select">
            <option value="RdYlBu_r">Temperature (RdYlBu_r)</option>
            <option value="viridis">Standard (viridis)</option>
            <option value="plasma">Plasma (plasma)</option>
            <option value="inferno">Dark Hot (inferno)</option>
            <option value="magma">Magma (magma)</option>
            <option value="RdBu_r">Anomalies (RdBu_r)</option>
            <option value="Spectral_r">Spectral (Spectral_r)</option>
            <option value="YlGnBu">Precipitation (YlGnBu)</option>
            <option value="Blues">Blues (Blues)</option>
            <option value="Greys">Greys (Greys)</option>
        </select>
    </div>

    <div class="control-group">
        <label for="plot-type-select">Plot Type</label>
        <select id="plot-type-select">
            <option value="pcolormesh">pcolormesh (Smooth)</option>
            <option value="contourf">contourf (Banded)</option>
            <option value="contour">contour (Lines)</option>
        </select>
    </div>

    <div class="control-group" id="levels-group" style="display: none;">
        <label for="levels-input">Contour Levels</label>
        <input type="text" id="levels-input" value="12" placeholder="e.g. 12 or -10,-5,0,5,10" style="width: 100%; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.15); color: white; padding: 6px; border-radius: 4px;"/>
    </div>

    <div class="control-group">
        <label for="surface-select">Earth Surface</label>
        <select id="surface-select">
            <option value="satellite">Satellite Imagery</option>
            <option value="shaded_relief">Shaded Relief (Stock)</option>
            <option value="outline">Vector Outline Only</option>
        </select>
    </div>

    <div class="control-group">
        <label for="alpha-slider">Opacity</label>
        <input type="range" id="alpha-slider" min="0" max="1" step="0.05" value="0.75"/>
    </div>
</div>

<div id="map"></div>

<div id="time-controller" style="display: none;">
    <button id="play-btn" title="Play">
        <!-- Play Icon -->
        <svg id="play-icon" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
        <!-- Pause Icon (hidden by default) -->
        <svg id="pause-icon" viewBox="0 0 24 24" style="display:none;"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
    </button>
    <div class="time-slider-wrapper">
        <div id="time-display">Time Step: Loading...</div>
        <input type="range" id="time-slider" min="0" max="0" value="0"/>
    </div>
</div>

<div id="colorbar-container">
    <div id="cbar-label">Legend</div>
    <canvas id="cbar" width="220" height="15"></canvas>
    <div id="cbar-ticks">
        <span id="tick-min">-</span>
        <span id="tick-mid">-</span>
        <span id="tick-max">-</span>
    </div>
</div>

{_IMPORTMAP}
<script type="module">
{bundle_js}

let mapInstance = null;
let datasetMetadata = null;
let currentVar = "";
let currentLevel = null;
let currentTimeIdx = 0;
let isPlaying = false;
let animationTimer = null;

// UI References
const varSelect = document.getElementById("var-select");
const levelSelect = document.getElementById("level-select");
const levelGroup = document.getElementById("level-group");
const projSelect = document.getElementById("proj-select");
const surfaceSelect = document.getElementById("surface-select");
const cmapSelect = document.getElementById("cmap-select");
const plotTypeSelect = document.getElementById("plot-type-select");
const alphaSlider = document.getElementById("alpha-slider");
const timeController = document.getElementById("time-controller");
const timeSlider = document.getElementById("time-slider");
const timeDisplay = document.getElementById("time-display");
const playBtn = document.getElementById("play-btn");
const playIcon = document.getElementById("play-icon");
const pauseIcon = document.getElementById("pause-icon");

// Init application
async function init() {{
    const response = await fetch("/metadata");
    datasetMetadata = await response.json();
    
    document.getElementById("filename-badge").textContent = datasetMetadata.filename;
    
    // Populate variables dropdown
    const variables = datasetMetadata.variables;
    varSelect.innerHTML = "";
    Object.keys(variables).forEach(name => {{
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = variables[name].long_name || name;
        varSelect.appendChild(opt);
    }});
    
    if (Object.keys(variables).length > 0) {{
        currentVar = varSelect.value;
        configureVariableUI();
    }}
    
    // Setup event listeners
    varSelect.addEventListener("change", e => {{
        currentVar = e.target.value;
        configureVariableUI();
    }});
    
    levelSelect.addEventListener("change", e => {{
        currentLevel = parseFloat(e.target.value) || null;
        loadField();
    }});
    
    projSelect.addEventListener("change", e => {{
        initializeMap();
        loadField();
    }});
    
    surfaceSelect.addEventListener("change", () => {{
        initializeMap();
        loadField();
    }});
    
    cmapSelect.addEventListener("change", () => loadField());
    plotTypeSelect.addEventListener("change", () => {{
        const type = plotTypeSelect.value;
        const lg = document.getElementById("levels-group");
        if (type === "contourf" || type === "contour") {{
            lg.style.display = "block";
        }} else {{
            lg.style.display = "none";
        }}
        loadField();
    }});
    
    document.getElementById("levels-input").addEventListener("change", () => loadField());
    alphaSlider.addEventListener("input", () => loadField());
    
    timeSlider.addEventListener("input", e => {{
        currentTimeIdx = parseInt(e.target.value);
        updateTimeDisplay();
        loadField();
    }});
    
    playBtn.addEventListener("click", togglePlay);
    
    // Start map
    initializeMap();
    loadField();
}}

function configureVariableUI() {{
    const variable = datasetMetadata.variables[currentVar];
    if (!variable) return;
    
    // 1. Configure levels
    if (variable.levels && variable.levels.length > 0) {{
        levelGroup.style.display = "block";
        levelSelect.innerHTML = "";
        variable.levels.forEach(lvl => {{
            const opt = document.createElement("option");
            opt.value = lvl;
            opt.textContent = lvl + " hPa";
            levelSelect.appendChild(opt);
        }});
        currentLevel = parseFloat(levelSelect.value);
    }} else {{
        levelGroup.style.display = "none";
        currentLevel = null;
    }}
    
    // 2. Configure time dimensions
    if (variable.times && variable.times.length > 1) {{
        timeController.style.display = "flex";
        timeSlider.max = variable.times.length - 1;
        timeSlider.value = 0;
        currentTimeIdx = 0;
        updateTimeDisplay();
    }} else {{
        stopPlay();
        timeController.style.display = "none";
        currentTimeIdx = 0;
    }}
    
    loadField();
}}

function updateTimeDisplay() {{
    const variable = datasetMetadata.variables[currentVar];
    if (variable && variable.times && variable.times.length > 0) {{
        timeDisplay.textContent = "Time Step: " + variable.times[currentTimeIdx];
    }}
}}

function initializeMap() {{
    const proj = projSelect.value;
    const surface = surfaceSelect.value;
    
    if (mapInstance) {{
        mapInstance.dispose();
        document.getElementById("map").innerHTML = "";
    }}
    
    if (proj === "3d") {{
        mapInstance = new GeoSphere("#map", {{ 
            earthSurface: surface,
            tooltip: true 
        }});
        if (surface === "outline") {{
            mapInstance.addFeature("coastlines");
            mapInstance.addFeature("borders");
        }}
    }} else {{
        mapInstance = new GeoMap("#map", {{
            projection: proj,
            center: [0, 0],
            background: "transparent",
            earthSurface: surface,
            tooltip: true
        }});
        mapInstance.addFeature("coastlines");
        if (surface === "outline") {{
            mapInstance.addFeature("borders");
        }}
    }}
}}

function parseLevelsValue() {{
    const val = document.getElementById("levels-input").value.trim();
    if (!val) return 12;
    if (val.includes(",")) {{
        const parts = val.split(",").map(x => parseFloat(x.trim())).filter(x => !isNaN(x));
        return parts.length > 0 ? parts : 12;
    }}
    const intVal = parseInt(val);
    return isNaN(intVal) ? 12 : intVal;
}}

async function loadField() {{
    if (!currentVar) return;
    
    let url = `/data?var=${{currentVar}}&time_idx=${{currentTimeIdx}}`;
    if (currentLevel !== null) {{
        url += `&level=${{currentLevel}}`;
    }}
    
    const res = await fetch(url);
    const payload = await res.json();
    
    // Update map layer
    const plotType = plotTypeSelect.value;
    const cmap = cmapSelect.value;
    const alpha = parseFloat(alphaSlider.value);
    
    // Calculate field min and max dynamically
    let minVal = Infinity;
    let maxVal = -Infinity;
    const flatField = payload.field.flat(Infinity);
    for (let i = 0; i < flatField.length; i++) {{
        const v = flatField[i];
        if (v != null && isFinite(v)) {{
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
        }}
    }}
    if (minVal === Infinity) {{
        minVal = 0;
        maxVal = 1;
    }}
    
    mapInstance.clearAll();
    if (projSelect.value !== "3d") {{
        mapInstance.addFeature("coastlines");
    }}
    
    const levelsVal = parseLevelsValue();
    
    if (plotType === "pcolormesh") {{
        mapInstance.pcolormesh(payload.lons, payload.lats, payload.field, {{
            cmap: cmap,
            alpha: alpha,
            vmin: minVal,
            vmax: maxVal,
            name: payload.name,
            units: payload.units
        }});
    }} else if (plotType === "contourf") {{
        mapInstance.contourf(payload.lons, payload.lats, payload.field, {{
            cmap: cmap,
            alpha: alpha,
            vmin: minVal,
            vmax: maxVal,
            levels: levelsVal,
            name: payload.name,
            units: payload.units
        }});
    }} else {{
        mapInstance.contour(payload.lons, payload.lats, payload.field, {{
            cmap: cmap,
            alpha: alpha,
            vmin: minVal,
            vmax: maxVal,
            levels: levelsVal,
            name: payload.name,
            units: payload.units
        }});
    }}
    
    updateColorbar(minVal, maxVal, cmap, payload.long_name, payload.units);
}}

function updateColorbar(min, max, cmap, longName, units) {{
    // Update labels
    document.getElementById("cbar-label").textContent = longName + (units ? ` (${{units}})` : "");
    document.getElementById("tick-min").textContent = min.toFixed(2);
    document.getElementById("tick-mid").textContent = ((min + max) / 2).toFixed(2);
    document.getElementById("tick-max").textContent = max.toFixed(2);
    
    // Draw canvas
    const canvas = document.getElementById("cbar");
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    
    const colorFn = resolveColormap(cmap);
    for (let x = 0; x < W; x++) {{
        const [r, g, b] = colorFn(x / (W - 1));
        ctx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
        ctx.fillRect(x, 0, 1, H);
    }}
}}

function togglePlay() {{
    if (isPlaying) {{
        stopPlay();
    }} else {{
        startPlay();
    }}
}}

function startPlay() {{
    isPlaying = true;
    playIcon.style.display = "none";
    pauseIcon.style.display = "block";
    
    const variable = datasetMetadata.variables[currentVar];
    const maxSteps = variable.times.length;
    
    animationTimer = setInterval(() => {{
        currentTimeIdx = (currentTimeIdx + 1) % maxSteps;
        timeSlider.value = currentTimeIdx;
        updateTimeDisplay();
        loadField();
    }}, 1000);
}}

function stopPlay() {{
    isPlaying = false;
    playIcon.style.display = "block";
    pauseIcon.style.display = "none";
    if (animationTimer) {{
        clearInterval(animationTimer);
        animationTimer = null;
    }}
}}

// Initialize on page load
window.addEventListener("load", () => {{
    setTimeout(init, 50);
}});

</script>
</body>
</html>"""
        headers = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
        return HTMLResponse(content=html_content, headers=headers)

    if open_browser:
        url = f"http://{host}:{port}"
        threading.Timer(1.5, webbrowser.open, args=[url]).start()

    uvicorn.run(app, host=host, port=port, log_level="warning")
