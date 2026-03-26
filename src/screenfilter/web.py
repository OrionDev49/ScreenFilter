from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .cli import cmd_collect, cmd_predict, cmd_train

app = FastAPI(title="ScreenFilter Web UI")

# Setup templates
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Store active subprocesses
active_tasks: Dict[str, Dict[str, Any]] = {}

class CommandRequest(BaseModel):
    command: str
    args: Dict[str, Any]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )


@app.get("/ls")
async def list_dir_contents(path: Optional[str] = None):
    if path is None or path == "":
        p = Path.cwd()
    else:
        p = Path(path)

    if not p.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    if not p.is_dir():
        p = p.parent

    try:
        items = []
        items.append({
            "name": "..",
            "path": str(p.parent),
            "is_dir": True
        })
        
        for entry in sorted(p.iterdir()):
            items.append({
                "name": entry.name,
                "path": str(entry.absolute()),
                "is_dir": entry.is_dir()
            })
        return {
            "current_path": str(p.absolute()),
            "items": items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run")
async def run_command(request: CommandRequest):
    try:
        args_dict = request.args.copy()
        arg_list = [request.command]
        
        for key, value in args_dict.items():
            if value is None:
                continue
            
            # Handle boolean flags
            if isinstance(value, bool):
                if value:
                    arg_list.append(f"--{key.replace('_', '-')}")
                continue
            
            # Handle list arguments (e.g., classes)
            if isinstance(value, list):
                arg_list.append(f"--{key.replace('_', '-')}")
                for item in value:
                    arg_list.append(str(item))
                continue
            
            # Handle other arguments
            arg_list.append(f"--{key.replace('_', '-')}")
            arg_list.append(str(value))

        task_id = str(uuid.uuid4())
        
        # Ensure unbuffered stdout/stderr to stream progressively
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        # Start subprocess asynchronously
        is_frozen = getattr(sys, 'frozen', False)
        if is_frozen:
            full_cmd_args = [sys.executable] + arg_list
        else:
            full_cmd_args = [sys.executable, "-m", "screenfilter"] + arg_list
            
        print(f"DEBUG: Running command: {' '.join(full_cmd_args)}")
        
        process = await asyncio.create_subprocess_exec(
            *full_cmd_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )
        
        active_tasks[task_id] = {
            "process": process,
            "stdout": b"",
            "stderr": b"",
        }

        async def read_stream(stream, key):
            while True:
                line = await stream.read(1024)
                if not line:
                    break
                # Safely update the stdout/stderr byte buffer
                if task_id in active_tasks:
                    active_tasks[task_id][key] += line

        asyncio.create_task(read_stream(process.stdout, "stdout"))
        asyncio.create_task(read_stream(process.stderr, "stderr"))

        return {"success": True, "task_id": task_id}

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = active_tasks[task_id]
    process = task["process"]
    
    return {
        "running": process.returncode is None,
        "stdout": task["stdout"].decode("utf-8", errors="replace"),
        "stderr": task["stderr"].decode("utf-8", errors="replace"),
        "success": process.returncode == 0 if process.returncode is not None else False
    }


@app.post("/stop/{task_id}")
async def stop_command(task_id: str):
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = active_tasks[task_id]
    process = task["process"]
    
    if process.returncode is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
            
    return {"success": True}


def start_server(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)
