from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional

from nexus import FitymiNexus, NexusContext

app = FastAPI(title="Fitymi Nexus API", version="2026.5.0", description="Multi-Agent AEO Copywriting Architecture")

class CopyRequest(BaseModel):
    brand: str
    target_audience: str
    product: str
    goal: str
    task_type: str
    constraints: Optional[Dict[str, Any]] = {"max_words": 150, "tone": "assertive, no hype"}

nexus_engine = FitymiNexus()

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    with open("templates/index.html", "r") as f:
        html_content = f.read()
    return html_content

@app.post("/api/v1/generate")
async def generate_copy(request: CopyRequest):
    ctx = NexusContext(
        brand=request.brand,
        target_audience=request.target_audience,
        product=request.product,
        goal=request.goal,
        task_type=request.task_type,
        constraints=request.constraints or {}
    )
    
    # Execute the MoA Direct Acyclic Graph
    result = await nexus_engine.execute_workflow(ctx)
    
    return {
        "status": "success",
        "data": result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
