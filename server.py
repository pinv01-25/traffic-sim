from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Literal

class TrafficDataMetadata(BaseModel):
    version: str = "1.0"
    type: Literal["data"]
    timestamp: str
    traffic_light_id: str
    controlled_edges: list[str]
    metrics: dict
    vehicle_stats: dict

app = FastAPI()

@app.post("/upload")
async def receive_data(data: TrafficDataMetadata, request: Request):
    client_ip = request.client.host
    print(f"[API] Recibido payload desde {client_ip}:\n{data.model_dump_json(indent=2)}\n")
    return {"status": "ok", "received_at": data.timestamp}
