import os
from fastapi import FastAPI, Request, requests
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

class TrafficDataMetadata(BaseModel):
    version: str = "1.0"
    type: Literal["data"]
    timestamp: str
    traffic_light_id: str
    controlled_edges: list[str]
    metrics: dict
    vehicle_stats: dict

CONTROL_URL = os.getenv("CONTROL_URL")

app = FastAPI()

@app.post("/upload")
async def receive_data(data: TrafficDataMetadata, request: Request):
    client_ip = request.client.host
    print(f"[API] Payload from {client_ip}:\n{data.model_dump_json(indent=2)}")

    # Forward to traffic-control
    response = requests.post(f"{CONTROL_URL}/process", json=data.model_dump())
    response.raise_for_status()

    return {"status": "ok", "control_response": response.json()}