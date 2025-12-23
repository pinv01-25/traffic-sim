import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Traffic Simulation Service",
    description="Service for running traffic simulations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs
CONTROL_URL = os.getenv("CONTROL_URL", "http://traffic-control:8003")
STORAGE_URL = os.getenv("STORAGE_URL", "http://traffic-storage:8000")
SYNC_URL = os.getenv("SYNC_URL", "http://traffic-sync:8002")

class SimulationStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

class SimulationManager:
    """Manages simulation instances and their lifecycle"""
    
    def __init__(self):
        self.simulations: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def create_simulation(self, simulation_id: str, config: dict) -> bool:
        """Create a new simulation instance"""
        try:
            with self.lock:
                if simulation_id in self.simulations:
                    logger.warning(f"Simulation {simulation_id} already exists")
                    return False
                
                # Create simulation entry
                self.simulations[simulation_id] = {
                    "id": simulation_id,
                    "config": config,
                    "status": SimulationStatus.IDLE,
                    "created_at": datetime.now(timezone.utc),
                    "started_at": None,
                    "stopped_at": None,
                    "error": None,
                    "stats": {
                        "current_time": 0.0,
                        "vehicle_count": 0,
                        "bottleneck_detections": 0,
                        "detection_history": []
                    },
                    "simulation_dir": None,
                    "orchestrator": None
                }
                
                logger.info(f"Created simulation {simulation_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating simulation {simulation_id}: {e}")
            return False
    
    def start_simulation(self, simulation_id: str) -> bool:
        """Start a simulation"""
        try:
            with self.lock:
                if simulation_id not in self.simulations:
                    logger.error(f"Simulation {simulation_id} not found")
                    return False
                
                sim = self.simulations[simulation_id]
                
                if sim["status"] == SimulationStatus.RUNNING:
                    logger.warning(f"Simulation {simulation_id} is already running")
                    return True
                
                # Start simulation in background thread
                sim["status"] = SimulationStatus.RUNNING
                sim["started_at"] = datetime.now(timezone.utc)
                sim["error"] = None
                
                # Start simulation thread
                thread = threading.Thread(
                    target=self._run_simulation,
                    args=(simulation_id,),
                    daemon=True
                )
                thread.start()
                
                logger.info(f"Started simulation {simulation_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error starting simulation {simulation_id}: {e}")
            self._set_simulation_error(simulation_id, str(e))
            return False
    
    def stop_simulation(self, simulation_id: str) -> bool:
        """Stop a simulation"""
        try:
            with self.lock:
                if simulation_id not in self.simulations:
                    logger.error(f"Simulation {simulation_id} not found")
                    return False
                
                sim = self.simulations[simulation_id]
                
                if sim["status"] == SimulationStatus.STOPPED:
                    logger.warning(f"Simulation {simulation_id} is already stopped")
                    return True
                
                # Stop simulation
                sim["status"] = SimulationStatus.STOPPED
                sim["stopped_at"] = datetime.now(timezone.utc)
                
                # Cleanup orchestrator if exists
                if sim["orchestrator"]:
                    try:
                        sim["orchestrator"]._cleanup()
                    except Exception as e:
                        logger.warning(f"Error cleaning up orchestrator: {e}")
                    sim["orchestrator"] = None
                
                # Cleanup simulation directory
                if sim["simulation_dir"] and os.path.exists(sim["simulation_dir"]):
                    try:
                        shutil.rmtree(sim["simulation_dir"])
                    except Exception as e:
                        logger.warning(f"Error cleaning up simulation directory: {e}")
                    sim["simulation_dir"] = None
                
                logger.info(f"Stopped simulation {simulation_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error stopping simulation {simulation_id}: {e}")
            return False
    
    def get_simulation_status(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """Get simulation status"""
        try:
            with self.lock:
                if simulation_id not in self.simulations:
                    return None
                
                sim = self.simulations[simulation_id]
                
                return {
                    "id": sim["id"],
                    "status": sim["status"].value,
                    "created_at": sim["created_at"].isoformat(),
                    "started_at": sim["started_at"].isoformat() if sim["started_at"] else None,
                    "stopped_at": sim["stopped_at"].isoformat() if sim["stopped_at"] else None,
                    "error": sim["error"],
                    "stats": sim["stats"]
                }
                
        except Exception as e:
            logger.error(f"Error getting simulation status {simulation_id}: {e}")
            return None
    
    def get_all_simulations(self) -> Dict[str, Dict[str, Any]]:
        """Get all simulations"""
        try:
            with self.lock:
                return {
                    sim_id: self.get_simulation_status(sim_id)
                    for sim_id in self.simulations.keys()
                }
        except Exception as e:
            logger.error(f"Error getting all simulations: {e}")
            return {}
    
    def _run_simulation(self, simulation_id: str):
        """Run simulation in background thread"""
        try:
            sim = self.simulations[simulation_id]
            config = sim["config"]
            
            # Extract simulation files if provided
            simulation_dir = self._setup_simulation_files(config)
            if not simulation_dir:
                raise Exception("Failed to setup simulation files")
            
            sim["simulation_dir"] = simulation_dir
            
            # Import here to avoid circular imports
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from simulation_orchestrator import SimulationOrchestrator
            
            # Create and setup orchestrator
            orchestrator = SimulationOrchestrator(simulation_dir)
            sim["orchestrator"] = orchestrator
            
            if not orchestrator.setup_simulation():
                raise Exception("Failed to setup simulation")
            
            # Run simulation
            orchestrator.run_simulation()
            
            # Update final stats
            final_stats = orchestrator.get_simulation_stats()
            sim["stats"].update(final_stats)
            
            # Mark as stopped
            sim["status"] = SimulationStatus.STOPPED
            sim["stopped_at"] = datetime.now(timezone.utc)
            
            logger.info(f"Simulation {simulation_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error running simulation {simulation_id}: {e}")
            self._set_simulation_error(simulation_id, str(e))
    
    def _setup_simulation_files(self, config: dict) -> Optional[str]:
        """Setup simulation files from config"""
        try:
            # Check if config has simulation files
            if "simulation_files" in config:
                # Extract ZIP file
                simulation_dir = tempfile.mkdtemp(prefix="traffic_sim_")
                
                # Extract files from base64 or file path
                if "zip_data" in config["simulation_files"]:
                    # Handle base64 ZIP data
                    import base64
                    zip_data = base64.b64decode(config["simulation_files"]["zip_data"])
                    zip_path = os.path.join(simulation_dir, "simulation.zip")
                    
                    with open(zip_path, "wb") as f:
                        f.write(zip_data)
                    
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(simulation_dir)
                    
                    os.remove(zip_path)
                    
                elif "zip_path" in config["simulation_files"]:
                    # Handle file path
                    zip_path = config["simulation_files"]["zip_path"]
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(simulation_dir)
                
                return simulation_dir
            
            # Use default simulation files
            default_sim_dir = Path(__file__).parent.parent / "simulation"
            if default_sim_dir.exists():
                return str(default_sim_dir)
            
            return None
            
        except Exception as e:
            logger.error(f"Error setting up simulation files: {e}")
            return None
    
    def _set_simulation_error(self, simulation_id: str, error: str):
        """Set simulation error status"""
        try:
            with self.lock:
                if simulation_id in self.simulations:
                    self.simulations[simulation_id]["status"] = SimulationStatus.ERROR
                    self.simulations[simulation_id]["error"] = error
                    self.simulations[simulation_id]["stopped_at"] = datetime.now(timezone.utc)
        except Exception as e:
            logger.error(f"Error setting simulation error: {e}")

# Global simulation manager
simulation_manager = SimulationManager()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Traffic Simulation Service is running", "service": "sim"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "sim"}

@app.post("/simulate")
async def start_simulation(simulation_config: dict):
    """Start a traffic simulation"""
    try:
        # Generate unique simulation ID
        simulation_id = f"sim_{int(time.time())}_{os.getpid()}"
        
        # Create simulation
        if not simulation_manager.create_simulation(simulation_id, simulation_config):
            raise HTTPException(status_code=400, detail="Failed to create simulation")
        
        # Store simulation data
        try:
            storage_data = {
                "simulation_id": simulation_id,
                "config": simulation_config,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            requests.post(f"{STORAGE_URL}/store", json=storage_data, timeout=5)
        except Exception as e:
            logger.warning(f"Could not store simulation config: {e}")
        
        # Start simulation
        if not simulation_manager.start_simulation(simulation_id):
            raise HTTPException(status_code=500, detail="Failed to start simulation")
        
        logger.info(f"Starting simulation {simulation_id} with config: {simulation_config}")
        
        return {
            "message": "Simulation started successfully", 
            "simulation_id": simulation_id,
            "config": simulation_config
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/simulation/status")
async def get_simulation_status(simulation_id: Optional[str] = None):
    """Get current simulation status"""
    try:
        if simulation_id:
            # Get specific simulation status
            status = simulation_manager.get_simulation_status(simulation_id)
            if not status:
                raise HTTPException(status_code=404, detail="Simulation not found")
            return status
        else:
            # Get all simulations status
            all_simulations = simulation_manager.get_all_simulations()
            return {
                "message": "Status retrieved successfully",
                "simulations": all_simulations,
                "total_simulations": len(all_simulations)
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting simulation status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/simulation/stop")
async def stop_simulation(simulation_id: str):
    """Stop current simulation"""
    try:
        if not simulation_manager.stop_simulation(simulation_id):
            raise HTTPException(status_code=404, detail="Simulation not found or already stopped")
        
        logger.info(f"Stopping simulation {simulation_id}")
        return {"message": "Simulation stopped successfully", "simulation_id": simulation_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/simulations")
async def list_simulations():
    """List all simulations"""
    try:
        simulations = simulation_manager.get_all_simulations()
        return {
            "simulations": simulations,
            "total": len(simulations)
        }
    except Exception as e:
        logger.error(f"Error listing simulations: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("SERVICE_PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port) 