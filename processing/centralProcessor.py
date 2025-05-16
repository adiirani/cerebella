from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import json
import numpy as np
import tensorflow as tf
import random
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or your frontend URL like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for aggregated data
aggregated_data: Dict[str, List[Dict]] = {}

# Pydantic model to represent the data format
class MachineData(BaseModel):
    timestamp: str
    machine_id: str
    machine_type: str
    temperature_C: float
    vibration_g: float
    feature_1: float
    operating_hours: float

# Load the pre-trained model (Assuming it's a saved model 'rul_prediction_model.h5')
model = tf.keras.models.load_model('../improved_rul_prediction_model.h5')

# Function to predict RUL (Remaining Useful Life) using the model
def predict_rul(model, X_input):
    return model.predict(X_input)

# Endpoint to receive data from machines
@app.post("/receive_data/")
async def receive_data(data: MachineData):
    """Receive and aggregate data from machine nodes based on machine_id."""
    
    # If the machine_id doesn't exist in the aggregated_data, create a new list
    if data.machine_id not in aggregated_data:
        aggregated_data[data.machine_id] = []

    # Append the received data to the list for this machine_id
    aggregated_data[data.machine_id].append(data.dict())
    
    # Log the aggregation for debugging purposes
    print(f"Received data from machine {data.machine_id}: {data.timestamp}")
    
    return {"status": "success", "machine_id": data.machine_id, "timestamp": data.timestamp}

# Endpoint to get aggregated data for a specific machine
@app.get("/get_aggregated_data/{machine_id}")
async def get_aggregated_data(machine_id: str):
    """Get aggregated data for a specific machine by its ID."""
    if machine_id in aggregated_data:
        return {"machine_id": machine_id, "data": aggregated_data[machine_id]}
    else:
        return {"error": "Machine ID not found."}

# Endpoint to get aggregated data for all machines
@app.get("/get_all_data/")
async def get_all_data():
    """Get aggregated data for all machines."""
    return {"data": aggregated_data}

# Function to suggest predictive maintenance based on operating hours
def suggest_maintenance(machine_id: str):
    """
    Suggest predictive maintenance based on the latest 30 time steps and 7 features.
    """
    if len(aggregated_data.get(machine_id, [])) < 30:
        raise ValueError(f"Not enough data to suggest maintenance for machine {machine_id}. At least 30 data points are required.")
    
    recent_data = aggregated_data[machine_id][-30:]  # Get the last 30 records

    # Extract 7 features per time step
    input_sequence = np.array([[
            entry['temperature_C'],
            entry['vibration_g'],
            entry['feature_1'],
            entry.get('feature_2', 0.0),
            entry.get('feature_3', 0.0),
            entry.get('feature_4', 0.0),
            entry['operating_hours']
        ] for entry in recent_data
    ])

    # Reshape to (1, 30, 7)
    input_sequence = input_sequence.reshape(1, 30, 7)

    # Predict
    predicted_rul = predict_rul(model, input_sequence)
    average_rul = np.mean(predicted_rul)

    if average_rul < 100:
        return f"Maintenance recommended. Predicted average RUL is {average_rul:.2f} hours."
    else:
        return f"Machine is in good condition. Predicted average RUL is {average_rul:.2f} hours."

@app.get("/suggest_maintenance/")
async def suggest_maintenance_endpoint(machine_id: str):
    """Suggest maintenance based on the aggregated data of a machine."""
    try:
        suggestion = suggest_maintenance(machine_id)
        return {"status": "success", "machine_id": machine_id, "message": suggestion}
    except ValueError as e:
        # Return an error response when not enough data is available
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint to simulate 20 data points with user-controlled values
@app.post("/simulate_machine_data/{machine_id}")
async def simulate_machine_data(machine_id: str, base_temperature_C: float = 70.0, base_vibration_g: float = 0.5, base_feature_1: float = 60.0, variation: float = 0.1):
    """Simulate 20 data points for a machine and run predictive maintenance with user-controlled values."""
    try:
        # Generate 20 synthetic data points for the machine
        simulated_data = []
        for i in range(20):
            # Simulate machine data (temperature, vibration, and operating hours)
            operating_hours = i * 0.2  # Each step represents 0.2 operating hours (12 minutes)
            
            # Apply variations to the base values within the specified range
            temperature = random.uniform(base_temperature_C * (1 - variation), base_temperature_C * (1 + variation))
            vibration = random.uniform(base_vibration_g * (1 - variation), base_vibration_g * (1 + variation))
            feature_1 = random.uniform(base_feature_1 * (1 - variation), base_feature_1 * (1 + variation))

            # Create the simulated data
            timestamp = (datetime(2023, 1, 1) + timedelta(minutes=i * 12)).strftime("%Y-%m-%d %H:%M:%S")
            simulated_data.append(MachineData(
                timestamp=timestamp,
                machine_id=machine_id,
                machine_type='CNC',  # You can change the machine type if needed
                temperature_C=temperature,
                vibration_g=vibration,
                feature_1=feature_1,
                operating_hours=operating_hours
            ))

            # Add to aggregated data
            if machine_id not in aggregated_data:
                aggregated_data[machine_id] = []
            aggregated_data[machine_id].extend([data.dict() for data in simulated_data])

        # Run predictive maintenance on the simulated data
        suggestion = suggest_maintenance(machine_id)

        # Return the result of the simulation and maintenance suggestion
        return {"status": "success", "machine_id": machine_id, "message": suggestion, "simulated_data": [data.dict() for data in simulated_data]}
    
    except Exception as e:
        # Return an error response if anything goes wrong
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app if this file is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
