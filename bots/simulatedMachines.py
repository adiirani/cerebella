import time
import requests
import numpy as np
from datetime import datetime, timedelta
from threading import Thread

class MachineNode:
    def __init__(self, machine_type='CNC', machine_id='001', server_url='http://your-server-url.com'):
        # Initialize machine parameters
        self.machine_type = machine_type
        self.machine_id = machine_id
        self.server_url = server_url
        
        # Time tracking and simulation setup
        self.time_step = 0
        self.base_time = datetime(2023, 1, 1, 0, 0)
        
        # Seed for reproducibility
        np.random.seed(int(self.machine_id[-2:]))  # Seed based on ID for repeatability
        
        # Time since last maintenance (in hours)
        self.time_since_last_maintenance = 0.0
        
        # Maintenance threshold (e.g., perform maintenance every 500 hours)
        self.maintenance_threshold = 500  # In hours

        print(f"Machine {self.machine_type} with ID {self.machine_id} started.")

    def simulate_parameters(self, operating_hours):
        """Simulate machine parameters based on operating hours."""
        if self.machine_type == 'CNC':
            temp = np.random.normal(65 + 0.005 * operating_hours, 2)
            vibration = np.random.normal(0.4 + 0.0003 * operating_hours, 0.05)
            feature = np.random.normal(55 + 0.02 * operating_hours, 3)
        elif self.machine_type == 'Machining':
            temp = np.random.normal(70 + 0.004 * operating_hours, 2.5)
            vibration = np.random.normal(0.5 + 0.0005 * operating_hours, 0.06)
            feature = np.random.normal(60 + 0.015 * operating_hours, 3)
        elif self.machine_type == 'PickAndPlace':
            temp = np.random.normal(40 + 0.003 * operating_hours, 1.5)
            vibration = np.random.normal(0.2 + 0.0002 * operating_hours, 0.04)
            feature = np.random.normal(1000 + 5 * self.time_step, 20)
        else:
            temp, vibration, feature = 0.0, 0.0, 0.0

        return round(temp, 2), round(vibration, 3), round(feature, 2)

    def perform_maintenance(self):
        """Simulate maintenance event."""
        self.time_since_last_maintenance = 0.0  # Reset maintenance timer
        print(f"{self.machine_id} Maintenance performed at {self.base_time + timedelta(minutes=self.time_step * 12)}")

    def create_message(self):
        """Generate the message data to send to the server."""
        timestamp = self.base_time + timedelta(minutes=self.time_step * 12)
        operating_hours = round(self.time_step * 0.2, 2)

        # Update time since last maintenance
        self.time_since_last_maintenance += operating_hours

        # Check if maintenance is due
        if self.time_since_last_maintenance >= self.maintenance_threshold:
            self.perform_maintenance()

        temp, vibration, feature = self.simulate_parameters(operating_hours)

        message = {
            'timestamp': str(timestamp),
            'machine_id': self.machine_id,
            'machine_type': self.machine_type,
            'temperature_C': temp,
            'vibration_g': vibration,
            'feature_1': feature,
            'operating_hours': operating_hours,
            'time_since_last_maintenance': round(self.time_since_last_maintenance, 2)  # Time in hours
        }

        return message

    def send_data(self):
        """Send the generated data to the server every 12 minutes."""
        while True:
            # Create the data message
            message = self.create_message()

            # POST the data to the server
            try:
                response = requests.post(self.server_url, json=message)
                if response.status_code == 200:
                    print(f"{self.machine_id} Data sent to server: {message['timestamp']}")
                else:
                    print(f"{self.machine_id} Failed to send data: {response.status_code}")
            except Exception as e:
                print(f"Error sending data: {e}")

            # Wait for 12 minutes before sending the next data
            time.sleep(1)  # Sleep for 12 minutes
            self.time_step += 1

# Function to instantiate and start multiple machine nodes
def start_multiple_machine_nodes(num_nodes=100):
    threads = []
    for i in range(num_nodes):
        machine_id = f"{i + 1:03}"  # Generate machine IDs like "001", "002", ..., "100"
        machine_node = MachineNode(machine_type='CNC', machine_id=machine_id, server_url='http://localhost:8000/receive_data')
        
        # Start each machine node in its own thread
        thread = Thread(target=machine_node.send_data)
        threads.append(thread)
        thread.start()
    
    # Join all threads to ensure the main thread waits for all to finish
    for thread in threads:
        thread.join()

# Example usage:
if __name__ == "__main__":
    start_multiple_machine_nodes(num_nodes=20)
