import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)

# Machines to simulate
machines = ['CNC', 'Machining', 'PickAndPlace']
train_samples_per_machine = 900
test_samples_per_machine = 100

def generate_machine_data(machine_type, machine_id, rul_values):
    max_rul = {
        'CNC': 5000,
        'Machining': 4000,
        'PickAndPlace': 3000
    }

    base_time = datetime(2023, 1, 1, 0, 0)
    records = []
    
    # Time tracking for maintenance
    time_since_last_maintenance = 0.0
    maintenance_threshold = 500  # Maintenance every 500 hours

    for i, rul in enumerate(rul_values):
        # Environmental conditions (influencing degradation)
        health_factor = rul / max_rul[machine_type]
        # Temperature and vibration impact the rate of degradation
        temperature_factor = random.uniform(0.8, 1.2)  # Random between 80% and 120% of the base temperature
        vibration_factor = random.uniform(0.8, 1.2)    # Random between 80% and 120% of the base vibration

        # Calculate time since last maintenance
        time_since_last_maintenance += 0.2  # Assume each step represents 0.2 hours

        # Check if maintenance is needed
        if time_since_last_maintenance >= maintenance_threshold:
            time_since_last_maintenance = 0.0  # Reset after maintenance

        # Operating hours are inversely proportional to RUL, scaled by environmental factors
        operating_hours = (max_rul[machine_type] * (1 - health_factor)) * (temperature_factor + vibration_factor) / 2
        operating_hours = max(0, operating_hours)

        # Calculate remaining useful life based on operating hours and environmental factors
        adjusted_rul = max(0, max_rul[machine_type] * (1 - operating_hours / max_rul[machine_type]) * random.uniform(0.9, 1.1))

        # Timestamp
        timestamp = base_time + timedelta(minutes=i * 12)

        # Sensor data with degradation effect
        if machine_type == 'CNC':
            temp = np.random.normal(65 + 15 * (1 - health_factor), 5)
            vibration = np.random.normal(0.4 + 0.6 * (1 - health_factor), 0.15)
            feature = np.random.normal(55 + 30 * (1 - health_factor), 8)

        elif machine_type == 'Machining':
            temp = np.random.normal(70 + 20 * (1 - health_factor), 6)
            vibration = np.random.normal(0.5 + 0.8 * (1 - health_factor), 0.18)
            feature = np.random.normal(60 + 40 * (1 - health_factor), 10)

        elif machine_type == 'PickAndPlace':
            temp = np.random.normal(40 + 10 * (1 - health_factor), 4)
            vibration = np.random.normal(0.2 + 0.5 * (1 - health_factor), 0.12)
            feature = np.random.normal(1000 + 500 * (1 - health_factor), 60)

        # Failure label (based on adjusted RUL)
        failure = 1 if adjusted_rul < random.randint(50, 200) else 0

        record = {
            "timestamp": timestamp,
            "machine_id": f"{machine_type}_{machine_id}",
            "machine_type": machine_type,
            "temperature_C": round(temp, 2),
            "vibration_g": round(vibration, 3),
            "feature_1": round(feature, 2),
            "time_since_last_maintenance": round(time_since_last_maintenance, 2),  # Time since last maintenance (in hours)
            "RUL": round(adjusted_rul, 2),
            "failure": failure
        }

        records.append(record)

    return records


def create_balanced_rul_dataset():
    all_data = []

    for machine_type in machines:
        max_rul = {
            'CNC': 5000,
            'Machining': 4000,
            'PickAndPlace': 3000
        }[machine_type]

        for machine_id in range(1, 6):
            # Generate skewed training RULs using a Gamma distribution
            shape_param = 2.0
            rul_train = np.random.gamma(shape=shape_param, scale=max_rul / shape_param, size=train_samples_per_machine)
            rul_train = np.clip(rul_train, 0, max_rul)

            # Generate evenly spaced RULs for test
            rul_test = np.linspace(0, max_rul, test_samples_per_machine)

            # Combine both
            train_records = generate_machine_data(machine_type, machine_id, rul_train)
            test_records = generate_machine_data(machine_type, machine_id, rul_test)

            all_data.extend(train_records + test_records)

    return all_data

# Generate dataset
records = create_balanced_rul_dataset()
df = pd.DataFrame(records)

# Human-readable feature names
df['feature_1_name'] = df['machine_type'].map({
    'CNC': 'spindle_load_%',
    'Machining': 'torque_Nm',
    'PickAndPlace': 'arm_cycles'
})

# Save to CSV
csv_path = "industrial_aerospace_pdm_dataset_with_maintenance.csv"
df.to_csv(csv_path, index=False)
print(f"✅ Dataset generated and saved to {csv_path}")
