import axios from "axios";
import { MachineData, MaintenanceSuggestion } from "@/types/machine";

const BASE_URL = "http://localhost:8000"; // Update if deployed

// Fetch all machines' data
export async function fetchAllMachinesData(): Promise<Record<string, MachineData[]>> {
  const response = await axios.get(`${BASE_URL}/get_all_data/`);
  return response.data.data;
}

// Fetch data for a specific machine
export async function fetchMachineData(machineId: string): Promise<MachineData[]> {
  const response = await axios.get(`${BASE_URL}/get_aggregated_data/${machineId}`);
  if (response.data.error) {
    throw new Error(response.data.error);
  }
  return response.data.data;
}

// Fetch maintenance suggestion for a machine
export async function fetchMaintenanceSuggestion(machineId: string): Promise<MaintenanceSuggestion> {
  const response = await axios.get(`${BASE_URL}/suggest_maintenance/`, {
    params: { machine_id: machineId },
  });

  return {
    status: response.data.status,
    machine_id: response.data.machine_id,
    message: response.data.message,
  };
}

// Optional: Simulate data for a machine (for dev/testing)
export async function simulateMachineData(
  machineId: string,
  base_temperature_C = 70.0,
  base_vibration_g = 0.5,
  base_feature_1 = 60.0,
  variation = 0.1
): Promise<{ simulated_data: MachineData[]; message: string }> {
  const response = await axios.post(`${BASE_URL}/simulate_machine_data/${machineId}`, null, {
    params: {
      base_temperature_C,
      base_vibration_g,
      base_feature_1,
      variation,
    },
  });
  return {
    simulated_data: response.data.simulated_data,
    message: response.data.message,
  };
}
