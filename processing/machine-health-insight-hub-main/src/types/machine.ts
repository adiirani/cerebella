
export interface MachineData {
  timestamp: string;
  machine_id: string;
  machine_type: string;
  temperature_C: number;
  vibration_g: number;
  feature_1: number;
  operating_hours: number;
}

export interface MaintenanceSuggestion {
  status: string;
  machine_id: string;
  message: string;
}

export interface AggregatedMachineData {
  machine_id: string;
  data: MachineData[];
}

export interface AllMachinesData {
  data: Record<string, MachineData[]>;
}
