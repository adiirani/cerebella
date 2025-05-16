
import { useContext, createContext } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { fetchAllMachinesData, fetchMachineData, fetchMaintenanceSuggestion, simulateMachineData } from "../services/api";
import { AllMachinesData, AggregatedMachineData, MaintenanceSuggestion } from "../types/machine";

interface MachineContextProps {
  allMachinesData: Record<string, any[]> | null;
  selectedMachineId: string | null;
  selectedMachineData: any[] | null;
  maintenanceSuggestion: MaintenanceSuggestion | null;
  isLoading: boolean;
  error: string | null;
  selectMachine: (machineId: string) => void;
  refreshData: () => Promise<void>;
  simulateMachineData: (machineId: string, temperature?: number, vibration?: number, feature1?: number, variation?: number) => Promise<any>;
}

const MachineContext = createContext<MachineContextProps | undefined>(undefined);

export function useMachine() {
  const context = useContext(MachineContext);
  if (!context) {
    throw new Error("useMachine must be used within a MachineProvider");
  }
  return context;
}

export function MachineProvider({ children }: { children: React.ReactNode }) {
  return children;
}
