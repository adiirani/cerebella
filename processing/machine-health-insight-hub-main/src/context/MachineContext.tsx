// context/MachineContext.tsx
import React, { createContext, useContext, useState, useEffect } from "react";
import api from "@/lib/api";

const MachineContext = createContext<any>(null);

export const MachineProvider = ({ children }) => {
  const [allMachinesData, setAllMachinesData] = useState({});
  const [selectedMachineId, setSelectedMachineId] = useState<string | null>(null);
  const [selectedMachineData, setSelectedMachineData] = useState([]);
  const [maintenanceSuggestion, setMaintenanceSuggestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAllData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await api.get("/get_all_data/");
      setAllMachinesData(res.data.data);
    } catch (err) {
      setError("Failed to fetch machine data");
    } finally {
      setIsLoading(false);
    }
  };

  const fetchMachineDetails = async (machineId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await api.get(`/get_aggregated_data/${machineId}`);
      setSelectedMachineData(res.data.data);

      const suggestionRes = await api.get(`/suggest_maintenance?machine_id=${machineId}`);
      setMaintenanceSuggestion(suggestionRes.data.message);
    } catch (err) {
      setError("Failed to fetch machine details or maintenance info");
    } finally {
      setIsLoading(false);
    }
  };

  const selectMachine = (machineId: string) => {
    setSelectedMachineId(machineId);
    fetchMachineDetails(machineId);
  };

  const refreshData = async () => {
    await fetchAllData();
    if (selectedMachineId) {
      await fetchMachineDetails(selectedMachineId);
    }
  };

  const simulateMachineData = async (
    machineId: string,
    temperature: number,
    vibration: number,
    feature1: number,
    variation: number
  ) => {
    setIsLoading(true);
    setError(null);
    try {
      // Simulate the machine data, either by sending it to the backend or updating local state
      const res = await api.post("/simulate_machine_data", {
        machineId,
        temperature,
        vibration,
        feature1,
        variation,
      });

      if (res.data.success) {
        setSelectedMachineData(res.data.simulatedData); // Update with new simulated data
        setMaintenanceSuggestion(res.data.maintenanceSuggestion); // Update with new maintenance suggestion
      }
    } catch (err) {
      setError("Simulation failed");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchAllData();
  }, []);

  return (
    <MachineContext.Provider
      value={{
        allMachinesData,
        selectedMachineId,
        selectedMachineData,
        maintenanceSuggestion,
        isLoading,
        error,
        selectMachine,
        refreshData,
        simulateMachineData, // Provide simulateMachineData function to the context
      }}
    >
      {children}
    </MachineContext.Provider>
  );
};

export const useMachine = () => useContext(MachineContext);
