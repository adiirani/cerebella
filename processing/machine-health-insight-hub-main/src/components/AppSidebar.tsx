
import React, { useState } from "react";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { useMachine } from "@/context/MachineContext";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Check, RefreshCw, LayoutDashboard, Gauge, Settings, AlertTriangle, Wrench } from "lucide-react";
import { toast } from "sonner";

interface SimulationFormData {
  temperature: number;
  vibration: number;
  feature1: number;
  variation: number;
}

const AppSidebar = () => {
  const { selectedMachineId, refreshData, simulateMachineData } = useMachine();
  const [simFormData, setSimFormData] = useState<SimulationFormData>({
    temperature: 70,
    vibration: 0.5,
    feature1: 60,
    variation: 0.1
  });
  const [isSimulating, setIsSimulating] = useState(false);

  const handleSimulationChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setSimFormData(prev => ({
      ...prev,
      [name]: parseFloat(value)
    }));
  };

  const handleSimulate = async () => {
    if (!selectedMachineId) {
      toast.error("Please select a machine first");
      return;
    }

    setIsSimulating(true);
    try {
      await simulateMachineData(
        selectedMachineId,
        simFormData.temperature,
        simFormData.vibration,
        simFormData.feature1,
        simFormData.variation
      );
      toast.success("Simulation completed successfully");
    } catch (error) {
      toast.error("Simulation failed: " + (error instanceof Error ? error.message : "Unknown error"));
    } finally {
      setIsSimulating(false);
    }
  };
  
  return (
    <Sidebar className="border-r">
      <SidebarHeader>
        <div className="px-4 py-2">
          <h2 className="text-lg font-semibold">Machine Monitor</h2>
          <p className="text-sm text-gray-500">Machine Health Dashboard</p>
        </div>
      </SidebarHeader>
      
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton tooltip="Dashboard">
                  <LayoutDashboard className="h-4 w-4 mr-2" />
                  <span>Dashboard</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton tooltip="Metrics">
                  <Gauge className="h-4 w-4 mr-2" />
                  <span>Machine Metrics</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton tooltip="Alerts">
                  <AlertTriangle className="h-4 w-4 mr-2" />
                  <span>Alerts</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton tooltip="Maintenance">
                  <Wrench className="h-4 w-4 mr-2" />
                  <span>Maintenance</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton tooltip="Settings">
                  <Settings className="h-4 w-4 mr-2" />
                  <span>Settings</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        {selectedMachineId && (
          <SidebarGroup>
            <SidebarGroupLabel>Machine Simulation</SidebarGroupLabel>
            <SidebarGroupContent>
              <div className="space-y-3 px-2">
                <div className="space-y-1">
                  <label className="text-xs">Temperature (°C)</label>
                  <Input 
                    type="number"
                    name="temperature"
                    value={simFormData.temperature}
                    onChange={handleSimulationChange}
                    className="h-7"
                  />
                </div>
                
                <div className="space-y-1">
                  <label className="text-xs">Vibration (g)</label>
                  <Input 
                    type="number"
                    step="0.1"
                    name="vibration"
                    value={simFormData.vibration}
                    onChange={handleSimulationChange}
                    className="h-7"
                  />
                </div>
                
                <div className="space-y-1">
                  <label className="text-xs">Feature 1</label>
                  <Input 
                    type="number"
                    name="feature1"
                    value={simFormData.feature1}
                    onChange={handleSimulationChange}
                    className="h-7"
                  />
                </div>
                
                <div className="space-y-1">
                  <label className="text-xs">Variation</label>
                  <Input 
                    type="number"
                    step="0.05"
                    name="variation"
                    value={simFormData.variation}
                    onChange={handleSimulationChange}
                    className="h-7"
                  />
                </div>
                
                <Button 
                  onClick={handleSimulate} 
                  className="w-full" 
                  size="sm"
                  disabled={isSimulating || !selectedMachineId}
                >
                  {isSimulating ? (
                    <>
                      <RefreshCw className="h-3 w-3 mr-2 animate-spin" />
                      Simulating...
                    </>
                  ) : (
                    <>
                      <Check className="h-3 w-3 mr-2" />
                      Run Simulation
                    </>
                  )}
                </Button>
              </div>
            </SidebarGroupContent>
          </SidebarGroup>
        )}
      </SidebarContent>
      
      <SidebarFooter>
        <div className="p-4">
          <Button 
            onClick={refreshData} 
            variant="outline" 
            size="sm" 
            className="w-full"
          >
            <RefreshCw className="h-3 w-3 mr-2" />
            Refresh All Data
          </Button>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
};

export default AppSidebar;
