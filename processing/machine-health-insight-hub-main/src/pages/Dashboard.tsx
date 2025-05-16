import { useState } from "react";
import { useMachine } from "@/context/MachineContext";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import MachineCard from "@/components/MachineCard";
import MachineStats from "@/components/MachineStats";
import TimeSeriesChart from "@/components/TimeSeriesChart";
import MaintenanceSuggestion from "@/components/MaintenanceSuggestion";
import GlassmorphicButton from "@/components/GlassmorphicButton";
import { RefreshCw, AlertCircle, ArrowLeft } from "lucide-react";
import { SidebarProvider } from "@/components/ui/sidebar";
import AppSidebar from "@/components/AppSidebar";

export default function Dashboard() {
  const { 
    allMachinesData, 
    selectedMachineId, 
    selectedMachineData, 
    maintenanceSuggestion,
    isLoading, 
    error, 
    selectMachine, 
    refreshData 
  } = useMachine();
  
  const [activeTab, setActiveTab] = useState<string>("overview");
  
  const handleMachineClick = (machineId: string) => {
    selectMachine(machineId);
    setActiveTab("details");
  };
  
  const handleRefresh = async () => {
    await refreshData();
  };
  
  const handleBackToOverview = () => {
    setActiveTab("overview");
  };

  return (
    <SidebarProvider defaultOpen={true}>
      <div className="flex w-full min-h-screen">
        <AppSidebar />
        
        <div className="flex-1 overflow-auto">
          <div className="container max-w-7xl mx-auto py-6 px-4 sm:px-6">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6">
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Machine Health Insight Hub</h1>
                <p className="text-gray-500 mt-1">Monitor and analyze machine performance metrics</p>
              </div>
              <Button 
                onClick={handleRefresh} 
                variant="outline" 
                size="sm"
                className="mt-2 sm:mt-0"
                disabled={isLoading}
              >
                <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
                Refresh Data
              </Button>
            </div>
            
            {error && (
              <Card className="mb-6 border-red-300 bg-red-50">
                <CardContent className="pt-6">
                  <div className="flex items-center text-red-700">
                    <AlertCircle className="h-5 w-5 mr-2" />
                    <span>{error}</span>
                  </div>
                </CardContent>
              </Card>
            )}
            
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <div className="flex items-center mb-6 gap-4">
                {activeTab === "details" && (
                  <GlassmorphicButton 
                    onClick={handleBackToOverview}
                    icon={<ArrowLeft className="h-4 w-4" />}
                    aria-label="Back to overview"
                  >
                    Back
                  </GlassmorphicButton>
                )}
                <TabsList className="grid w-full max-w-md grid-cols-2">
                  <TabsTrigger value="overview">Machines Overview</TabsTrigger>
                  <TabsTrigger value="details" disabled={!selectedMachineId}>Machine Details</TabsTrigger>
                </TabsList>
              </div>
              
              <TabsContent value="overview" className="mt-0">
                {isLoading && !allMachinesData ? (
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                    {[...Array(6)].map((_, i) => (
                      <Card key={i} className="h-40">
                        <CardContent className="p-6">
                          <div className="animate-pulse flex flex-col gap-2">
                            <div className="flex justify-between">
                              <div className="h-5 bg-gray-200 rounded w-16"></div>
                              <div className="h-4 bg-gray-200 rounded w-20"></div>
                            </div>
                            <div className="h-4 bg-gray-200 rounded w-24 mt-1"></div>
                            <div className="grid grid-cols-3 gap-2 mt-3">
                              <div className="h-8 bg-gray-200 rounded"></div>
                              <div className="h-8 bg-gray-200 rounded"></div>
                              <div className="h-8 bg-gray-200 rounded"></div>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                ) : (
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                    {allMachinesData && Object.entries(allMachinesData).map(([machineId, data]) => (
                      <MachineCard 
                        key={machineId} 
                        data={Array.isArray(data) ? data : []} // Ensure data is always an array
                        onClick={() => handleMachineClick(machineId)}
                        isSelected={machineId === selectedMachineId}
                      />
                    ))}
                  </div>
                )}
              </TabsContent>
              
              <TabsContent value="details" className="mt-0">
                {selectedMachineId && selectedMachineData ? (
                  <div>
                    <div className="mb-6">
                      <h2 className="text-xl font-bold">
                        Machine {selectedMachineId} - {selectedMachineData[0]?.machine_type}
                      </h2>
                      <p className="text-gray-500 text-sm mt-1">
                        Detailed performance metrics and maintenance recommendations
                      </p>
                    </div>
                    
                    <MachineStats data={selectedMachineData[selectedMachineData.length - 1]} />
                    
                    <TimeSeriesChart data={selectedMachineData} />
                    
                    <MaintenanceSuggestion 
                      suggestion={maintenanceSuggestion} 
                      isLoading={isLoading} 
                    />
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <p className="text-gray-500">Select a machine to view details</p>
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>
    </SidebarProvider>
  );
}
