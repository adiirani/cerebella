
import { MachineData } from "@/types/machine";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Thermometer, Activity, Clock } from "lucide-react";
import { cn } from "@/lib/utils";

interface MachineCardProps {
  data: MachineData[];
  onClick: () => void;
  isSelected: boolean;
}

export default function MachineCard({ data, onClick, isSelected }: MachineCardProps) {
  // Get the latest data point
  const latestData = data[data.length - 1];
  
  // Determine health status based on temperature and vibration
  const getHealthStatus = (tempC: number, vibration: number, feature: number) => {
    if (tempC > 75 || vibration > 0.9 || feature < 60) return "critical";
    if (tempC > 65 || vibration > 0.7 || feature < 70) return "warning";
    return "normal";
  };
  
  const healthStatus = getHealthStatus(
    latestData.temperature_C, 
    latestData.vibration_g,
    latestData.feature_1
  );
  
  const healthColors = {
    normal: "bg-green-500",
    warning: "bg-yellow-500",
    critical: "bg-red-500"
  };
  
  const statusText = {
    normal: "Normal",
    warning: "Warning",
    critical: "Critical"
  };

  return (
    <Card 
      className={cn(
        "cursor-pointer transition-all hover:shadow-md hover:border-dashboard-blue",
        isSelected && "border-dashboard-blue shadow-md"
      )}
      onClick={onClick}
    >
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <CardTitle className="text-lg">{latestData.machine_id}</CardTitle>
          <div className="flex items-center">
            <div className={cn("h-3 w-3 rounded-full mr-2 animate-pulse-soft", healthColors[healthStatus])}></div>
            <span className="text-sm font-medium text-gray-600">{statusText[healthStatus]}</span>
          </div>
        </div>
        <p className="text-sm text-gray-500 font-medium">{latestData.machine_type}</p>
      </CardHeader>
      <CardContent className="pb-3">
        <div className="grid grid-cols-3 gap-2">
          <div className="flex flex-col items-center">
            <div className="flex items-center text-dashboard-blue mb-1">
              <Thermometer size={16} className="mr-1" />
              <span className="text-xs font-medium">Temp</span>
            </div>
            <span className={cn(
              "text-sm font-bold",
              latestData.temperature_C > 75 ? "text-red-500" :
              latestData.temperature_C > 65 ? "text-yellow-500" : "text-green-600"
            )}>
              {latestData.temperature_C.toFixed(1)}°C
            </span>
          </div>
          
          <div className="flex flex-col items-center">
            <div className="flex items-center text-dashboard-teal mb-1">
              <Activity size={16} className="mr-1" />
              <span className="text-xs font-medium">Vib</span>
            </div>
            <span className={cn(
              "text-sm font-bold",
              latestData.vibration_g > 0.9 ? "text-red-500" :
              latestData.vibration_g > 0.7 ? "text-yellow-500" : "text-green-600" 
            )}>
              {latestData.vibration_g.toFixed(2)} g
            </span>
          </div>
          
          <div className="flex flex-col items-center">
            <div className="flex items-center text-dashboard-indigo mb-1">
              <Clock size={16} className="mr-1" />
              <span className="text-xs font-medium">Hours</span>
            </div>
            <span className="text-sm font-bold">
              {Math.floor(latestData.operating_hours).toLocaleString()}
            </span>
          </div>
        </div>
      </CardContent>
      <CardFooter className="pt-0">
        <div className="w-full text-center">
          <p className="text-xs text-gray-500">
            Last updated: {new Date(latestData.timestamp).toLocaleString(undefined, {
              month: 'short',
              day: 'numeric',
              hour: '2-digit',
              minute: '2-digit'
            })}
          </p>
        </div>
      </CardFooter>
    </Card>
  );
}
