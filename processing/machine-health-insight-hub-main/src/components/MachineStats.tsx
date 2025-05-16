
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MachineData } from "@/types/machine";
import { Thermometer, Activity, Clock, Gauge } from "lucide-react";
import { cn } from "@/lib/utils";

interface MachineStatsProps {
  data: MachineData | null;
}

export default function MachineStats({ data }: MachineStatsProps) {
  if (!data) return null;

  // Function to determine if a value is in warning or critical state
  const getStatusColor = (value: number, thresholds: [number, number], inverse = false) => {
    const [warning, critical] = thresholds;
    if (inverse) {
      if (value < critical) return "text-red-500";
      if (value < warning) return "text-yellow-500";
      return "text-green-600";
    } else {
      if (value > critical) return "text-red-500";
      if (value > warning) return "text-yellow-500";
      return "text-green-600";
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-gray-500 flex items-center">
            <Thermometer className="mr-2 h-4 w-4 text-dashboard-blue" />
            Temperature
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            <span className={getStatusColor(data.temperature_C, [65, 75])}>
              {data.temperature_C.toFixed(1)}°C
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            {data.temperature_C > 75 ? "Critical - Shutdown recommended" :
             data.temperature_C > 65 ? "Warning - Monitor closely" :
             "Normal operating temperature"}
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-gray-500 flex items-center">
            <Activity className="mr-2 h-4 w-4 text-dashboard-teal" />
            Vibration
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            <span className={getStatusColor(data.vibration_g, [0.7, 0.9])}>
              {data.vibration_g.toFixed(2)} g
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            {data.vibration_g > 0.9 ? "Critical - Check for loose components" :
             data.vibration_g > 0.7 ? "Warning - Increased vibration detected" :
             "Normal vibration levels"}
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-gray-500 flex items-center">
            <Gauge className="mr-2 h-4 w-4 text-dashboard-indigo" />
            Feature 1
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            <span className={getStatusColor(data.feature_1, [70, 60], true)}>
              {data.feature_1.toFixed(1)}
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            {data.feature_1 < 60 ? "Critical - Maintenance required" :
             data.feature_1 < 70 ? "Warning - Performance degrading" :
             "Feature functioning normally"}
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-gray-500 flex items-center">
            <Clock className="mr-2 h-4 w-4 text-dashboard-yellow" />
            Operating Hours
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {Math.floor(data.operating_hours).toLocaleString()}
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            Total machine runtime
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
