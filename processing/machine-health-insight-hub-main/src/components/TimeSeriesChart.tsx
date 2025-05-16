
import React, { useState } from "react";
import { MachineData } from "@/types/machine";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, TooltipProps } from "recharts";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface TimeSeriesChartProps {
  data: MachineData[];
}

// Define interface for the custom tooltip props
interface CustomTooltipProps {
  active?: boolean;
  payload?: any[];
  label?: any;
}

export default function TimeSeriesChart({ data }: TimeSeriesChartProps) {
  const [timeRange, setTimeRange] = useState<"all" | "24h" | "7d" | "30d">("all");
  
  if (!data || data.length === 0) {
    return <div>No data available</div>;
  }

  // Format data for chart
  const chartData = data.map(item => ({
    timestamp: new Date(item.timestamp).getTime(),
    temperature: item.temperature_C,
    vibration: item.vibration_g,
    feature: item.feature_1,
    hours: item.operating_hours,
    formattedTime: new Date(item.timestamp).toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }));

  // Filter data based on time range
  const filterDataByTimeRange = () => {
    if (timeRange === "all") return chartData;
    
    const now = new Date().getTime();
    let milliseconds;
    
    switch(timeRange) {
      case "24h":
        milliseconds = 24 * 60 * 60 * 1000;
        break;
      case "7d":
        milliseconds = 7 * 24 * 60 * 60 * 1000;
        break;
      case "30d":
        milliseconds = 30 * 24 * 60 * 60 * 1000;
        break;
      default:
        return chartData;
    }
    
    const cutoff = now - milliseconds;
    return chartData.filter(item => item.timestamp >= cutoff);
  };

  const filteredData = filterDataByTimeRange();

  // Custom tooltip formatter with proper interface
  const CustomTooltip = ({ active, payload, label }: CustomTooltipProps) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-200 shadow-md rounded-md">
          <p className="text-sm font-semibold">{new Date(label).toLocaleString()}</p>
          {payload.map((entry, index) => (
            <p key={index} className="text-sm" style={{ color: entry.color }}>
              {entry.name}: {entry.value}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <Card className="mt-5">
      <CardHeader>
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
          <CardTitle>Machine Performance Metrics</CardTitle>
          <div className="flex items-center">
            <span className="mr-2 text-sm text-gray-500">Time Range:</span>
            <select 
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value as any)}
              className="text-sm border border-gray-300 rounded p-1"
            >
              <option value="all">All Data</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="temperature" className="w-full">
          <TabsList className="grid grid-cols-3 mb-4">
            <TabsTrigger value="temperature">Temperature</TabsTrigger>
            <TabsTrigger value="vibration">Vibration</TabsTrigger>
            <TabsTrigger value="feature">Feature 1</TabsTrigger>
          </TabsList>
          
          <TabsContent value="temperature">
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={filteredData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis 
                    dataKey="timestamp" 
                    scale="time" 
                    type="number"
                    domain={['dataMin', 'dataMax']} 
                    tickFormatter={(timestamp) => {
                      return new Date(timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
                    }}
                  />
                  <YAxis 
                    domain={[
                      dataMin => Math.max(30, Math.floor(dataMin - 5)),
                      dataMax => Math.min(90, Math.ceil(dataMax + 5))
                    ]}
                    label={{ value: 'Temperature (°C)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Line 
                    type="monotone" 
                    dataKey="temperature" 
                    name="Temperature (°C)"
                    stroke="#0EA5E9" 
                    strokeWidth={2} 
                    dot={false} 
                    activeDot={{ r: 6 }} 
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
          
          <TabsContent value="vibration">
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={filteredData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis 
                    dataKey="timestamp" 
                    scale="time" 
                    type="number"
                    domain={['dataMin', 'dataMax']} 
                    tickFormatter={(timestamp) => {
                      return new Date(timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
                    }}
                  />
                  <YAxis 
                    domain={[
                      dataMin => Math.max(0, Math.floor(dataMin * 10) / 10 - 0.1),
                      dataMax => Math.min(2, Math.ceil(dataMax * 10) / 10 + 0.1)
                    ]}
                    label={{ value: 'Vibration (g)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Line 
                    type="monotone" 
                    dataKey="vibration" 
                    name="Vibration (g)"
                    stroke="#14B8A6" 
                    strokeWidth={2} 
                    dot={false} 
                    activeDot={{ r: 6 }} 
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
          
          <TabsContent value="feature">
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={filteredData} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis 
                    dataKey="timestamp" 
                    scale="time" 
                    type="number"
                    domain={['dataMin', 'dataMax']} 
                    tickFormatter={(timestamp) => {
                      return new Date(timestamp).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
                    }}
                  />
                  <YAxis 
                    domain={[
                      dataMin => Math.max(40, Math.floor(dataMin - 5)),
                      dataMax => Math.min(110, Math.ceil(dataMax + 5))
                    ]}
                    label={{ value: 'Feature 1', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Line 
                    type="monotone" 
                    dataKey="feature" 
                    name="Feature 1"
                    stroke="#6366F1" 
                    strokeWidth={2} 
                    dot={false} 
                    activeDot={{ r: 6 }} 
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
