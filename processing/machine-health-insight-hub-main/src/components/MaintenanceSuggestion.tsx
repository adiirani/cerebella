import React from "react";

interface MaintenanceSuggestionProps {
  suggestion: string | null;
  isLoading: boolean;
}

const MaintenanceSuggestion: React.FC<MaintenanceSuggestionProps> = ({ suggestion, isLoading }) => {
  if (isLoading) {
    return <div className="text-center">Loading maintenance suggestions...</div>;
  }

  if (!suggestion) {
    return <div className="text-center text-gray-500">No maintenance suggestion available.</div>;
  }

  return (
    <div className="p-4 bg-white rounded shadow-sm">
      <h3 className="font-semibold text-gray-900">Maintenance Suggestion</h3>
      <p>{suggestion}</p>
    </div>
  );
};

export default MaintenanceSuggestion;
