import React from 'react';

interface ErrorMetricsWidgetProps {
  errors: number;
  kspc: number;
  avgIki: number;
  avgChunkLength: number;
}

const ErrorMetricsWidget: React.FC<ErrorMetricsWidgetProps> = ({ errors, kspc, avgIki, avgChunkLength }) => {
  const MetricItem: React.FC<{ label: string; value: string | number; unit?: string }> = ({ label, value, unit = '' }) => (
    <div className="mb-4">
      <p className="text-gray-400 text-xs font-medium uppercase tracking-wider mb-1">{label}</p>
      <p className="text-2xl font-bold text-gray-100 font-mono">
        {typeof value === 'number' ? value.toFixed(1) : value}
        {unit && <span className="text-sm text-gray-500 ml-1">{unit}</span>}
      </p>
    </div>
  );

  return (
    <div className="w-full bg-gray-900 rounded-xl p-6 border border-gray-800 h-full flex flex-col justify-center">
      <h3 className="text-gray-400 text-xs font-medium uppercase tracking-wider mb-6">Error & Efficiency</h3>
      
      <MetricItem label="Total Errors" value={errors} />
      <MetricItem label="KSPC" value={kspc} unit="presses/char" />
      <MetricItem label="Avg IKI" value={avgIki} unit="ms" />
      <MetricItem label="Avg Chunk Length" value={avgChunkLength} unit="chars" />
    </div>
  );
};

export default ErrorMetricsWidget;
