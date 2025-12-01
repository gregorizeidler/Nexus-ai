import React from 'react';
import { motion } from 'framer-motion';
import { GlassCard } from './GlassCard';

interface MetricCardProps {
  name: string;
  value: string | number;
  icon: React.ElementType;
  color: string; // e.g., 'text-blue-400'
  bgGradient: string; // e.g., 'from-blue-500/20 to-blue-600/5'
  trend?: string;
  trendUp?: boolean;
  delay?: number;
}

export const MetricCard = ({ 
  name, 
  value, 
  icon: Icon, 
  color, 
  bgGradient,
  trend,
  trendUp,
  delay = 0 
}: MetricCardProps) => {
  const colorMap: Record<string, string> = {
    'text-blue-400': 'bg-blue-50 text-blue-600',
    'text-yellow-400': 'bg-yellow-50 text-yellow-600',
    'text-red-400': 'bg-red-50 text-red-600',
    'text-green-400': 'bg-green-50 text-green-600',
  };
  
  const iconBg = colorMap[color] || 'bg-slate-50 text-slate-600';

  return (
    <GlassCard delay={delay} className="relative overflow-hidden group">
      <div className="relative flex items-start justify-between mb-4">
        <div className={`p-3 rounded-lg ${iconBg} shadow-sm`}>
          <Icon className="w-6 h-6" />
        </div>
        {trend && (
          <div className={`flex items-center gap-1 text-xs font-medium px-2 py-1 rounded-full ${
            trendUp === true 
              ? 'bg-green-50 text-green-700 border border-green-200' 
              : trendUp === false
              ? 'bg-red-50 text-red-700 border border-red-200'
              : 'bg-slate-50 text-slate-700 border border-slate-200'
          }`}>
            {trendUp === true ? '↑' : trendUp === false ? '↓' : ''} {trend}
          </div>
        )}
      </div>

      <div>
        <h3 className="text-slate-600 text-sm font-medium mb-1">{name}</h3>
        <div className="text-3xl font-bold text-slate-900 tracking-tight">
          {value}
        </div>
      </div>
    </GlassCard>
  );
};

