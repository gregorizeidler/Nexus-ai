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
  return (
    <GlassCard delay={delay} className="relative overflow-hidden group">
      <div className={`absolute top-0 right-0 w-32 h-32 bg-gradient-to-br ${bgGradient} blur-3xl opacity-50 -mr-10 -mt-10 transition-opacity group-hover:opacity-70`} />
      
      <div className="relative flex items-start justify-between mb-4">
        <div className={`p-3 rounded-xl bg-white/5 border border-white/5 ${color}`}>
          <Icon className="w-6 h-6" />
        </div>
        {trend && (
          <div className={`flex items-center gap-1 text-xs font-medium px-2 py-1 rounded-full border ${trendUp ? 'bg-green-500/10 border-green-500/20 text-green-400' : 'bg-red-500/10 border-red-500/20 text-red-400'}`}>
            {trendUp ? '↑' : '↓'} {trend}
          </div>
        )}
      </div>

      <div>
        <h3 className="text-slate-400 text-sm font-medium mb-1">{name}</h3>
        <div className="text-2xl font-bold text-white tracking-tight">
          {value}
        </div>
      </div>
    </GlassCard>
  );
};

