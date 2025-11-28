import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { systemApi, alertApi } from '../services/api';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
} from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';
import { 
  ExclamationTriangleIcon, 
  DocumentTextIcon, 
  CreditCardIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline';
import { motion } from 'framer-motion';
import { GlassCard } from '../components/ui/GlassCard';
import { MetricCard } from '../components/ui/MetricCard';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

export default function Dashboard() {
  const { data: metrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['system-metrics'],
    queryFn: systemApi.metrics,
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  const { data: alerts } = useQuery({
    queryKey: ['alerts-list'],
    queryFn: () => alertApi.list({ limit: 100 }),
    refetchInterval: 15000,
  });

  if (metricsLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="relative">
          <div className="w-16 h-16 rounded-full border-4 border-blue-500/30 border-t-blue-500 animate-spin"></div>
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
          </div>
        </div>
      </div>
    );
  }

  // Alert risk level distribution
  const riskLevelData = {
    labels: ['Low', 'Medium', 'High', 'Critical'],
    datasets: [
      {
        label: 'Alerts by Risk Level',
        data: [
          metrics?.alerts.by_risk_level?.low || 0,
          metrics?.alerts.by_risk_level?.medium || 0,
          metrics?.alerts.by_risk_level?.high || 0,
          metrics?.alerts.by_risk_level?.critical || 0,
        ],
        backgroundColor: [
          'rgba(34, 197, 94, 0.5)',
          'rgba(234, 179, 8, 0.5)',
          'rgba(249, 115, 22, 0.5)',
          'rgba(239, 68, 68, 0.5)',
        ],
        borderColor: [
          'rgb(34, 197, 94)',
          'rgb(234, 179, 8)',
          'rgb(249, 115, 22)',
          'rgb(239, 68, 68)',
        ],
        borderWidth: 1,
      },
    ],
  };

  // Alert status distribution
  const statusData = {
    labels: Object.keys(metrics?.alerts.by_status || {}).map(s => 
      s.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
    ),
    datasets: [
      {
        data: Object.values(metrics?.alerts.by_status || {}),
        backgroundColor: [
          'rgba(59, 130, 246, 0.6)',
          'rgba(139, 92, 246, 0.6)',
          'rgba(16, 185, 129, 0.6)',
        ],
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
      },
    ],
  };

  const chartOptions: ChartOptions<any> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#94a3b8', font: { family: 'Inter' } },
        position: 'bottom',
      },
    },
    scales: {
      y: {
        ticks: { color: '#64748b' },
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        border: { display: false },
      },
      x: {
        ticks: { color: '#64748b' },
        grid: { display: false },
        border: { display: false },
      },
    },
  };

  return (
    <div className="space-y-8">
      {/* Header Section */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-4xl font-bold text-white tracking-tight mb-2 neon-text-blue">
            Command Center
          </h1>
          <p className="text-slate-400 text-lg">
            Real-time AML forensic analysis & monitoring
          </p>
        </div>
        <div className="flex gap-3">
          <button className="btn-secondary">
            <span>📅 Last 24h</span>
          </button>
          <button className="btn-primary">
            <span>⚡ Live Stream</span>
          </button>
        </div>
      </div>

      {/* Bento Grid - Metrics */}
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          name="Transactions Processed"
          value={(metrics?.transactions.total_processed || 0).toLocaleString()}
          icon={CreditCardIcon}
          color="text-blue-400"
          bgGradient="from-blue-500/20 to-cyan-500/5"
          trend="12.5%"
          trendUp={true}
          delay={0.1}
        />
        <MetricCard
          name="Active Alerts"
          value={metrics?.alerts.pending_count || 0}
          icon={ExclamationTriangleIcon}
          color="text-yellow-400"
          bgGradient="from-yellow-500/20 to-orange-500/5"
          trend="4.2%"
          trendUp={false}
          delay={0.2}
        />
        <MetricCard
          name="SARs Generated"
          value={metrics?.sars.total_sars || 0}
          icon={DocumentTextIcon}
          color="text-red-400"
          bgGradient="from-red-500/20 to-pink-500/5"
          trend="2.1%"
          trendUp={true}
          delay={0.3}
        />
        <MetricCard
          name="AI Agents Active"
          value={metrics?.agents.enabled_agents || 0}
          icon={CheckCircleIcon}
          color="text-green-400"
          bgGradient="from-green-500/20 to-emerald-500/5"
          trend="Stable"
          trendUp={true}
          delay={0.4}
        />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Main Chart - Spans 2 columns */}
        <GlassCard className="lg:col-span-2" delay={0.5}>
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-white">Risk Level Distribution</h3>
            <div className="flex gap-2">
              <span className="w-2 h-2 rounded-full bg-red-500"></span>
              <span className="text-xs text-slate-400">Live Updates</span>
            </div>
          </div>
          <div className="h-80">
            <Bar data={riskLevelData} options={chartOptions} />
          </div>
        </GlassCard>

        {/* Secondary Chart */}
        <GlassCard delay={0.6}>
          <h3 className="text-lg font-semibold text-white mb-6">Alert Status</h3>
          <div className="h-64 flex items-center justify-center">
            <Doughnut 
              data={statusData} 
              options={{
                ...chartOptions,
                scales: undefined,
                cutout: '70%',
              }} 
            />
          </div>
          <div className="mt-6 grid grid-cols-3 gap-2 text-center">
            {Object.entries(metrics?.alerts.by_status || {}).slice(0, 3).map(([status, count]) => (
              <div key={status} className="p-2 rounded-lg bg-white/5">
                <div className="text-xs text-slate-400 mb-1 capitalize">
                  {status.split('_')[0]}
                </div>
                <div className="text-lg font-bold text-white">{count as number}</div>
              </div>
            ))}
          </div>
        </GlassCard>
      </div>

      {/* High Priority Alerts Table */}
      <GlassCard delay={0.7} noHover className="overflow-hidden p-0">
        <div className="px-6 py-5 border-b border-white/5 flex items-center justify-between">
          <h3 className="text-lg font-semibold text-white">High-Priority Intelligence</h3>
          <button className="text-xs text-blue-400 hover:text-blue-300 font-medium">
            View All Alerts →
          </button>
        </div>
        
        <div className="divide-y divide-white/5">
          {alerts
            ?.filter(a => a.risk_level === 'high' || a.risk_level === 'critical')
            .slice(0, 5)
            .map((alert, i) => (
              <motion.div 
                key={alert.alert_id} 
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.8 + (i * 0.1) }}
                className="px-6 py-4 hover:bg-white/5 transition-colors cursor-pointer group"
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-3">
                      <span
                        className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-bold tracking-wide
                          ${alert.risk_level === 'critical' 
                            ? 'bg-red-500/10 text-red-400 border border-red-500/20 shadow-[0_0_10px_rgba(239,68,68,0.2)]' 
                            : 'bg-orange-500/10 text-orange-400 border border-orange-500/20'
                          }`}
                      >
                        {alert.risk_level.toUpperCase()}
                      </span>
                      <span className="text-sm font-medium text-slate-200 group-hover:text-white transition-colors">
                        {alert.alert_type.replace(/_/g, ' ')}
                      </span>
                    </div>
                    <p className="mt-1 text-xs text-slate-500 font-mono">
                      ID: <span className="text-slate-400">{alert.alert_id}</span>
                    </p>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-bold text-white group-hover:text-blue-400 transition-colors">
                      Risk: {(alert.priority_score * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-slate-500">
                      {new Date(alert.created_at).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          {(!alerts || alerts.filter(a => a.risk_level === 'high' || a.risk_level === 'critical').length === 0) && (
            <div className="px-6 py-12 text-center">
              <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-green-500/10 mb-3">
                <CheckCircleIcon className="w-6 h-6 text-green-500" />
              </div>
              <p className="text-slate-400">No critical threats detected</p>
            </div>
          )}
        </div>
      </GlassCard>
    </div>
  );
}
