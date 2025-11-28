import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { alertApi } from '../services/api';
import type { Alert } from '../types';
import { 
  ExclamationTriangleIcon,
  FunnelIcon,
  MagnifyingGlassIcon 
} from '@heroicons/react/24/outline';

export default function Alerts() {
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [riskFilter, setRiskFilter] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');

  const { data: alerts, isLoading } = useQuery({
    queryKey: ['alerts', statusFilter, riskFilter],
    queryFn: () => alertApi.list({
      ...(statusFilter !== 'all' && { status: statusFilter }),
      ...(riskFilter !== 'all' && { risk_level: riskFilter }),
      limit: 100,
    }),
    refetchInterval: 15000,
  });

  const filteredAlerts = alerts?.filter(alert => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      alert.alert_id.toLowerCase().includes(query) ||
      alert.alert_type.toLowerCase().includes(query) ||
      alert.customer_ids.some(id => id.toLowerCase().includes(query))
    );
  });

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'critical': return 'bg-red-500/20 text-red-400 border-red-500';
      case 'high': return 'bg-orange-500/20 text-orange-400 border-orange-500';
      case 'medium': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500';
      case 'low': return 'bg-green-500/20 text-green-400 border-green-500';
      default: return 'bg-slate-500/20 text-slate-400 border-slate-500';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'bg-yellow-500/20 text-yellow-400';
      case 'under_investigation': return 'bg-blue-500/20 text-blue-400';
      case 'resolved_suspicious': return 'bg-red-500/20 text-red-400';
      case 'resolved_false_positive': return 'bg-green-500/20 text-green-400';
      case 'sar_generated': return 'bg-purple-500/20 text-purple-400';
      default: return 'bg-slate-500/20 text-slate-400';
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-white">Alerts</h1>
        <p className="mt-1 text-sm text-slate-400">
          Suspicious activity alerts from the AML detection system
        </p>
      </div>

      {/* Filters */}
      <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
        <div className="flex flex-col sm:flex-row gap-4">
          {/* Search */}
          <div className="flex-1">
            <div className="relative">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400" />
              <input
                type="text"
                placeholder="Search by Alert ID, Type, or Customer ID..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>
          </div>

          {/* Status Filter */}
          <div>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value="all">All Status</option>
              <option value="pending">Pending</option>
              <option value="under_investigation">Under Investigation</option>
              <option value="resolved_suspicious">Resolved - Suspicious</option>
              <option value="resolved_false_positive">Resolved - False Positive</option>
              <option value="sar_generated">SAR Generated</option>
            </select>
          </div>

          {/* Risk Filter */}
          <div>
            <select
              value={riskFilter}
              onChange={(e) => setRiskFilter(e.target.value)}
              className="px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              <option value="all">All Risk Levels</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </div>
        </div>
      </div>

      {/* Alerts List */}
      <div className="bg-slate-800 rounded-lg border border-slate-700 overflow-hidden">
        {filteredAlerts && filteredAlerts.length > 0 ? (
          <div className="divide-y divide-slate-700">
            {filteredAlerts.map((alert) => (
              <Link
                key={alert.alert_id}
                to={`/alerts/${alert.alert_id}`}
                className="block px-6 py-4 hover:bg-slate-700/50 transition-colors"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3 mb-2">
                      <span className={`inline-flex items-center rounded-md px-2 py-1 text-xs font-medium border ${getRiskColor(alert.risk_level)}`}>
                        {alert.risk_level.toUpperCase()}
                      </span>
                      <span className={`inline-flex items-center rounded-md px-2 py-1 text-xs font-medium ${getStatusColor(alert.status)}`}>
                        {alert.status.replace(/_/g, ' ').toUpperCase()}
                      </span>
                    </div>
                    
                    <h3 className="text-base font-semibold text-white mb-1">
                      {alert.alert_type.replace(/_/g, ' ')}
                    </h3>
                    
                    <p className="text-sm text-slate-400 mb-2">
                      {alert.alert_id}
                    </p>
                    
                    <div className="flex flex-wrap gap-2 text-xs text-slate-400">
                      <span>Confidence: {(alert.confidence_score * 100).toFixed(0)}%</span>
                      <span>•</span>
                      <span>{alert.patterns_detected.length} patterns detected</span>
                      <span>•</span>
                      <span>{alert.transaction_ids.length} transactions</span>
                    </div>
                  </div>
                  
                  <div className="text-right shrink-0">
                    <div className="text-2xl font-bold text-white mb-1">
                      {(alert.priority_score * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-slate-400">
                      Risk Score
                    </div>
                    <div className="text-xs text-slate-500 mt-2">
                      {new Date(alert.created_at).toLocaleDateString()}
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        ) : (
          <div className="px-6 py-12 text-center">
            <ExclamationTriangleIcon className="mx-auto h-12 w-12 text-slate-600" />
            <h3 className="mt-2 text-sm font-medium text-slate-400">No alerts found</h3>
            <p className="mt-1 text-sm text-slate-500">
              {searchQuery || statusFilter !== 'all' || riskFilter !== 'all'
                ? 'Try adjusting your filters'
                : 'No alerts have been generated yet'}
            </p>
          </div>
        )}
      </div>

      {/* Summary Stats */}
      {filteredAlerts && filteredAlerts.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
          <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
            <div className="text-sm text-slate-400">Total Alerts</div>
            <div className="text-2xl font-bold text-white">{filteredAlerts.length}</div>
          </div>
          <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
            <div className="text-sm text-slate-400">Pending</div>
            <div className="text-2xl font-bold text-yellow-400">
              {filteredAlerts.filter(a => a.status === 'pending').length}
            </div>
          </div>
          <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
            <div className="text-sm text-slate-400">High/Critical</div>
            <div className="text-2xl font-bold text-red-400">
              {filteredAlerts.filter(a => a.risk_level === 'high' || a.risk_level === 'critical').length}
            </div>
          </div>
          <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
            <div className="text-sm text-slate-400">Avg Risk Score</div>
            <div className="text-2xl font-bold text-white">
              {(filteredAlerts.reduce((sum, a) => sum + a.priority_score, 0) / filteredAlerts.length * 100).toFixed(0)}%
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

