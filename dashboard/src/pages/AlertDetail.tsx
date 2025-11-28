import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { alertApi } from '../services/api';
import { ArrowLeftIcon } from '@heroicons/react/24/outline';

export default function AlertDetail() {
  const { alertId } = useParams<{ alertId: string }>();

  const { data: alert, isLoading } = useQuery({
    queryKey: ['alert', alertId],
    queryFn: () => alertApi.getById(alertId!),
    enabled: !!alertId,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  if (!alert) {
    return (
      <div className="text-center py-12">
        <h2 className="text-2xl font-bold text-white">Alert not found</h2>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Link
          to="/alerts"
          className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
        >
          <ArrowLeftIcon className="h-5 w-5" />
          Back to Alerts
        </Link>
      </div>

      <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
        <div className="flex items-start justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-white mb-2">
              {alert.alert_type.replace(/_/g, ' ')}
            </h1>
            <p className="text-slate-400">{alert.alert_id}</p>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-white">
              {(alert.priority_score * 100).toFixed(0)}%
            </div>
            <div className="text-sm text-slate-400">Risk Score</div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div>
            <div className="text-sm text-slate-400 mb-1">Risk Level</div>
            <span className={`inline-flex items-center rounded-md px-3 py-1 text-sm font-medium
              ${alert.risk_level === 'critical' ? 'bg-red-500/20 text-red-400' :
                alert.risk_level === 'high' ? 'bg-orange-500/20 text-orange-400' :
                alert.risk_level === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                'bg-green-500/20 text-green-400'
              }`}>
              {alert.risk_level.toUpperCase()}
            </span>
          </div>
          <div>
            <div className="text-sm text-slate-400 mb-1">Status</div>
            <span className="inline-flex items-center rounded-md bg-blue-500/20 text-blue-400 px-3 py-1 text-sm font-medium">
              {alert.status.replace(/_/g, ' ').toUpperCase()}
            </span>
          </div>
          <div>
            <div className="text-sm text-slate-400 mb-1">Confidence</div>
            <div className="text-lg font-semibold text-white">
              {(alert.confidence_score * 100).toFixed(0)}%
            </div>
          </div>
        </div>

        <div className="border-t border-slate-700 pt-6">
          <h3 className="text-lg font-semibold text-white mb-3">Explanation</h3>
          <pre className="text-sm text-slate-300 whitespace-pre-wrap font-mono bg-slate-900 p-4 rounded-lg">
            {alert.explanation}
          </pre>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
          <div>
            <h3 className="text-lg font-semibold text-white mb-3">Patterns Detected</h3>
            <div className="space-y-2">
              {alert.patterns_detected.map((pattern, idx) => (
                <div key={idx} className="bg-slate-900 rounded-lg px-4 py-2">
                  <span className="text-sm text-slate-300">
                    {pattern.replace(/_/g, ' ').toUpperCase()}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-3">Triggered By</h3>
            <div className="space-y-2">
              {alert.triggered_by.map((agent, idx) => (
                <div key={idx} className="bg-slate-900 rounded-lg px-4 py-2">
                  <span className="text-sm text-slate-300">
                    {agent}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
          <div>
            <h3 className="text-lg font-semibold text-white mb-3">Related Transactions</h3>
            <div className="text-slate-300">
              {alert.transaction_ids.length} transaction(s)
            </div>
            <div className="mt-2 space-y-1">
              {alert.transaction_ids.slice(0, 5).map((txnId, idx) => (
                <div key={idx} className="text-sm text-slate-400 font-mono">
                  {txnId}
                </div>
              ))}
              {alert.transaction_ids.length > 5 && (
                <div className="text-sm text-slate-500">
                  +{alert.transaction_ids.length - 5} more
                </div>
              )}
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-3">Related Customers</h3>
            <div className="text-slate-300">
              {alert.customer_ids.length} customer(s)
            </div>
            <div className="mt-2 space-y-1">
              {alert.customer_ids.map((custId, idx) => (
                <div key={idx} className="text-sm text-slate-400 font-mono">
                  {custId}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-6 pt-6 border-t border-slate-700">
          <div className="text-xs text-slate-500">
            Created: {new Date(alert.created_at).toLocaleString()} • 
            Updated: {new Date(alert.updated_at).toLocaleString()}
          </div>
        </div>
      </div>
    </div>
  );
}

