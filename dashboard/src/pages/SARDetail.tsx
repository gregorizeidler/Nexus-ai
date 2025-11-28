import React from 'react';
import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { sarApi } from '../services/api';
import { ArrowLeftIcon } from '@heroicons/react/24/outline';

export default function SARDetail() {
  const { sarId } = useParams<{ sarId: string }>();

  const { data: sar, isLoading } = useQuery({
    queryKey: ['sar', sarId],
    queryFn: () => sarApi.getById(sarId!),
    enabled: !!sarId,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  if (!sar) {
    return (
      <div className="text-center py-12">
        <h2 className="text-2xl font-bold text-white">SAR not found</h2>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Link
          to="/sars"
          className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
        >
          <ArrowLeftIcon className="h-5 w-5" />
          Back to SARs
        </Link>
      </div>

      <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
        <div className="flex items-start justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-white mb-2">
              Suspicious Activity Report
            </h1>
            <p className="text-slate-400">{sar.sar_id}</p>
          </div>
          <span className={`inline-flex items-center rounded-md px-3 py-1.5 text-sm font-medium
            ${sar.filed 
              ? 'bg-green-500/20 text-green-400' 
              : 'bg-yellow-500/20 text-yellow-400'
            }`}>
            {sar.filed ? 'FILED' : 'PENDING FILING'}
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div>
            <h3 className="text-sm font-medium text-slate-400 mb-1">Subject</h3>
            <p className="text-lg text-white">{sar.subject_name}</p>
            <p className="text-sm text-slate-400">{sar.subject_id}</p>
          </div>
          <div>
            <h3 className="text-sm font-medium text-slate-400 mb-1">Activity Type</h3>
            <p className="text-lg text-white">{sar.activity_type.replace(/_/g, ' ')}</p>
          </div>
          <div>
            <h3 className="text-sm font-medium text-slate-400 mb-1">Total Amount</h3>
            <p className="text-lg font-semibold text-white">
              {sar.total_amount.toLocaleString()} {sar.currency}
            </p>
          </div>
          <div>
            <h3 className="text-sm font-medium text-slate-400 mb-1">Transactions</h3>
            <p className="text-lg text-white">{sar.transaction_count}</p>
          </div>
        </div>

        {sar.filed && sar.confirmation_number && (
          <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-4 mb-6">
            <h3 className="text-sm font-medium text-green-400 mb-1">Filing Confirmation</h3>
            <p className="text-lg font-mono text-white">{sar.confirmation_number}</p>
            {sar.filed_at && (
              <p className="text-sm text-slate-400 mt-1">
                Filed on {new Date(sar.filed_at).toLocaleString()}
              </p>
            )}
          </div>
        )}

        <div className="border-t border-slate-700 pt-6">
          <h3 className="text-lg font-semibold text-white mb-3">Narrative</h3>
          <pre className="text-sm text-slate-300 whitespace-pre-wrap font-mono bg-slate-900 p-4 rounded-lg max-h-96 overflow-y-auto">
            {sar.narrative}
          </pre>
        </div>

        <div className="mt-6 pt-6 border-t border-slate-700">
          <h3 className="text-lg font-semibold text-white mb-3">Related Information</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <div className="text-sm text-slate-400 mb-2">Related Alert</div>
              <Link
                to={`/alerts/${sar.alert_id}`}
                className="text-sm text-primary-400 hover:text-primary-300 font-mono"
              >
                {sar.alert_id}
              </Link>
            </div>
            <div>
              <div className="text-sm text-slate-400 mb-2">Created</div>
              <div className="text-sm text-white">
                {new Date(sar.created_at).toLocaleString()}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

