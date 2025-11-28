import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { sarApi } from '../services/api';
import { DocumentTextIcon } from '@heroicons/react/24/outline';

export default function SARs() {
  const { data: sars, isLoading } = useQuery({
    queryKey: ['sars'],
    queryFn: () => sarApi.list({ limit: 100 }),
    refetchInterval: 30000,
  });

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
        <h1 className="text-3xl font-bold text-white">Suspicious Activity Reports</h1>
        <p className="mt-1 text-sm text-slate-400">
          Generated SARs for regulatory filing
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
          <div className="text-sm text-slate-400">Total SARs</div>
          <div className="text-2xl font-bold text-white">{sars?.length || 0}</div>
        </div>
        <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
          <div className="text-sm text-slate-400">Filed</div>
          <div className="text-2xl font-bold text-green-400">
            {sars?.filter(s => s.filed).length || 0}
          </div>
        </div>
        <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
          <div className="text-sm text-slate-400">Pending Filing</div>
          <div className="text-2xl font-bold text-yellow-400">
            {sars?.filter(s => !s.filed).length || 0}
          </div>
        </div>
      </div>

      {/* SARs List */}
      <div className="bg-slate-800 rounded-lg border border-slate-700 overflow-hidden">
        {sars && sars.length > 0 ? (
          <div className="divide-y divide-slate-700">
            {sars.map((sar) => (
              <Link
                key={sar.sar_id}
                to={`/sars/${sar.sar_id}`}
                className="block px-6 py-4 hover:bg-slate-700/50 transition-colors"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <span className={`inline-flex items-center rounded-md px-2 py-1 text-xs font-medium
                        ${sar.filed 
                          ? 'bg-green-500/20 text-green-400' 
                          : 'bg-yellow-500/20 text-yellow-400'
                        }`}>
                        {sar.filed ? 'FILED' : 'PENDING'}
                      </span>
                      {sar.filed && sar.confirmation_number && (
                        <span className="text-xs text-slate-500 font-mono">
                          {sar.confirmation_number}
                        </span>
                      )}
                    </div>
                    
                    <h3 className="text-base font-semibold text-white mb-1">
                      {sar.activity_type.replace(/_/g, ' ')}
                    </h3>
                    
                    <p className="text-sm text-slate-400 mb-2">
                      {sar.sar_id}
                    </p>
                    
                    <div className="flex flex-wrap gap-2 text-xs text-slate-400">
                      <span>Subject: {sar.subject_name}</span>
                      <span>•</span>
                      <span>{sar.transaction_count} transactions</span>
                      <span>•</span>
                      <span>{sar.total_amount.toLocaleString()} {sar.currency}</span>
                    </div>
                  </div>
                  
                  <div className="text-right shrink-0">
                    <div className="text-lg font-semibold text-white mb-1">
                      {sar.total_amount.toLocaleString()}
                    </div>
                    <div className="text-xs text-slate-400">
                      {sar.currency}
                    </div>
                    <div className="text-xs text-slate-500 mt-2">
                      {new Date(sar.created_at).toLocaleDateString()}
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        ) : (
          <div className="px-6 py-12 text-center">
            <DocumentTextIcon className="mx-auto h-12 w-12 text-slate-600" />
            <h3 className="mt-2 text-sm font-medium text-slate-400">No SARs generated yet</h3>
            <p className="mt-1 text-sm text-slate-500">
              SARs will be generated from confirmed suspicious alerts
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

