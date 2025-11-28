import React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { systemApi } from '../services/api';
import { CpuChipIcon, CheckCircleIcon, XCircleIcon } from '@heroicons/react/24/outline';

export default function Agents() {
  const queryClient = useQueryClient();

  const { data: agentStatus, isLoading } = useQuery({
    queryKey: ['agent-status'],
    queryFn: systemApi.getAgentStatus,
    refetchInterval: 10000,
  });

  const toggleAgentMutation = useMutation({
    mutationFn: async ({ agentId, enable }: { agentId: string; enable: boolean }) => {
      return enable ? systemApi.enableAgent(agentId) : systemApi.disableAgent(agentId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['agent-status'] });
    },
  });

  const getAgentTypeColor = (type: string) => {
    switch (type) {
      case 'ingestion': return 'bg-blue-500/20 text-blue-400';
      case 'enrichment': return 'bg-purple-500/20 text-purple-400';
      case 'rules_analysis': return 'bg-green-500/20 text-green-400';
      case 'ml_analysis': return 'bg-orange-500/20 text-orange-400';
      case 'network_analysis': return 'bg-pink-500/20 text-pink-400';
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
        <h1 className="text-3xl font-bold text-white">Analysis Agents</h1>
        <p className="mt-1 text-sm text-slate-400">
          Manage and monitor the multi-agent detection system
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
          <div className="text-sm text-slate-400">Total Agents</div>
          <div className="text-2xl font-bold text-white">{agentStatus?.total_agents || 0}</div>
        </div>
        <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
          <div className="text-sm text-slate-400">Active</div>
          <div className="text-2xl font-bold text-green-400">
            {agentStatus?.enabled_agents || 0}
          </div>
        </div>
        <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
          <div className="text-sm text-slate-400">Disabled</div>
          <div className="text-2xl font-bold text-red-400">
            {(agentStatus?.total_agents || 0) - (agentStatus?.enabled_agents || 0)}
          </div>
        </div>
      </div>

      {/* Agents List */}
      <div className="bg-slate-800 rounded-lg border border-slate-700 overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-700">
          <h2 className="text-lg font-semibold text-white">Agent Pipeline</h2>
          <p className="text-sm text-slate-400 mt-1">
            Execution order: {agentStatus?.execution_order?.join(' → ')}
          </p>
        </div>
        
        <div className="divide-y divide-slate-700">
          {agentStatus?.agents && Object.entries(agentStatus.agents).map(([agentId, agent]: [string, any]) => (
            <div key={agentId} className="px-6 py-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4 flex-1">
                  <CpuChipIcon className={`h-8 w-8 ${agent.enabled ? 'text-green-400' : 'text-slate-600'}`} />
                  
                  <div className="flex-1">
                    <h3 className="text-base font-semibold text-white mb-1">
                      {agentId.split('_').map((w: string) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
                    </h3>
                    
                    <div className="flex items-center gap-2">
                      <span className={`inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium ${getAgentTypeColor(agent.type)}`}>
                        {agent.type.replace(/_/g, ' ').toUpperCase()}
                      </span>
                      
                      {agent.enabled ? (
                        <span className="flex items-center gap-1 text-xs text-green-400">
                          <CheckCircleIcon className="h-4 w-4" />
                          Active
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 text-xs text-red-400">
                          <XCircleIcon className="h-4 w-4" />
                          Disabled
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => toggleAgentMutation.mutate({ agentId, enable: !agent.enabled })}
                  disabled={toggleAgentMutation.isPending}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors
                    ${agent.enabled
                      ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                      : 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
                    }
                    disabled:opacity-50 disabled:cursor-not-allowed
                  `}
                >
                  {toggleAgentMutation.isPending ? 'Processing...' : (agent.enabled ? 'Disable' : 'Enable')}
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Agent Descriptions */}
      <div className="bg-slate-800 rounded-lg border border-slate-700 p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Agent Descriptions</h3>
        <div className="space-y-4 text-sm">
          <div>
            <h4 className="font-medium text-white mb-1">Data Ingestion Agent</h4>
            <p className="text-slate-400">Validates and normalizes raw transaction data from multiple sources.</p>
          </div>
          <div>
            <h4 className="font-medium text-white mb-1">Enrichment Agent</h4>
            <p className="text-slate-400">Adds contextual information including sanctions lists, PEP status, and country risk assessments.</p>
          </div>
          <div>
            <h4 className="font-medium text-white mb-1">Customer Profile Agent</h4>
            <p className="text-slate-400">Enriches transactions with historical customer behavior and KYC information.</p>
          </div>
          <div>
            <h4 className="font-medium text-white mb-1">Rules-Based Agent</h4>
            <p className="text-slate-400">Applies predefined AML/CFT rules based on regulatory requirements (FATF, OFAC).</p>
          </div>
          <div>
            <h4 className="font-medium text-white mb-1">Behavioral ML Agent</h4>
            <p className="text-slate-400">Uses machine learning to detect anomalies in customer behavior patterns.</p>
          </div>
          <div>
            <h4 className="font-medium text-white mb-1">Network Analysis Agent</h4>
            <p className="text-slate-400">Analyzes transaction networks to identify layering, smurfing, and circular patterns.</p>
          </div>
        </div>
      </div>
    </div>
  );
}

