import axios from 'axios';
import type { 
  Transaction, 
  Alert, 
  SAR, 
  SystemMetrics,
  ProcessTransactionResponse 
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Transaction APIs
export const transactionApi = {
  submit: async (transaction: Omit<Transaction, 'risk_score'>): Promise<ProcessTransactionResponse> => {
    const { data } = await api.post('/api/v1/transactions', transaction);
    return data;
  },

  submitBatch: async (transactions: Transaction[]) => {
    const { data } = await api.post('/api/v1/transactions/batch', transactions);
    return data;
  },

  getById: async (transactionId: string): Promise<Transaction> => {
    const { data } = await api.get(`/api/v1/transactions/${transactionId}`);
    return data.transaction;
  },
};

// Alert APIs
export const alertApi = {
  list: async (params?: {
    status?: string;
    risk_level?: string;
    limit?: number;
  }): Promise<Alert[]> => {
    const { data } = await api.get('/api/v1/alerts', { params });
    return data;
  },

  getPrioritized: async (): Promise<Alert[]> => {
    const { data } = await api.get('/api/v1/alerts/prioritized');
    return data;
  },

  getById: async (alertId: string): Promise<Alert> => {
    const { data } = await api.get(`/api/v1/alerts/${alertId}`);
    return data;
  },

  updateStatus: async (
    alertId: string,
    status: string,
    notes?: string,
    assigned_to?: string
  ) => {
    const { data } = await api.put(`/api/v1/alerts/${alertId}/status`, {
      status,
      notes,
      assigned_to,
    });
    return data;
  },

  getStatistics: async () => {
    const { data } = await api.get('/api/v1/alerts/statistics');
    return data;
  },
};

// SAR APIs
export const sarApi = {
  generate: async (alertId: string, additionalInfo?: Record<string, any>): Promise<SAR> => {
    const { data } = await api.post('/api/v1/sar/generate', null, {
      params: { alert_id: alertId },
      data: additionalInfo,
    });
    return data;
  },

  file: async (sarId: string, filedBy: string) => {
    const { data } = await api.post(`/api/v1/sar/${sarId}/file`, null, {
      params: { filed_by: filedBy },
    });
    return data;
  },

  getById: async (sarId: string): Promise<SAR> => {
    const { data } = await api.get(`/api/v1/sar/${sarId}`);
    return data;
  },

  list: async (params?: { filed?: boolean; limit?: number }): Promise<SAR[]> => {
    const { data } = await api.get('/api/v1/sar', { params });
    return data;
  },

  getStatistics: async () => {
    const { data } = await api.get('/api/v1/sar/statistics');
    return data;
  },
};

// System APIs
export const systemApi = {
  health: async () => {
    const { data } = await api.get('/health');
    return data;
  },

  metrics: async (): Promise<SystemMetrics> => {
    const { data } = await api.get('/api/v1/system/metrics');
    return data;
  },

  getAgentStatus: async () => {
    const { data } = await api.get('/api/v1/agents/status');
    return data;
  },

  enableAgent: async (agentId: string) => {
    const { data } = await api.put(`/api/v1/agents/${agentId}/enable`);
    return data;
  },

  disableAgent: async (agentId: string) => {
    const { data } = await api.put(`/api/v1/agents/${agentId}/disable`);
    return data;
  },
};

export default api;

