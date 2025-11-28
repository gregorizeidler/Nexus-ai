// API Types
export interface Transaction {
  transaction_id: string;
  timestamp: string;
  amount: number;
  currency: string;
  transaction_type: string;
  sender_id: string;
  receiver_id: string;
  country_origin: string;
  country_destination: string;
  risk_score?: number;
}

export interface Alert {
  alert_id: string;
  created_at: string;
  updated_at: string;
  alert_type: string;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  priority_score: number;
  status: 'pending' | 'under_investigation' | 'resolved_false_positive' | 'resolved_suspicious' | 'sar_generated' | 'closed';
  transaction_ids: string[];
  customer_ids: string[];
  triggered_by: string[];
  patterns_detected: string[];
  confidence_score: number;
  explanation: string;
  evidence: Record<string, any>;
}

export interface SAR {
  sar_id: string;
  created_at: string;
  filed_at?: string;
  alert_id: string;
  subject_name: string;
  subject_id: string;
  activity_type: string;
  total_amount: number;
  currency: string;
  narrative: string;
  transaction_count: number;
  filed: boolean;
  confirmation_number?: string;
}

export interface SystemMetrics {
  timestamp: string;
  transactions: {
    total_processed: number;
  };
  alerts: {
    total_alerts: number;
    pending_count: number;
    by_status: Record<string, number>;
    by_risk_level: Record<string, number>;
  };
  sars: {
    total_sars: number;
    filed: number;
    pending_filing: number;
  };
  agents: {
    total_agents: number;
    enabled_agents: number;
  };
}

export interface ProcessTransactionResponse {
  transaction_id: string;
  status: string;
  processing_time: number;
  risk_assessment: {
    suspicious: boolean;
    risk_score: number;
    risk_level: string;
    confidence: number;
  };
  alert_created: boolean;
  alert_id?: string;
  patterns_detected: string[];
}

