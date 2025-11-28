import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Transactions from './pages/Transactions';
import Alerts from './pages/Alerts';
import AlertDetail from './pages/AlertDetail';
import SARs from './pages/SARs';
import SARDetail from './pages/SARDetail';
import Agents from './pages/Agents';
import AIChat from './pages/AIChat';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 30000,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="transactions" element={<Transactions />} />
            <Route path="alerts" element={<Alerts />} />
            <Route path="alerts/:alertId" element={<AlertDetail />} />
            <Route path="sars" element={<SARs />} />
            <Route path="sars/:sarId" element={<SARDetail />} />
            <Route path="agents" element={<Agents />} />
            <Route path="ai-chat" element={<AIChat />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;

