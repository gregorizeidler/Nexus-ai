import React from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  HomeIcon, 
  CreditCardIcon, 
  ExclamationTriangleIcon,
  DocumentTextIcon,
  CpuChipIcon,
  SparklesIcon,
  Bars3Icon,
  XMarkIcon
} from '@heroicons/react/24/outline';

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
  { name: 'Transactions', href: '/transactions', icon: CreditCardIcon },
  { name: 'Alerts', href: '/alerts', icon: ExclamationTriangleIcon },
  { name: 'SARs', href: '/sars', icon: DocumentTextIcon },
  { name: 'Agents', href: '/agents', icon: CpuChipIcon },
  { name: 'AI Chat', href: '/ai-chat', icon: SparklesIcon, special: true },
];

export default function Layout() {
  const location = useLocation();

  return (
    <div className="min-h-screen flex">
      {/* Sidebar for Desktop */}
      <aside className="hidden md:flex w-64 flex-col fixed inset-y-0 bg-white border-r border-slate-200 z-50 shadow-sm">
        <div className="flex items-center h-16 px-6 border-b border-slate-200">
          <Link to="/" className="flex items-center gap-2 group">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shadow-sm group-hover:shadow-md transition-shadow">
              <span className="text-white font-bold text-lg">N</span>
            </div>
            <span className="text-lg font-bold text-slate-900 tracking-tight">NEXUS AI</span>
          </Link>
        </div>

        <nav className="flex-1 px-4 py-6 space-y-2 overflow-y-auto">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href;
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`sidebar-link ${isActive ? 'active' : ''} group`}
              >
                <item.icon className={`h-5 w-5 transition-colors ${
                  item.special ? 'text-purple-400 group-hover:text-purple-300' : 
                  isActive ? 'text-blue-400' : 'text-slate-500 group-hover:text-slate-300'
                }`} />
                <span className={item.special ? 'text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-400 font-semibold' : ''}>
                  {item.name}
                </span>
                {isActive && (
                  <motion.div
                    layoutId="active-pill"
                    className="absolute left-0 w-1 h-6 bg-blue-500 rounded-r-full"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                  />
                )}
              </Link>
            );
          })}
        </nav>

        <div className="p-4 border-t border-slate-200">
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 flex items-center gap-3">
            <div className="relative">
              <div className="w-2 h-2 rounded-full bg-green-500" />
            </div>
            <div>
              <p className="text-xs font-medium text-slate-600">System Status</p>
              <p className="text-xs font-semibold text-green-700">Operational</p>
            </div>
          </div>
        </div>
      </aside>

      {/* Mobile Header */}
      <div className="md:hidden fixed top-0 left-0 right-0 h-16 bg-white border-b border-slate-200 z-50 flex items-center justify-between px-4 shadow-sm">
        <span className="font-bold text-slate-900">NEXUS AI</span>
        {/* Mobile menu button would go here */}
      </div>

      {/* Main Content */}
      <main className="flex-1 md:ml-64 min-h-screen p-4 md:p-8 pt-20 md:pt-8 bg-slate-50">
        <Outlet />
      </main>
    </div>
  );
}
