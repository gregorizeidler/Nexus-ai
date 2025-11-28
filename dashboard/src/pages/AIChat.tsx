import React, { useState, useRef, useEffect } from 'react';
import { useMutation } from '@tanstack/react-query';
import axios from 'axios';
import { 
  PaperAirplaneIcon, 
  SparklesIcon,
  UserIcon,
  CpuChipIcon
} from '@heroicons/react/24/outline';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export default function AIChat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: 'Hello! I\'m your AI Compliance Analyst. I can help you with:\n\n- Analyzing suspicious transactions and alerts\n- Explaining AML patterns (structuring, layering, etc.)\n- Searching historical cases for similar patterns\n- Providing regulatory guidance\n- Assisting with SAR preparation\n\nWhat would you like to know?',
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const chatMutation = useMutation({
    mutationFn: async (message: string) => {
      const { data } = await axios.post('/api/v1/ai/chat', {
        message,
        session_id: sessionId || undefined
      });
      return data;
    },
    onSuccess: (data) => {
      setSessionId(data.session_id);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.response,
        timestamp: new Date()
      }]);
    }
  });

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!input.trim()) return;

    // Add user message
    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    chatMutation.mutate(input);
    setInput('');
  };

  const quickQuestions = [
    "Explain what structuring is",
    "Show me similar high-risk alerts",
    "What are the red flags for layering?",
    "Help me analyze alert ALT-20240115-ABC"
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <SparklesIcon className="h-8 w-8 text-yellow-400" />
        <div>
          <h1 className="text-3xl font-bold text-white">AI Compliance Analyst</h1>
          <p className="mt-1 text-sm text-slate-400">
            Powered by GPT-4 with RAG-enhanced knowledge base
          </p>
        </div>
      </div>

      {/* Chat Interface */}
      <div className="bg-slate-800 rounded-lg border border-slate-700 flex flex-col" style={{ height: 'calc(100vh - 280px)' }}>
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((message, idx) => (
            <div
              key={idx}
              className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {message.role === 'assistant' && (
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                    <CpuChipIcon className="h-5 w-5 text-white" />
                  </div>
                </div>
              )}
              
              <div className={`max-w-3xl ${message.role === 'user' ? 'order-first' : ''}`}>
                <div
                  className={`rounded-lg px-4 py-3 ${
                    message.role === 'user'
                      ? 'bg-primary-600 text-white'
                      : 'bg-slate-700 text-slate-100'
                  }`}
                >
                  <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                </div>
                <p className="text-xs text-slate-500 mt-1 px-1">
                  {message.timestamp.toLocaleTimeString()}
                </p>
              </div>

              {message.role === 'user' && (
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 rounded-full bg-slate-600 flex items-center justify-center">
                    <UserIcon className="h-5 w-5 text-white" />
                  </div>
                </div>
              )}
            </div>
          ))}
          
          {chatMutation.isPending && (
            <div className="flex gap-3 justify-start">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                  <CpuChipIcon className="h-5 w-5 text-white animate-pulse" />
                </div>
              </div>
              <div className="bg-slate-700 rounded-lg px-4 py-3">
                <div className="flex gap-2">
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Quick Questions */}
        {messages.length === 1 && (
          <div className="px-6 py-3 border-t border-slate-700">
            <p className="text-xs text-slate-400 mb-2">Quick questions:</p>
            <div className="flex flex-wrap gap-2">
              {quickQuestions.map((question, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setInput(question);
                  }}
                  className="text-xs px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-full transition-colors"
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Input */}
        <div className="p-4 border-t border-slate-700">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask me anything about AML compliance..."
              disabled={chatMutation.isPending}
              className="flex-1 px-4 py-3 bg-slate-900 border border-slate-700 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={!input.trim() || chatMutation.isPending}
              className="px-6 py-3 bg-primary-600 hover:bg-primary-700 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <PaperAirplaneIcon className="h-5 w-5" />
              Send
            </button>
          </form>
        </div>
      </div>

      {/* Session Info */}
      {sessionId && (
        <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-3">
          <p className="text-xs text-slate-400">
            Session ID: <span className="font-mono text-slate-300">{sessionId}</span>
          </p>
        </div>
      )}
    </div>
  );
}

