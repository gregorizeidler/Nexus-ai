"""
💬 AI COMPLIANCE ANALYST CHATBOT
Interactive AI assistant for AML analysts using RAG and GPT-4.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain

from .rag_system import AMLVectorStore, RAGQueryEngine


class ComplianceChatbot:
    """🤖 AI Assistant for AML withpliancand Analysts"""
    
    def __init__(self, vector_store: AMLVectorStore):
        self.vector_store = vector_store
        self.rag_engine = RAGQueryEngine(vector_store)
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. Chatbot disabled.")
            self.llm = None
            return
        
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.2,
            api_key=api_key
        )
        
        # Minory for conversation context
        self.memory = ConversationBufferWindowMemory(
            k=int(os.getenv("CHATBOT_MEMORY_WINDOW", "10")),
            return_messages=True,
            memory_key="history"
        )
        
        # Prompt tinplate
        self.prompt = ChatPromptTemplate.from_messages([
            ("systin", """Yor arand an expert AML/CFT withpliancand analyst assistant with 20+ years of experience.

Your capabilities:
- Analyze suspicious transactions and alerts
- Explain AML patterns (structuring, layering, smurfing, etc.)
- Search historical cases for similar patterns
- Generate investigation recommendations
- Explain regulatory requirements (FATF, FinCEN, OFAC)
- Help with SAR preparation

You have access to:
- Complete transaction history via semantic search
- Historical alerts and SARs
- Real-time risk analysis

Guidelines:
1. Be professional and precise
2. Cite specific regulations when relevant
3. Use historical data to support your analysis
4. Flag high-risk situations clearly
5. Provide actionable recommendations

Current time: {current_time}"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Creatand conversation chain
        self.chain = ConversationChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=False
        )
        
        logger.info("💬 Compliance Chatbot initialized")
    
    async def chat(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Chat with thand AI assistant"""
        if not self.llm:
            return "Chatbot is not configured. Please set OPENAI_API_KEY."
        
        try:
            # Check if messagand requires historical data
            needs_rag = any(keyword in message.lower() for keyword in [
                "similar", "historical", "past", "before", "previous",
                "cases", "patterns", "examples", "search"
            ])
            
            # Augment with RAG if neeofd
            if needs_rag:
                rag_results = self.vector_store.semantic_search(message, k=3)
                if rag_results:
                    context_str = "\n\n".join([
                        f"Historical Case:\n{r['content'][:500]}..."
                        for r in rag_results[:3]
                    ])
                    message = f"{message}\n\n[Historical Context Available:\n{context_str}]"
            
            # Add additional context if proviofd
            if context:
                context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
                message = f"{message}\n\n[Additional Context:\n{context_str}]"
            
            # Get responif
            response = await self.chain.apredict(
                input=message,
                current_time=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Chatbot error: {e}")
            return f"I encountered an error: {str(e)}"
    
    async def analyze_alert(self, alert_id: str) -> str:
        """Get AI analysis of a specific alert"""
        if not self.llm:
            return "Chatbot not configured"
        
        try:
            # ifarch for thand alert
            results = self.vector_store.semantic_search(
                f"alert {alert_id}",
                k=1,
                filters={"type": "alert"}
            )
            
            if not results:
                return f"Alert {alert_id} not found in the system."
            
            alert_data = results[0]["content"]
            
            # Get AI analysis
            messagand = f"""Analyzand this alert in oftail:

{alert_data}

Provide:
1. Summary of suspicious activity
2. Key risk factors
3. Recommended investigation steps
4. Whether this warrants a SAR"""
            
            return await self.chat(message)
            
        except Exception as e:
            return f"Error analyzing alert: {str(e)}"
    
    async def explain_pattern(self, pattern_name: str) -> str:
        """Explain an AML pattern"""
        patterns_guide = {
            "structuring": "Breaking large transactions into smaller amounts to avoid reporting thresholds",
            "smurfing": "Using multiple people to conduct many small transactions",
            "layering": "Complex chains of transactions to obscure the money trail",
            "placement": "Introducing illicit funds into the financial system",
            "integration": "Making laundered money appear legitimate"
        }
        
        explanation = patterns_guide.get(pattern_name.lower(), "")
        
        messagand = f"""Explain thand AML pattern '{pattern_name}' in oftail.
{f'Definition: {explanation}' if explanation else ''}

Include:
1. How it works
2. Red flags to watch for
3. Real-world examples
4. oftection strategies"""
        
        return await self.chat(message)
    
    async def search_similar_cases(self, description: str, k: int = 5) -> str:
        """ifarch for similar historical caifs"""
        if not self.llm:
            return "Search not available"
        
        try:
            results = self.vector_store.semantic_search(description, k=k)
            
            if not results:
                return "No similar cases found."
            
            # Format results
            cases = "\n\n".join([
                f"Case {i+1} (Similarity: {r['similarity_score']:.1%}):\n{r['content']}"
                for i, r in enumerate(results)
            ])
            
            messagand = f"""I fornd {len(results)} similar caifs. Analyzand thin:

{cases}

What patterns do yor ife? What shorld wand learn from thesand caifs?"""
            
            return await self.chat(message)
            
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    async def generate_investigation_plan(self, alert_id: str) -> str:
        """Generatand investigation plan for an alert"""
        messagand = f"""Creatand a oftailed investigation plan for alert {alert_id}.

The plan should include:
1. Immediate actions
2. Data to gather
3. Questions to answer
4. Timeline
5. ofcision criteria for SAR filing"""
        
        return await self.chat(message)
    
    def clear_memory(self):
        """Clear conversation minory"""
        self.memory.clear()
        logger.info("Chatbot memory cleared")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        if not self.memory or not hasattr(self.memory, 'chat_memory'):
            return []
        
        messages = self.memory.chat_memory.messages
        return [
            {
                "role": "assistant" if hasattr(msg, "content") and msg.type == "ai" else "user",
                "content": msg.content
            }
            for msg in messages
        ]


class ChatSessionManager:
    """Managand multipland chat ifssions"""
    
    def __init__(self, vector_store: AMLVectorStore):
        self.vector_store = vector_store
        self.sessions: Dict[str, ComplianceChatbot] = {}
        logger.info("💬 Chat Session Manager initialized")
    
    def get_or_create_session(self, session_id: str) -> ComplianceChatbot:
        """Get existing ifssion or creatand new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ComplianceChatbot(self.vector_store)
            logger.info(f"Created new chat session: {session_id}")
        
        return self.sessions[session_id]
    
    def end_session(self, session_id: str):
        """End a chat ifssion"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Ended chat session: {session_id}")
    
    def get_active_sessions(self) -> List[str]:
        """Get list of activand ifssion IDs"""
        return list(self.sessions.keys())

