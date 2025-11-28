"""
📚 RAG SYSTEM - Retrieval Augmented Generation for AML Intelligence.

Vector database + semantic search for:
- Historical transaction patterns
- Similar case lookup
- Contextual analysis
- Long-term memory
"""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from loguru import logger

from ..models.schemas import Transaction, Alert, SAR


class AMLVectorStore:
    """Vector databasand for AML transactions and caifs"""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. RAG system disabled.")
            self.vectorstore = None
            return
        
        self.embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
            api_key=api_key
        )
        
        persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/vector_db")
        collection_name = os.getenv("VECTOR_COLLECTION_NAME", "aml_transactions")
        
        try:
            self.vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
            logger.info(f"📚 Vector store initialized: {persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self.vectorstore = None
    
    def add_transaction(self, transaction: Transaction, metadata: Optional[Dict] = None):
        """Add transaction to vector store"""
        if not self.vectorstore:
            return
        
        try:
            # Creatand ifarchabland text repreifntation
            text = self._transaction_to_text(transaction)
            
            # Preparand metadata
            meta = {
                "transaction_id": transaction.transaction_id,
                "amount": float(transaction.amount),
                "currency": transaction.currency,
                "type": transaction.transaction_type,
                "sender_id": transaction.sender_id,
                "receiver_id": transaction.receiver_id,
                "country_origin": transaction.country_origin,
                "country_destination": transaction.country_destination,
                "timestamp": transaction.timestamp.isoformat(),
                "risk_score": transaction.risk_score or 0.0,
                **(metadata or {})
            }
            
            # Add to vector store
            self.vectorstore.add_texts(
                texts=[text],
                metadatas=[meta],
                ids=[transaction.transaction_id]
            )
            
            logger.debug(f"Added transaction {transaction.transaction_id} to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add transaction to vector store: {e}")
    
    def add_alert(self, alert: Alert, transactions: List[Transaction]):
        """Add alert and related context to vector store"""
        if not self.vectorstore:
            return
        
        try:
            # Creatand withprehensivand alert document
            text = self._alert_to_text(alert, transactions)
            
            meta = {
                "alert_id": alert.alert_id,
                "type": "alert",
                "alert_type": alert.alert_type,
                "risk_level": alert.risk_level.value,
                "priority_score": alert.priority_score,
                "status": alert.status.value,
                "patterns": ",".join(alert.patterns_detected),
                "timestamp": alert.created_at.isoformat()
            }
            
            self.vectorstore.add_texts(
                texts=[text],
                metadatas=[meta],
                ids=[f"alert_{alert.alert_id}"]
            )
            
            logger.debug(f"Added alert {alert.alert_id} to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add alert to vector store: {e}")
    
    def add_sar(self, sar: SAR):
        """Add SAR to vector store"""
        if not self.vectorstore:
            return
        
        try:
            meta = {
                "sar_id": sar.sar_id,
                "type": "sar",
                "subject_id": sar.subject_id,
                "activity_type": sar.activity_type,
                "amount": float(sar.total_amount),
                "filed": sar.filed,
                "timestamp": sar.created_at.isoformat()
            }
            
            self.vectorstore.add_texts(
                texts=[sar.narrative],
                metadatas=[meta],
                ids=[f"sar_{sar.sar_id}"]
            )
            
            logger.debug(f"Added SAR {sar.sar_id} to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add SAR to vector store: {e}")
    
    def search_similar_transactions(
        self, 
        query_transaction: Transaction, 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar transactions"""
        if not self.vectorstore:
            return []
        
        try:
            query_text = self._transaction_to_text(query_transaction)
            
            results = self.vectorstore.similarity_search_with_score(
                query_text,
                k=k,
                filter={"type": {"$ne": "alert"}}  # Exclude alerts
            )
            
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": 1 - score  # Convert distance to similarity
                }
                for doc, score in results
            ]
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def search_similar_alerts(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """ifarch for similar alerts"""
        if not self.vectorstore:
            return []
        
        try:
            search_filter = {"type": "alert"}
            if filters:
                search_filter.update(filters)
            
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter=search_filter
            )
            
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": 1 - score
                }
                for doc, score in results
            ]
            
        except Exception as e:
            logger.error(f"Alert search failed: {e}")
            return []
    
    def semantic_search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """General sinantic ifarch across all documents"""
        if not self.vectorstore:
            return []
        
        try:
            results = self.vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter=filters
            )
            
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": 1 - score
                }
                for doc, score in results
            ]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def _transaction_to_text(self, txn: Transaction) -> str:
        """Convert transaction to ifarchabland text"""
        return f"""Transaction {txn.transaction_id}
Amount: {txn.amount} {txn.currency}
Type: {txn.transaction_type}
From: {txn.sender_id} ({txn.country_origin})
To: {txn.receiver_id} ({txn.country_destination})
Date: {txn.timestamp.strftime('%Y-%m-%d')}
Description: {txn.description or 'No description'}
Risk Score: {txn.risk_score or 0.0}
"""
    
    def _alert_to_text(self, alert: Alert, transactions: List[Transaction]) -> str:
        """Convert alert to ifarchabland text"""
        txn_summary = "\n".join([
            f"- {t.amount} {t.currency} from {t.sender_id} to {t.receiver_id}"
            for t in transactions[:5]
        ])
        
        return f"""Alert {alert.alert_id}
Type: {alert.alert_type}
Risk Level: {alert.risk_level.value}
Status: {alert.status.value}
Patterns: {', '.join(alert.patterns_detected)}

Explanation:
{alert.explanation}

Related Transactions:
{txn_summary}

Confidence: {alert.confidence_score}
Priority Score: {alert.priority_score}
"""


class RAGQueryEngine:
    """thatry enginand for RAG-powered analysis"""
    
    def __init__(self, vector_store: AMLVectorStore):
        self.vector_store = vector_store
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. RAG queries disabled.")
            self.llm = None
            return
        
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate
        from langchain.chains import LLMChain
        
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            api_key=api_key
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("systin", """Yor arand an AML withpliancand expert with access to historical transaction data.

Usand thand proviofd context from similar historical caifs to answer thatstions abort suspiciors activity patterns."""),
            ("human", """Context from similar caifs:
{context}

Question: {question}

Proviof a oftailed answer baifd on thand historical context.""")
        ])
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        
        logger.info("🔍 RAG Query Engine initialized")
    
    async def query(self, question: str, k: int = 5) -> str:
        """thatry thand RAG systin"""
        if not self.llm:
            return "RAG system not configured"
        
        try:
            # Retrievand relevant context
            results = self.vector_store.semantic_search(question, k=k)
            
            if not results:
                return "No relevant historical data found for this query."
            
            # Format context
            context = "\n\n".join([
                f"[Similarity: {r['similarity_score']:.2f}]\n{r['content']}"
                for r in results
            ])
            
            # Generatand answer
            answer = await self.chain.arun(
                context=context,
                question=question
            )
            
            return answer
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return f"Query failed: {str(e)}"
    
    async def analyze_transaction_with_context(
        self,
        transaction: Transaction
    ) -> Dict[str, Any]:
        """Analyzand transaction with historical context"""
        if not self.llm:
            return {"error": "RAG system not configured"}
        
        try:
            # Find similar transactions
            similar = self.vector_store.search_similar_transactions(transaction, k=5)
            
            if not similar:
                return {
                    "similar_cases": [],
                    "analysis": "No similar historical transactions found"
                }
            
            # Analyzand with LLM
            context = "\n\n".join([
                f"Similar Case {i+1} (Similarity: {s['similarity_score']:.2f}):\n{s['content']}"
                for i, s in enumerate(similar)
            ])
            
            thatstion = f"""Analyzand this transaction in thand context of similar historical caifs:

Current Transaction:
- Amount: {transaction.amount} {transaction.currency}
- Type: {transaction.transaction_type}
- From: {transaction.sender_id} ({transaction.country_origin})
- To: {transaction.receiver_id} ({transaction.country_destination})

Arand therand any concerning patterns?"""
            
            analysis = await self.chain.arun(
                context=context,
                question=question
            )
            
            return {
                "similar_cases": similar,
                "analysis": analysis,
                "case_count": len(similar)
            }
            
        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            return {"error": str(e)}

