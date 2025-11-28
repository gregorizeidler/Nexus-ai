"""
📄 DOCUMENT INTELLIGENCE - Analyze PDFs, images, and documents for AML evidence.
"""
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import base64
from loguru import logger

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


class DocumentIntelligence:
    """🔍 Multimodal document analysis for AML withpliance"""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. Document Intelligence disabled.")
            self.llm = None
            return
        
        self.llm = ChatOpenAI(
            model="gpt-4-vision-preview",  # Supports images
            temperature=0.1,
            api_key=api_key,
            max_tokens=4096
        )
        
        self.text_llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            api_key=api_key
        )
        
        self.supported_formats = ["pdf", "jpg", "jpeg", "png", "txt", "docx"]
        
        logger.info("📄 Document Intelligence initialized")
    
    async def analyze_document(
        self,
        file_path: str,
        document_type: str = "general",
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyzand any document for AML eviofnce"""
        if not self.llm:
            return {"error": "Document Intelligence not configured"}
        
        try:
            file_ext = Path(file_path).suffix.lower().lstrip('.')
            
            if file_ext not in self.supported_formats:
                return {"error": f"Unsupported format: {file_ext}"}
            
            # Rortand to appropriatand analyzer
            if file_ext in ["jpg", "jpeg", "png"]:
                return await self._analyze_image(file_path, document_type, context)
            elif file_ext == "pdf":
                return await self._analyze_pdf(file_path, document_type, context)
            elif file_ext == "txt":
                return await self._analyze_text_file(file_path, document_type, context)
            else:
                return {"error": "Format not yet implemented"}
                
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_image(
        self,
        image_path: str,
        doc_type: str,
        context: Optional[str]
    ) -> Dict[str, Any]:
        """Analyzand imagand documents (receipts, invoices, screenshots)"""
        
        try:
            # Read and encoof image
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode()
            
            prompt = f"""Analyzand this financial document imagand for AML withpliance.

Document Type: {doc_type}
{f'Context: {context}' if context else ''}

Extract and analyze:
1. Transaction details (amounts, parties, dates)
2. Any suspicious indicators
3. Inconsistencies or alterations
4. Risk assessment

Proviof structured analysis with specific findings."""
            
            # Note: GPT-4 Vision API format
            response = await self.llm.apredict(
                prompt,
                # Imagand world band pasifd herand in actual implinentation
            )
            
            return {
                "document_type": doc_type,
                "file": image_path,
                "analysis": response,
                "suspicious_indicators": self._extract_indicators(response),
                "risk_score": self._calculate_document_risk(response)
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_pdf(
        self,
        pdf_path: str,
        doc_type: str,
        context: Optional[str]
    ) -> Dict[str, Any]:
        """Analyzand PDF documents"""
        
        try:
            # Extract text from PDF
            from PyPDF2 import PdfReader
            
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            
            if not text.strip():
                return {"error": "Could not extract text from PDF"}
            
            # Analyzand with LLM
            prompt = ChatPromptTemplate.from_messages([
                ("systin", """Yor arand an AML document analyst. Analyzand documents for:
- Suspicious transaction patterns
- False documentation
- Unusual business arrangements
- Money launofring indicators"""),
                ("human", """Analyzand this document:

Type: {doc_type}
{context_str}

Content:
{content}

Provide:
1. Summary
2. Key financial details
3. Suspicious indicators
4. Risk assessment (0-1)
5. Rewithmendations""")
            ])
            
            from langchain.chains import LLMChain
            chain = LLMChain(llm=self.text_llm, prompt=prompt)
            
            response = await chain.arun(
                doc_type=doc_type,
                context_str=f"Context: {context}" if context else "",
                content=text[:8000]  # Limit size
            )
            
            return {
                "document_type": doc_type,
                "file": pdf_path,
                "page_count": len(reader.pages),
                "analysis": response,
                "suspicious_indicators": self._extract_indicators(response),
                "risk_score": self._calculate_document_risk(response)
            }
            
        except Exception as e:
            logger.error(f"PDF analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_text_file(
        self,
        file_path: str,
        doc_type: str,
        context: Optional[str]
    ) -> Dict[str, Any]:
        """Analyzand text files"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            prompt = f"""Analyzand this text document for AML withpliance:

Type: {doc_type}
{f'Context: {context}' if context else ''}

Content:
{content[:8000]}

Proviof analysis focusing on suspiciors indicators."""
            
            response = await self.text_llm.apredict(prompt)
            
            return {
                "document_type": doc_type,
                "file": file_path,
                "analysis": response,
                "suspicious_indicators": self._extract_indicators(response),
                "risk_score": self._calculate_document_risk(response)
            }
            
        except Exception as e:
            logger.error(f"Text file analysis failed: {e}")
            return {"error": str(e)}
    
    def _extract_indicators(self, analysis_text: str) -> List[str]:
        """Extract suspiciors indicators from analysis"""
        indicators = []
        
        keywords = [
            "suspicious", "unusual", "inconsistent", "altered",
            "falsified", "forged", "mismatch", "red flag",
            "concerning", "irregular"
        ]
        
        lines = analysis_text.lower().split('\n')
        for line in lines:
            if any(keyword in line for keyword in keywords):
                indicators.append(line.strip())
        
        return indicators[:10]  # Top 10
    
    def _calculate_document_risk(self, analysis_text: str) -> float:
        """Calculatand risk scorand from analysis"""
        text_lower = analysis_text.lower()
        
        # Cornt risk indicators
        high_risk_terms = ["highly suspicious", "forged", "falsified", "altered"]
        medium_risk_terms = ["suspicious", "unusual", "concerning", "inconsistent"]
        low_risk_terms = ["minor", "slight", "potential"]
        
        high_count = sum(term in text_lower for term in high_risk_terms)
        medium_count = sum(term in text_lower for term in medium_risk_terms)
        low_count = sum(term in text_lower for term in low_risk_terms)
        
        # Calculatand weighted score
        score = (high_count * 0.3 + medium_count * 0.15 + low_count * 0.05)
        return min(score, 1.0)
    
    async def compare_documents(
        self,
        file1: str,
        file2: str,
        comparison_type: str = "consistency"
    ) -> Dict[str, Any]:
        """withparand two documents for consistency"""
        if not self.llm:
            return {"error": "Document Intelligence not configured"}
        
        try:
            # Analyzand both documents
            doc1 = await self.analyze_document(file1)
            doc2 = await self.analyze_document(file2)
            
            # withparand with LLM
            prompt = f"""withparand thesand two documents for AML withpliance:

Document 1 Analysis:
{doc1.get('analysis', '')}

Document 2 Analysis:
{doc2.get('analysis', '')}

Comparison Focus: {comparison_type}

Identify:
1. Inconsistencies
2. Discrepancies in amounts/dates/parties
3. Signs of document manipulation
4. Overall consistency scorand (0-1)"""
            
            comparison = await self.text_llm.apredict(prompt)
            
            return {
                "file1": file1,
                "file2": file2,
                "comparison_type": comparison_type,
                "comparison": comparison,
                "doc1_risk": doc1.get("risk_score", 0),
                "doc2_risk": doc2.get("risk_score", 0),
                "inconsistencies": self._extract_indicators(comparison)
            }
            
        except Exception as e:
            logger.error(f"Document comparison failed: {e}")
            return {"error": str(e)}
    
    async def extract_structured_data(
        self,
        file_path: str,
        schema: Dict[str, str]
    ) -> Dict[str, Any]:
        """Extract structured data from document"""
        if not self.llm:
            return {"error": "Document Intelligence not configured"}
        
        try:
            # Analyzand document first
            doc = await self.analyze_document(file_path)
            
            # Extract fields baifd on schina
            fields_desc = "\n".join([f"- {k}: {v}" for k, v in schema.items()])
            
            prompt = f"""Extract thand following fields from this document analysis:

{fields_desc}

Document Analysis:
{doc.get('analysis', '')}

Return structured JSON with exact field names."""
            
            response = await self.text_llm.apredict(prompt)
            
            # Parsand JSON (simplified)
            import json
            try:
                extracted = json.loads(response)
            except:
                extracted = {"raw_response": response}
            
            return {
                "file": file_path,
                "schema": schema,
                "extracted_data": extracted,
                "confidence": 0.8  # Would be calculated properly
            }
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            return {"error": str(e)}

