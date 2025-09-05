#!/usr/bin/env python3
"""
Enhanced RAG Pipeline with LlamaIndex Excel processing integration.
"""

from typing import List, Dict, Any, Optional, Tuple, Generator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger

from .ollama_client import OllamaClient
from src.vector_db.document_indexer import DocumentIndexer
from src.vector_db.unified_vector_store import UnifiedVectorStore
from src.document_processing.llamaindex_excel_processor import LlamaIndexExcelProcessor, LlamaIndexExcelChunk


@dataclass
class EnhancedRAGQuery:
    """Enhanced RAG query with Excel-specific context."""
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    excel_chunks: List[LlamaIndexExcelChunk]
    generated_response: str
    metadata: Dict[str, Any]


@dataclass
class EnhancedRAGResponse:
    """Enhanced RAG response with Excel-specific information."""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    excel_sources: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any]
    direct_answer: Optional[str] = None


class EnhancedRAGPipeline:
    """Enhanced RAG pipeline with LlamaIndex Excel processing."""
    
    def __init__(self, document_indexer: DocumentIndexer, ollama_client: OllamaClient, 
                 enable_llamaindex: bool = True):
        """Initialize the enhanced RAG pipeline."""
        self.document_indexer = document_indexer
        self.ollama_client = ollama_client
        self.enable_llamaindex = enable_llamaindex
        
        # Initialize LlamaIndex Excel processor if enabled
        if self.enable_llamaindex:
            self.excel_processor = LlamaIndexExcelProcessor(enable_semantic_search=True)
            logger.info("✅ LlamaIndex Excel processor enabled")
        else:
            self.excel_processor = None
            logger.info("LlamaIndex Excel processor disabled")
        
        # Enhanced system prompt for Excel data
        self.excel_system_prompt = """You are a helpful AI assistant specialized in analyzing financial and Excel data. 
        Use the provided context to answer questions about financial records, budgets, expenses, and income. 
        When dealing with Excel data, pay special attention to:
        - Row data that contains actual values and amounts
        - Column descriptions that explain what each field represents
        - Table summaries that provide overview information
        
        Be precise with numbers and financial data. If you find specific amounts or values in the context, 
        cite them exactly. If the context doesn't contain enough information, say so clearly."""
        
        logger.info("Enhanced RAG pipeline initialized")
    
    def query(self, question: str, n_chunks: int = 5, 
              temperature: float = 0.7, max_tokens: int = 2048,
              system_prompt: str = None, use_direct_answer: bool = True,
              excel_only: bool = False) -> EnhancedRAGResponse:
        """Execute an enhanced RAG query with Excel processing."""
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing enhanced RAG query: {question[:100]}...")
            
            # Step 1: Retrieve relevant documents (skip LEANN if Excel-only mode)
            retrieved_chunks = []
            if not excel_only:
                retrieved_chunks = self._retrieve_relevant_chunks(question, n_chunks)
            
            # Step 2: Check if we have Excel files and process them
            excel_chunks = []
            direct_answer = None
            
            if self.enable_llamaindex and (self._is_excel_related_query(question) or excel_only):
                excel_chunks = self._process_excel_files(question)
                
                # Try direct answer first if enabled
                if use_direct_answer and excel_chunks:
                    direct_answer = self.excel_processor.get_direct_answer(question, excel_chunks)
                    if direct_answer and "No" not in direct_answer and "error" not in direct_answer.lower():
                        logger.info(f"✅ Direct answer found: {direct_answer}")
            
            # Step 3: Combine all chunks
            all_chunks = retrieved_chunks + self._convert_excel_chunks_to_rag_format(excel_chunks)
            
            if not all_chunks:
                logger.warning("No relevant chunks found for query")
                return EnhancedRAGResponse(
                    query=question,
                    answer="I couldn't find any relevant information to answer your question.",
                    sources=[],
                    excel_sources=[],
                    confidence_score=0.0,
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    metadata={"error": "No relevant chunks found"},
                    direct_answer=direct_answer
                )
            
            # Step 4: Generate response using retrieved context
            answer = self._generate_enhanced_answer(
                question, all_chunks, temperature, max_tokens, system_prompt, direct_answer
            )
            
            # Step 5: Calculate confidence score
            confidence_score = self._calculate_enhanced_confidence_score(
                question, all_chunks, answer, direct_answer
            )
            
            # Step 6: Prepare sources
            sources = self._prepare_sources(retrieved_chunks)
            excel_sources = self._prepare_excel_sources(excel_chunks)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ Enhanced RAG query completed in {processing_time:.2f}s")
            
            return EnhancedRAGResponse(
                query=question,
                answer=answer,
                sources=sources,
                excel_sources=excel_sources,
                confidence_score=confidence_score,
                processing_time=processing_time,
                metadata={
                    "chunks_retrieved": len(retrieved_chunks),
                    "excel_chunks": len(excel_chunks),
                    "model_used": self.ollama_client.current_model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "llamaindex_enabled": self.enable_llamaindex
                },
                direct_answer=direct_answer
            )
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced RAG pipeline: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return EnhancedRAGResponse(
                query=question,
                answer=f"I encountered an error while processing your question: {str(e)}",
                sources=[],
                excel_sources=[],
                confidence_score=0.0,
                processing_time=processing_time,
                metadata={"error": str(e)},
                direct_answer=direct_answer
            )
    
    def _is_excel_related_query(self, question: str) -> bool:
        """Check if the query is related to Excel/financial data."""
        excel_keywords = [
            'salary', 'earn', 'income', 'expense', 'budget', 'financial', 'amount', 'value',
            'cost', 'price', 'total', 'sum', 'largest', 'highest', 'maximum', 'biggest',
            'smallest', 'minimum', 'lowest', 'data', 'table', 'row', 'column', 'excel'
        ]
        
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in excel_keywords)
    
    def _process_excel_files(self, question: str) -> List[LlamaIndexExcelChunk]:
        """Process Excel files using LlamaIndex processor."""
        try:
            # Find Excel files in the uploads directory
            uploads_dir = Path("data/uploads")
            excel_files = list(uploads_dir.glob("*.xlsx")) + list(uploads_dir.glob("*.xls"))
            
            if not excel_files:
                logger.info("No Excel files found for processing")
                return []
            
            all_chunks = []
            
            for excel_file in excel_files:
                try:
                    logger.info(f"Processing Excel file: {excel_file.name}")
                    chunks = self.excel_processor.process_excel_file(excel_file)
                    all_chunks.extend(chunks)
                    
                except Exception as e:
                    logger.error(f"Error processing {excel_file.name}: {e}")
                    continue
            
            logger.info(f"Processed {len(all_chunks)} Excel chunks from {len(excel_files)} files")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error processing Excel files: {e}")
            return []
    
    def _convert_excel_chunks_to_rag_format(self, excel_chunks: List[LlamaIndexExcelChunk]) -> List[Dict[str, Any]]:
        """Convert Excel chunks to RAG format."""
        rag_chunks = []
        
        for chunk in excel_chunks:
            rag_chunk = {
                'text': chunk.text,
                'metadata': {
                    'file_name': chunk.metadata.get('source_file', 'Unknown'),
                    'file_path': chunk.metadata.get('source_file', 'Unknown'),
                    'chunk_type': chunk.chunk_type,
                    'sheet_name': chunk.sheet_name,
                    'row_range': chunk.row_range,
                    'column_range': chunk.column_range,
                    'excel_chunk': True
                },
                'distance': 0.5  # Default distance for Excel chunks
            }
            rag_chunks.append(rag_chunk)
        
        return rag_chunks
    
    def _retrieve_relevant_chunks(self, question: str, n_chunks: int) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for the question."""
        try:
            logger.info(f"Retrieving {n_chunks} relevant chunks for query")
            
            chunks = self.document_indexer.search_documents(question, n_results=n_chunks)
            
            if chunks:
                logger.info(f"✅ Retrieved {len(chunks)} relevant chunks")
                # Prioritize row chunks over column chunks for better data accuracy
                def chunk_priority(chunk):
                    text = chunk.get('text', '')
                    # Row chunks should have higher priority (lower score = higher priority)
                    if '[ROW]' in text or 'Row ' in text:
                        return chunk.get('distance', float('inf')) - 2.0  # Boost row chunks
                    elif '[COLUMN]' in text or 'Column ' in text:
                        return chunk.get('distance', float('inf')) + 1.0  # Lower column chunks
                    else:
                        return chunk.get('distance', float('inf'))
                
                chunks.sort(key=chunk_priority)
                return chunks
            else:
                logger.warning("No chunks retrieved from search")
                return []
                
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def _generate_enhanced_answer(self, question: str, chunks: List[Dict[str, Any]], 
                                temperature: float, max_tokens: int, 
                                system_prompt: str = None, direct_answer: str = None) -> str:
        """Generate an enhanced answer using the retrieved context."""
        try:
            # Prepare context from chunks
            context = self._prepare_enhanced_context_from_chunks(chunks)
            
            # Create enhanced prompt with direct answer if available
            prompt = self._create_enhanced_rag_prompt(question, context, direct_answer)
            
            # Use provided system prompt or default
            system_prompt = system_prompt or self.excel_system_prompt
            
            logger.info("Generating enhanced answer using LLM")
            
            # Generate response
            answer = self.ollama_client.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if answer:
                logger.info(f"✅ Generated enhanced answer ({len(answer)} characters)")
                return answer
            else:
                logger.error("Failed to generate answer from LLM")
                return "I was unable to generate an answer to your question."
                
        except Exception as e:
            logger.error(f"Error generating enhanced answer: {e}")
            return f"I encountered an error while generating an answer: {str(e)}"
    
    def _prepare_enhanced_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Prepare enhanced context string from retrieved chunks."""
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})
            file_name = metadata.get('file_name', 'Unknown')
            chunk_type = metadata.get('chunk_type', 'unknown')
            chunk_index = metadata.get('chunk_index', i)
            
            # Enhanced context formatting
            if metadata.get('excel_chunk', False):
                context_parts.append(f"--- Excel Data {i+1} ({chunk_type}) from {file_name} ---")
            else:
                context_parts.append(f"--- Document {i+1} from {file_name}, part {chunk_index} ---")
            
            context_parts.append(chunk_text)
            context_parts.append("")  # Empty line for separation
        
        return "\n".join(context_parts)
    
    def _create_enhanced_rag_prompt(self, question: str, context: str, direct_answer: str = None) -> str:
        """Create an enhanced prompt for RAG generation."""
        prompt = f"""Based on the following context, please answer the question below.

Context:
{context}

Question: {question}"""

        if direct_answer:
            prompt += f"""

Note: A direct answer was found: {direct_answer}
Please use this information to provide a comprehensive answer."""

        prompt += """

Please provide a comprehensive answer based only on the information in the context above. 
If the context doesn't contain enough information to fully answer the question, acknowledge 
what you can answer and what you cannot. Be precise with numbers and financial data."""
        
        return prompt
    
    def _calculate_enhanced_confidence_score(self, question: str, chunks: List[Dict[str, Any]], 
                                          answer: str, direct_answer: str = None) -> float:
        """Calculate an enhanced confidence score for the generated answer."""
        try:
            if not chunks:
                return 0.0
            
            # Base confidence from chunk relevance
            distances = [chunk.get('distance', 1.0) for chunk in chunks]
            avg_distance = sum(distances) / len(distances)
            
            # Convert distance to confidence (lower distance = higher confidence)
            if avg_distance <= 0.5:
                base_confidence = 0.9
            elif avg_distance <= 1.0:
                base_confidence = 0.7
            elif avg_distance <= 1.5:
                base_confidence = 0.5
            else:
                base_confidence = 0.3
            
            # Boost confidence if we have a direct answer
            if direct_answer and "No" not in direct_answer and "error" not in direct_answer.lower():
                base_confidence = min(base_confidence + 0.2, 1.0)
            
            # Adjust based on number of chunks
            chunk_factor = min(len(chunks) / 5.0, 1.0)
            
            # Adjust based on answer length
            answer_factor = min(len(answer) / 500.0, 1.0)
            
            # Calculate final confidence
            confidence = (base_confidence * 0.6 + chunk_factor * 0.2 + answer_factor * 0.2)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating enhanced confidence score: {e}")
            return 0.5
    
    def _prepare_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare source information for the response."""
        sources = []
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            if not metadata.get('excel_chunk', False):  # Only non-Excel chunks
                source = {
                    'file_name': metadata.get('file_name', 'Unknown'),
                    'file_path': metadata.get('file_path', 'Unknown'),
                    'chunk_index': metadata.get('chunk_index', 0),
                    'relevance_score': 1.0 - chunk.get('distance', 0.0),
                    'text_preview': chunk.get('text', '')[:200] + '...' if len(chunk.get('text', '')) > 200 else chunk.get('text', '')
                }
                sources.append(source)
        
        return sources
    
    def _prepare_excel_sources(self, excel_chunks: List[LlamaIndexExcelChunk]) -> List[Dict[str, Any]]:
        """Prepare Excel-specific source information."""
        sources = []
        
        for chunk in excel_chunks:
            source = {
                'file_name': chunk.metadata.get('source_file', 'Unknown'),
                'sheet_name': chunk.sheet_name,
                'chunk_type': chunk.chunk_type,
                'row_range': chunk.row_range,
                'column_range': chunk.column_range,
                'text_preview': chunk.text[:200] + '...' if len(chunk.text) > 200 else chunk.text,
                'semantic_score': chunk.semantic_score
            }
            sources.append(source)
        
        return sources
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the enhanced RAG pipeline."""
        info = {
            "ollama_health": self.ollama_client.health_check(),
            "available_models": self.ollama_client.list_models(),
            "default_model": self.ollama_client.current_model,
            "index_info": self.document_indexer.get_index_info(),
            "system_prompt": self.excel_system_prompt,
            "llamaindex_enabled": self.enable_llamaindex
        }
        
        if self.excel_processor:
            info["excel_processor"] = {
                "semantic_search_enabled": self.excel_processor.enable_semantic_search,
                "max_rows_per_chunk": self.excel_processor.max_rows_per_chunk
            }
        
        return info
