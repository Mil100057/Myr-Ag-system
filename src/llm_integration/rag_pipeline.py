"""
RAG (Retrieval-Augmented Generation) Pipeline for Myr-Ag.
"""
from typing import List, Dict, Any, Optional, Tuple, Generator
from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from .ollama_client import OllamaClient
from src.vector_db.document_indexer import DocumentIndexer
from src.vector_db.unified_vector_store import UnifiedVectorStore


@dataclass
class RAGQuery:
    """Represents a RAG query with context and results."""
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    generated_response: str
    metadata: Dict[str, Any]


@dataclass
class RAGResponse:
    """Represents a complete RAG response."""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    metadata: Dict[str, Any]


class RAGPipeline:
    """RAG pipeline combining retrieval and generation."""
    
    def __init__(self, document_indexer: DocumentIndexer, ollama_client: OllamaClient):
        """Initialize the RAG pipeline."""
        self.document_indexer = document_indexer
        self.ollama_client = ollama_client
        
        # Default system prompt for RAG
        self.default_system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
        Use only the information from the context to answer questions. If the context doesn't contain enough information 
        to answer the question, say so. Be accurate, helpful, and cite specific parts of the context when possible."""
        
        logger.info("RAG pipeline initialized")
    
    def query(self, question: str, n_chunks: int = 5, 
              temperature: float = 0.7, max_tokens: int = 2048,
              system_prompt: str = None) -> RAGResponse:
        """Execute a complete RAG query."""
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing RAG query: {question[:100]}...")
            
            # Step 1: Retrieve relevant documents
            retrieved_chunks = self._retrieve_relevant_chunks(question, n_chunks)
            
            if not retrieved_chunks:
                logger.warning("No relevant chunks found for query")
                return RAGResponse(
                    query=question,
                    answer="I couldn't find any relevant information to answer your question.",
                    sources=[],
                    confidence_score=0.0,
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    metadata={"error": "No relevant chunks found"}
                )
            
            # Step 2: Generate response using retrieved context
            answer = self._generate_answer_with_context(
                question, retrieved_chunks, temperature, max_tokens, system_prompt
            )
            
            # Step 3: Calculate confidence score
            confidence_score = self._calculate_confidence_score(question, retrieved_chunks, answer)
            
            # Step 4: Prepare sources
            sources = self._prepare_sources(retrieved_chunks)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ RAG query completed in {processing_time:.2f}s")
            
            return RAGResponse(
                query=question,
                answer=answer,
                sources=sources,
                confidence_score=confidence_score,
                processing_time=processing_time,
                metadata={
                    "chunks_retrieved": len(retrieved_chunks),
                    "model_used": self.ollama_client.current_model,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Error in RAG pipeline: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return RAGResponse(
                query=question,
                answer=f"I encountered an error while processing your question: {str(e)}",
                sources=[],
                confidence_score=0.0,
                processing_time=processing_time,
                metadata={"error": str(e)}
            )
    
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
                    if '[ROW]' in text:
                        return chunk.get('distance', float('inf')) - 2.0  # Boost row chunks
                    elif '[COLUMN]' in text:
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
    
    def _generate_answer_with_context(self, question: str, chunks: List[Dict[str, Any]], 
                                    temperature: float, max_tokens: int, 
                                    system_prompt: str = None) -> str:
        """Generate an answer using the retrieved context."""
        try:
            # Prepare context from chunks
            context = self._prepare_context_from_chunks(chunks)
            
            # Create prompt with context
            prompt = self._create_rag_prompt(question, context)
            
            # Use provided system prompt or default
            system_prompt = system_prompt or self.default_system_prompt
            
            logger.info("Generating answer using LLM")
            
            # Generate response
            answer = self.ollama_client.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if answer:
                logger.info(f"✅ Generated answer ({len(answer)} characters)")
                return answer
            else:
                logger.error("Failed to generate answer from LLM")
                return "I was unable to generate an answer to your question."
                
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I encountered an error while generating an answer: {str(e)}"
    
    def _prepare_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Prepare context string from retrieved chunks."""
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})
            file_name = metadata.get('file_name', 'Unknown')
            chunk_index = metadata.get('chunk_index', i)
            
            context_parts.append(f"--- Chunk {i+1} (from {file_name}, part {chunk_index}) ---")
            context_parts.append(chunk_text)
            context_parts.append("")  # Empty line for separation
        
        return "\n".join(context_parts)
    
    def _create_rag_prompt(self, question: str, context: str) -> str:
        """Create a prompt for RAG generation."""
        return f"""Based on the following context, please answer the question below.

Context:
{context}

Question: {question}

Please provide a comprehensive answer based only on the information in the context above. If the context doesn't contain enough information to fully answer the question, acknowledge what you can answer and what you cannot."""
    
    def _calculate_confidence_score(self, question: str, chunks: List[Dict[str, Any]], 
                                  answer: str) -> float:
        """Calculate a confidence score for the generated answer."""
        try:
            if not chunks:
                return 0.0
            
            # Calculate average relevance score (lower distance = higher relevance)
            distances = [chunk.get('distance', 1.0) for chunk in chunks]
            avg_distance = sum(distances) / len(distances)
            
            # Convert distance to confidence (lower distance = higher confidence)
            # Distance typically ranges from 0.0 to 2.0, where 0.0 is most similar
            if avg_distance <= 0.5:
                base_confidence = 0.9
            elif avg_distance <= 1.0:
                base_confidence = 0.7
            elif avg_distance <= 1.5:
                base_confidence = 0.5
            else:
                base_confidence = 0.3
            
            # Adjust based on number of chunks (more chunks = potentially more context)
            chunk_factor = min(len(chunks) / 5.0, 1.0)  # Normalize to 0-1
            
            # Adjust based on answer length (longer answers might indicate more comprehensive responses)
            answer_factor = min(len(answer) / 500.0, 1.0)  # Normalize to 0-1
            
            # Calculate final confidence
            confidence = (base_confidence * 0.6 + chunk_factor * 0.2 + answer_factor * 0.2)
            
            return min(confidence, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5  # Default confidence
    
    def _prepare_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare source information for the response."""
        sources = []
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            source = {
                'file_name': metadata.get('file_name', 'Unknown'),
                'file_path': metadata.get('file_path', 'Unknown'),
                'chunk_index': metadata.get('chunk_index', 0),
                'relevance_score': 1.0 - chunk.get('distance', 0.0),  # Convert distance to relevance
                'text_preview': chunk.get('text', '')[:200] + '...' if len(chunk.get('text', '')) > 200 else chunk.get('text', '')
            }
            sources.append(source)
        
        return sources
    
    def streaming_query(self, question: str, n_chunks: int = 5,
                       temperature: float = 0.7, max_tokens: int = 2048,
                       system_prompt: str = None) -> Tuple[List[Dict[str, Any]], Generator[str, None, None]]:
        """Execute a streaming RAG query."""
        try:
            # Step 1: Retrieve relevant chunks
            retrieved_chunks = self._retrieve_relevant_chunks(question, n_chunks)
            
            if not retrieved_chunks:
                logger.warning("No relevant chunks found for streaming query")
                return [], iter([])
            
            # Step 2: Prepare context and prompt
            context = self._prepare_context_from_chunks(retrieved_chunks)
            prompt = self._create_rag_prompt(question, context)
            system_prompt = system_prompt or self.default_system_prompt
            
            # Step 3: Generate streaming response
            streaming_response = self.ollama_client.generate_response_streaming(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return retrieved_chunks, streaming_response
            
        except Exception as e:
            logger.error(f"Error in streaming RAG query: {e}")
            return [], iter([])
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the RAG pipeline."""
        return {
            "ollama_health": self.ollama_client.health_check(),
            "available_models": self.ollama_client.list_models(),
            "default_model": self.ollama_client.current_model,
            "index_info": self.document_indexer.get_index_info(),
            "system_prompt": self.default_system_prompt
        }
