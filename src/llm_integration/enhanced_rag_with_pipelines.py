"""
Enhanced RAG Pipeline with Specialized Domain Pipelines.

This module provides an enhanced RAG pipeline that automatically selects
and uses specialized pipelines based on document type and query content.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .rag_pipeline import RAGPipeline, RAGResponse
from .specialized_pipelines import PipelineManager, PipelineResponse
from .ollama_client import OllamaClient
from ..vector_db.document_indexer import DocumentIndexer

logger = logging.getLogger(__name__)


@dataclass
class EnhancedRAGResponse:
    """Enhanced RAG response with pipeline information."""
    
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    pipeline_used: str
    domain: str
    enhanced_query: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None


class EnhancedRAGWithPipelines:
    """Enhanced RAG pipeline with specialized domain pipelines."""
    
    def __init__(self, document_indexer: DocumentIndexer, ollama_client: OllamaClient, base_data_dir: Path = None):
        """Initialize the enhanced RAG pipeline."""
        self.document_indexer = document_indexer
        self.ollama_client = ollama_client
        
        # Initialize base RAG pipeline
        self.base_rag_pipeline = RAGPipeline(document_indexer, ollama_client)
        
        # Initialize specialized pipeline manager with domain-specific indexes
        self.pipeline_manager = PipelineManager(ollama_client, base_data_dir)
        
        # Pipeline selection settings
        self.auto_detect_pipeline = False
        self.fallback_to_base = True
        
        logger.info("Enhanced RAG pipeline with specialized pipelines initialized")
    
    def query(self, question: str, n_chunks: int = 5, 
              interaction_mode: str = "precise_question",
              force_pipeline: Optional[str] = None) -> EnhancedRAGResponse:
        """Execute a query using the enhanced RAG pipeline."""
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing enhanced RAG query: {question[:100]}...")
            
            # Determine which pipeline to use
            if force_pipeline:
                # Use specific pipeline
                pipeline_domain = force_pipeline
                logger.info(f"Using forced pipeline: {pipeline_domain}")
            else:
                # Use base pipeline
                pipeline_domain = "base"
                logger.info("Using base RAG pipeline")
            
            # Execute query with selected pipeline
            if pipeline_domain == "base":
                response = self._query_with_base_pipeline(question, n_chunks)
            else:
                response = self._query_with_specialized_pipeline(
                    question, pipeline_domain, n_chunks, interaction_mode
                )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Convert to enhanced response format
            enhanced_response = self._convert_to_enhanced_response(
                response, pipeline_domain, processing_time
            )
            
            logger.info(f"✅ Enhanced RAG query completed in {processing_time:.2f}s using {pipeline_domain} pipeline")
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced RAG pipeline: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return EnhancedRAGResponse(
                query=question,
                answer=f"I encountered an error while processing your question: {str(e)}",
                sources=[],
                confidence_score=0.0,
                processing_time=processing_time,
                pipeline_used="error",
                domain="error",
                metadata={"error": str(e)}
            )
    
    def _query_with_base_pipeline(self, question: str, n_chunks: int) -> RAGResponse:
        """Query using the base RAG pipeline."""
        return self.base_rag_pipeline.query(question, n_chunks)
    
    def _query_with_specialized_pipeline(self, question: str, domain: str, 
                                       n_chunks: int, interaction_mode: str) -> PipelineResponse:
        """Query using a specialized pipeline."""
        try:
            return self.pipeline_manager.query_with_specific_pipeline(
                question, domain, n_chunks, interaction_mode
            )
        except Exception as e:
            logger.warning(f"Error with specialized pipeline {domain}: {e}")
            
            if self.fallback_to_base:
                logger.info("Falling back to base RAG pipeline")
                base_response = self._query_with_base_pipeline(question, n_chunks)
                # Convert base response to pipeline response format
                return self._convert_base_to_pipeline_response(base_response, domain)
            else:
                raise
    
    def _convert_base_to_pipeline_response(self, base_response: RAGResponse, domain: str) -> PipelineResponse:
        """Convert base RAG response to pipeline response format."""
        from .specialized_pipelines import PipelineResponse as PR
        
        return PR(
            query=base_response.query,
            enhanced_query=None,
            answer=base_response.answer,
            sources=base_response.sources,
            confidence_score=base_response.confidence_score,
            processing_time=base_response.processing_time,
            pipeline_used=f"base_{domain}",
            domain=domain,
            metadata=base_response.metadata
        )
    
    def _convert_to_enhanced_response(self, response, pipeline_domain: str, 
                                    processing_time: float) -> EnhancedRAGResponse:
        """Convert pipeline response to enhanced RAG response format."""
        if hasattr(response, 'enhanced_query'):
            # Specialized pipeline response
            enhanced_query = {
                "original": response.enhanced_query.original_query if response.enhanced_query else None,
                "enhanced": response.enhanced_query.enhanced_query if response.enhanced_query else None,
                "reason": response.enhanced_query.enhancement_reason if response.enhanced_query else None,
                "confidence": response.enhanced_query.confidence if response.enhanced_query else None
            } if response.enhanced_query else None
        else:
            # Base RAG response
            enhanced_query = None
        
        return EnhancedRAGResponse(
            query=response.query,
            answer=response.answer,
            sources=response.sources,
            confidence_score=response.confidence_score,
            processing_time=processing_time,
            pipeline_used=response.pipeline_used if hasattr(response, 'pipeline_used') else pipeline_domain,
            domain=response.domain if hasattr(response, 'domain') else pipeline_domain,
            enhanced_query=enhanced_query,
            metadata=response.metadata if hasattr(response, 'metadata') else {}
        )
    
    
    def query_with_specific_pipeline(self, question: str, domain: str, n_chunks: int = 5, 
                                   interaction_mode: str = "precise_question") -> EnhancedRAGResponse:
        """Query with a specific pipeline."""
        return self.query(question, n_chunks, interaction_mode, force_pipeline=domain)
    
    def list_available_pipelines(self) -> List[str]:
        """List all available pipelines."""
        pipelines = ["base"] + self.pipeline_manager.list_available_pipelines()
        return pipelines
    
    def get_pipeline_info(self, domain: str = None) -> Dict[str, Any]:
        """Get information about pipelines."""
        if domain and domain != "base":
            pipeline = self.pipeline_manager.get_pipeline(domain)
            if pipeline:
                return {
                    "pipeline_id": pipeline.config.pipeline_id,
                    "domain": pipeline.config.domain,
                    "supported_formats": pipeline.config.supported_formats,
                    "chunk_size": pipeline.config.chunk_size,
                    "temperature": pipeline.config.temperature,
                    "enable_query_enhancement": pipeline.config.enable_query_enhancement
                }
            else:
                return {"error": f"Pipeline '{domain}' not found"}
        else:
            return {
                "base_pipeline": self.base_rag_pipeline.get_pipeline_info(),
                "specialized_pipelines": {
                    domain: self.get_pipeline_info(domain) 
                    for domain in self.pipeline_manager.list_available_pipelines()
                },
                "fallback_enabled": self.fallback_to_base
            }
    
    
    def set_fallback(self, enabled: bool):
        """Enable or disable fallback to base pipeline."""
        self.fallback_to_base = enabled
        logger.info(f"Fallback to base pipeline {'enabled' if enabled else 'disabled'}")
    
    def test_pipeline_detection(self, test_queries: List[str]) -> Dict[str, str]:
        """Test pipeline detection with sample queries."""
        results = {}
        
        for query in test_queries:
            detected_pipeline = self.pipeline_manager.auto_detect_pipeline(query)
            results[query] = detected_pipeline
        
        return results
    
    def index_document(self, file_path: Path, force_domain: str = None) -> Tuple[bool, str]:
        """Index a document in the appropriate domain index."""
        return self.pipeline_manager.index_document(file_path, force_domain)
    
    def index_directory(self, directory_path: Path, force_domain: str = None) -> Dict[str, int]:
        """Index all documents in a directory, routing to appropriate domains."""
        return self.pipeline_manager.index_directory(directory_path, force_domain)
    
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get statistics for all domain indexes."""
        return self.pipeline_manager.get_domain_statistics()
    
    def reset_domain_index(self, domain: str) -> bool:
        """Reset a specific domain index."""
        return self.pipeline_manager.reset_domain_index(domain)
    
    def rebuild_domain_index(self, domain: str) -> bool:
        """Rebuild a specific domain index."""
        return self.pipeline_manager.rebuild_domain_index(domain)
    
    def reset_all_domain_indexes(self) -> Dict[str, bool]:
        """Reset all domain indexes."""
        return self.pipeline_manager.reset_all_domain_indexes()
    
    def rebuild_all_domain_indexes(self) -> Dict[str, bool]:
        """Rebuild all domain indexes."""
        return self.pipeline_manager.rebuild_all_domain_indexes()
