"""
Specialized RAG pipelines for different document types and domains.

This module provides specialized processing pipelines for:
- Financial documents
- Legal documents  
- Medical documents
- Academic documents

Each pipeline includes domain-specific chunking, query enhancement, and prompt templates.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from ..document_processing.document_processor import DocumentProcessor
from ..vector_db.document_indexer import DocumentIndexer
from ..vector_db.domain_index_manager import DomainIndexManager
from .ollama_client import OllamaClient

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for a specialized pipeline."""
    
    # Pipeline identification
    pipeline_id: str
    domain: str
    supported_formats: List[str]
    
    # Processing configuration
    chunk_size: int
    chunk_overlap: int
    temperature: float
    max_tokens: int
    
    # Query enhancement settings
    enable_query_enhancement: bool
    enhancement_temperature: float
    
    # Domain-specific settings
    custom_settings: Dict[str, Any]


@dataclass
class EnhancedQuery:
    """Enhanced query with original and improved versions."""
    
    original_query: str
    enhanced_query: str
    enhancement_reason: str
    confidence: float


@dataclass
class PipelineResponse:
    """Response from a specialized pipeline."""
    
    query: str
    enhanced_query: Optional[EnhancedQuery]
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    pipeline_used: str
    domain: str
    metadata: Dict[str, Any]


class BaseSpecializedPipeline(ABC):
    """Base class for specialized pipelines."""
    
    def __init__(self, config: PipelineConfig, document_indexer: DocumentIndexer, 
                 ollama_client: OllamaClient):
        """Initialize the specialized pipeline."""
        self.config = config
        self.document_indexer = document_indexer
        self.ollama_client = ollama_client
        
        logger.info(f"Initialized {config.pipeline_id} pipeline for {config.domain} domain")
    
    @abstractmethod
    def get_domain_prompt_template(self, interaction_mode: str) -> str:
        """Get the domain-specific prompt template."""
        pass
    
    @abstractmethod
    def get_query_enhancement_prompt(self) -> str:
        """Get the query enhancement prompt for this domain."""
        pass
    
    @abstractmethod
    def get_chunking_strategy(self) -> Dict[str, Any]:
        """Get the chunking strategy for this domain."""
        pass
    
    def enhance_query(self, query: str) -> Optional[EnhancedQuery]:
        """Enhance the query using domain-specific knowledge."""
        if not self.config.enable_query_enhancement:
            return None
        
        try:
            enhancement_prompt = self.get_query_enhancement_prompt()
            full_prompt = f"{enhancement_prompt}\n\nOriginal Query: {query}"
            
            enhanced_query_text = self.ollama_client.generate_response(
                prompt=full_prompt,
                temperature=self.config.enhancement_temperature,
                max_tokens=512
            )
            
            # Extract the enhanced query and reason
            lines = enhanced_query_text.strip().split('\n')
            enhanced_query = lines[0] if lines else query
            reason = '\n'.join(lines[1:]) if len(lines) > 1 else "Query enhanced for better domain-specific results"
            
            return EnhancedQuery(
                original_query=query,
                enhanced_query=enhanced_query,
                enhancement_reason=reason,
                confidence=0.8  # Default confidence
            )
            
        except Exception as e:
            logger.warning(f"Query enhancement failed: {e}")
            return None
    
    def query(self, question: str, n_chunks: int = 5, 
              interaction_mode: str = "precise_question") -> PipelineResponse:
        """Execute a query using the specialized pipeline."""
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing {self.config.domain} query: {question[:100]}...")
            
            # Step 1: Enhance the query
            enhanced_query = self.enhance_query(question)
            query_to_use = enhanced_query.enhanced_query if enhanced_query else question
            
            # Step 2: Retrieve relevant documents
            retrieved_chunks = self._retrieve_relevant_chunks(query_to_use, n_chunks)
            
            if not retrieved_chunks:
                logger.warning("No relevant chunks found for query")
                return PipelineResponse(
                    query=question,
                    enhanced_query=enhanced_query,
                    answer="I couldn't find any relevant information to answer your question.",
                    sources=[],
                    confidence_score=0.0,
                    processing_time=(datetime.now() - start_time).total_seconds(),
                    pipeline_used=self.config.pipeline_id,
                    domain=self.config.domain,
                    metadata={"error": "No relevant chunks found"}
                )
            
            # Step 3: Generate response using domain-specific prompt
            answer = self._generate_domain_specific_answer(
                question, query_to_use, retrieved_chunks, interaction_mode
            )
            
            # Step 4: Calculate confidence score
            confidence_score = self._calculate_confidence_score(question, retrieved_chunks, answer)
            
            # Step 5: Prepare sources
            sources = self._prepare_sources(retrieved_chunks)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ {self.config.domain} query completed in {processing_time:.2f}s")
            
            return PipelineResponse(
                query=question,
                enhanced_query=enhanced_query,
                answer=answer,
                sources=sources,
                confidence_score=confidence_score,
                processing_time=processing_time,
                pipeline_used=self.config.pipeline_id,
                domain=self.config.domain,
                metadata={
                    "chunks_retrieved": len(retrieved_chunks),
                    "model_used": self.ollama_client.current_model,
                    "temperature": self.config.temperature,
                    "interaction_mode": interaction_mode
                }
            )
            
        except Exception as e:
            logger.error(f"Error in {self.config.domain} pipeline: {e}")
            return PipelineResponse(
                query=question,
                enhanced_query=enhanced_query,
                answer=f"Error processing query: {str(e)}",
                sources=[],
                confidence_score=0.0,
                processing_time=(datetime.now() - start_time).total_seconds(),
                pipeline_used=self.config.pipeline_id,
                domain=self.config.domain,
                metadata={"error": str(e)}
            )
    
    def _retrieve_relevant_chunks(self, query: str, n_chunks: int) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks using the document indexer."""
        try:
            # Use the document indexer to search for relevant chunks
            search_results = self.document_indexer.search_documents(query, n_chunks)
            return search_results
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def _generate_domain_specific_answer(self, original_question: str, enhanced_query: str, 
                                       retrieved_chunks: List[Dict[str, Any]], 
                                       interaction_mode: str) -> str:
        """Generate answer using domain-specific prompt template."""
        try:
            # Prepare context from retrieved chunks
            context = "\n\n".join([chunk.get('text', '') for chunk in retrieved_chunks])
            
            # Get domain-specific prompt template
            prompt_template = self.get_domain_prompt_template(interaction_mode)
            
            # Format the prompt
            prompt = prompt_template.format(
                original_question=original_question,
                enhanced_query=enhanced_query,
                context=context,
                domain=self.config.domain
            )
            
            # Generate response
            response = self.ollama_client.generate_response(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating domain-specific answer: {e}")
            return f"Error generating response: {str(e)}"
    
    def _calculate_confidence_score(self, question: str, chunks: List[Dict[str, Any]], 
                                  answer: str) -> float:
        """Calculate confidence score for the response."""
        try:
            # Simple confidence calculation based on chunk relevance
            if not chunks:
                return 0.0
            
            # Base confidence on number of relevant chunks
            base_confidence = min(len(chunks) / 5.0, 1.0)
            
            # Adjust based on answer length and content
            if len(answer) < 50:
                base_confidence *= 0.7
            elif "I couldn't find" in answer or "No relevant" in answer:
                base_confidence *= 0.3
            
            return min(base_confidence, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.5
    
    def _prepare_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare source information from chunks."""
        sources = []
        for i, chunk in enumerate(chunks):
            metadata = chunk.get('metadata', {})
            source = {
                "chunk_id": i,
                "file_name": metadata.get('file_name', 'Unknown'),
                "file_path": metadata.get('file_path', 'Unknown'),
                "chunk_index": metadata.get('chunk_index', i),
                "text": chunk.get('text', '')[:200] + "..." if len(chunk.get('text', '')) > 200 else chunk.get('text', ''),
                "text_preview": chunk.get('text', '')[:200] + "..." if len(chunk.get('text', '')) > 200 else chunk.get('text', ''),
                "metadata": metadata,
                "relevance_score": chunk.get('score', 0.0)
            }
            sources.append(source)
        return sources


class FinancialPipeline(BaseSpecializedPipeline):
    """Specialized pipeline for financial documents."""
    
    def __init__(self, document_indexer: DocumentIndexer, ollama_client: OllamaClient):
        config = PipelineConfig(
            pipeline_id="financial",
            domain="financial",
            supported_formats=["pdf", "docx", "txt", "md", "html", "xhtml", "csv"],
            chunk_size=500,
            chunk_overlap=100,
            temperature=0.3,  # Lower temperature for financial accuracy
            max_tokens=2048,
            enable_query_enhancement=True,
            enhancement_temperature=0.4,
            custom_settings={
                "extract_financial_entities": True,
                "preserve_tables": True,
                "confidence_threshold": 0.8
            }
        )
        super().__init__(config, document_indexer, ollama_client)
    
    def get_domain_prompt_template(self, interaction_mode: str) -> str:
        """Get financial domain prompt template."""
        if interaction_mode == "summary":
            return """You are a senior financial analyst. Summarize the following financial document:

Original Question: {original_question}
Enhanced Query: {enhanced_query}

Document Content:
{context}

Focus on:
- Key financial metrics and KPIs
- Revenue and profit trends  
- Financial health indicators
- Risk factors and opportunities
- Executive summary of financial position

Provide a structured summary with clear sections and specific data points."""
        
        elif interaction_mode == "synthesis":
            return """You are a senior financial analyst. Analyze and synthesize the following financial documents:

Original Question: {original_question}
Enhanced Query: {enhanced_query}

Document Content:
{context}

Provide:
- Comparative analysis across documents
- Trend identification and patterns
- Financial insights and recommendations
- Risk assessment and opportunities
- Executive summary with key findings

Use specific financial data and metrics in your analysis."""
        
        else:  # precise_question
            return """You are a financial analyst. Answer the specific question using the provided financial documents:

Original Question: {original_question}
Enhanced Query: {enhanced_query}

Document Content:
{context}

Requirements:
- Use exact data and figures from the documents
- Cite specific sections and page numbers when possible
- Provide confidence level for your answer
- Include relevant context and assumptions
- Focus on financial accuracy and precision"""
    
    def get_chunking_strategy(self) -> Dict[str, Any]:
        """Get the chunking strategy for financial documents."""
        return {
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "preserve_tables": True,
            "extract_financial_entities": True,
            "financial_metadata": True
        }
    
    def get_query_enhancement_prompt(self) -> str:
        """Get query enhancement prompt for financial domain."""
        return """You are a financial analyst. Enhance the following query to be more specific and effective for financial document analysis.

Consider:
- Financial terminology and concepts
- Specific metrics (revenue, profit, ROI, etc.)
- Time periods and comparisons
- Financial ratios and KPIs
- Regulatory and compliance aspects

Provide the enhanced query on the first line, followed by a brief explanation of the enhancements made."""


class LegalPipeline(BaseSpecializedPipeline):
    """Specialized pipeline for legal documents."""
    
    def __init__(self, document_indexer: DocumentIndexer, ollama_client: OllamaClient):
        config = PipelineConfig(
            pipeline_id="legal",
            domain="legal",
            supported_formats=["pdf", "docx", "txt", "md", "html", "xhtml"],
            chunk_size=600,
            chunk_overlap=150,
            temperature=0.2,  # Very low temperature for legal precision
            max_tokens=2048,
            enable_query_enhancement=True,
            enhancement_temperature=0.3,
            custom_settings={
                "extract_legal_entities": True,
                "preserve_sections": True,
                "confidence_threshold": 0.9
            }
        )
        super().__init__(config, document_indexer, ollama_client)
    
    def get_domain_prompt_template(self, interaction_mode: str) -> str:
        """Get legal domain prompt template."""
        if interaction_mode == "summary":
            return """You are a legal expert. Summarize the following legal document:

Original Question: {original_question}
Enhanced Query: {enhanced_query}

Document Content:
{context}

Focus on:
- Key legal provisions and clauses
- Rights and obligations
- Legal implications and consequences
- Important deadlines and requirements
- Risk factors and compliance issues

Provide a structured legal summary with clear sections."""
        
        elif interaction_mode == "synthesis":
            return """You are a legal expert. Analyze and synthesize the following legal documents:

Original Question: {original_question}
Enhanced Query: {enhanced_query}

Document Content:
{context}

Provide:
- Comparative analysis of legal provisions
- Identification of conflicts or inconsistencies
- Legal precedents and case law references
- Risk assessment and compliance requirements
- Executive summary of legal implications

Use precise legal terminology and cite specific clauses."""
        
        else:  # precise_question
            return """You are a legal expert. Answer the specific legal question using the provided documents:

Original Question: {original_question}
Enhanced Query: {enhanced_query}

Document Content:
{context}

Requirements:
- Use exact legal language and citations
- Reference specific clauses, articles, or sections
- Provide confidence level for your answer
- Include relevant legal context and precedents
- Focus on legal accuracy and precision"""
    
    def get_chunking_strategy(self) -> Dict[str, Any]:
        """Get the chunking strategy for legal documents."""
        return {
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "preserve_sections": True,
            "extract_legal_entities": True,
            "legal_metadata": True
        }
    
    def get_query_enhancement_prompt(self) -> str:
        """Get query enhancement prompt for legal domain."""
        return """You are a legal expert. Enhance the following query to be more specific and effective for legal document analysis.

Consider:
- Legal terminology and concepts
- Specific legal provisions and clauses
- Jurisdiction and applicable law
- Legal precedents and case law
- Compliance and regulatory aspects

Provide the enhanced query on the first line, followed by a brief explanation of the enhancements made."""


class MedicalPipeline(BaseSpecializedPipeline):
    """Specialized pipeline for medical documents."""
    
    def __init__(self, document_indexer: DocumentIndexer, ollama_client: OllamaClient):
        config = PipelineConfig(
            pipeline_id="medical",
            domain="medical",
            supported_formats=["pdf", "docx", "txt", "md", "html", "xhtml"],
            chunk_size=400,
            chunk_overlap=100,
            temperature=0.1,  # Very low temperature for medical accuracy
            max_tokens=2048,
            enable_query_enhancement=True,
            enhancement_temperature=0.2,
            custom_settings={
                "extract_medical_entities": True,
                "preserve_medical_structure": True,
                "confidence_threshold": 0.95
            }
        )
        super().__init__(config, document_indexer, ollama_client)
    
    def get_domain_prompt_template(self, interaction_mode: str) -> str:
        """Get medical domain prompt template."""
        if interaction_mode == "summary":
            return """You are a medical professional. Summarize the following medical document:

Original Question: {original_question}
Enhanced Query: {enhanced_query}

Document Content:
{context}

Focus on:
- Key medical findings and diagnoses
- Treatment plans and recommendations
- Patient symptoms and history
- Medical procedures and outcomes
- Important medical warnings or precautions

Provide a structured medical summary with clear sections."""
        
        elif interaction_mode == "synthesis":
            return """You are a medical professional. Analyze and synthesize the following medical documents:

Original Question: {original_question}
Enhanced Query: {enhanced_query}

Document Content:
{context}

Provide:
- Comparative analysis of medical findings
- Treatment effectiveness and outcomes
- Medical trends and patterns
- Risk factors and complications
- Clinical recommendations

Use precise medical terminology and cite specific findings."""
        
        else:  # precise_question
            return """You are a medical professional. Answer the specific medical question using the provided documents:

Original Question: {original_question}
Enhanced Query: {enhanced_query}

Document Content:
{context}

Requirements:
- Use exact medical terminology and citations
- Reference specific findings, tests, or procedures
- Provide confidence level for your answer
- Include relevant medical context and considerations
- Focus on medical accuracy and patient safety"""
    
    def get_chunking_strategy(self) -> Dict[str, Any]:
        """Get the chunking strategy for medical documents."""
        return {
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "preserve_medical_structure": True,
            "extract_medical_entities": True,
            "medical_metadata": True
        }
    
    def get_query_enhancement_prompt(self) -> str:
        """Get query enhancement prompt for medical domain."""
        return """You are a medical professional. Enhance the following query to be more specific and effective for medical document analysis.

Consider:
- Medical terminology and concepts
- Specific medical conditions and symptoms
- Treatment protocols and procedures
- Medical tests and diagnostics
- Patient safety and clinical considerations

Provide the enhanced query on the first line, followed by a brief explanation of the enhancements made."""


class AcademicPipeline(BaseSpecializedPipeline):
    """Specialized pipeline for academic documents."""
    
    def __init__(self, document_indexer: DocumentIndexer, ollama_client: OllamaClient):
        config = PipelineConfig(
            pipeline_id="academic",
            domain="academic",
            supported_formats=["pdf", "docx", "txt", "md", "html", "xhtml"],
            chunk_size=450,
            chunk_overlap=100,
            temperature=0.4,  # Moderate temperature for academic analysis
            max_tokens=2048,
            enable_query_enhancement=True,
            enhancement_temperature=0.5,
            custom_settings={
                "extract_academic_entities": True,
                "preserve_citations": True,
                "confidence_threshold": 0.85
            }
        )
        super().__init__(config, document_indexer, ollama_client)
    
    def get_domain_prompt_template(self, interaction_mode: str) -> str:
        """Get academic domain prompt template."""
        if interaction_mode == "summary":
            return """You are an academic researcher. Summarize the following academic document:

Original Question: {original_question}
Enhanced Query: {enhanced_query}

Document Content:
{context}

Focus on:
- Research question and objectives
- Methodology and approach
- Key findings and results
- Conclusions and implications
- Limitations and future research directions

Provide a structured academic summary with clear sections."""
        
        elif interaction_mode == "synthesis":
            return """You are an academic researcher. Analyze and synthesize the following academic documents:

Original Question: {original_question}
Enhanced Query: {enhanced_query}

Document Content:
{context}

Provide:
- Comparative analysis of research findings
- Identification of research gaps and opportunities
- Methodological comparisons and critiques
- Theoretical implications and contributions
- Executive summary of academic insights

Use precise academic terminology and cite specific studies."""
        
        else:  # precise_question
            return """You are an academic researcher. Answer the specific research question using the provided documents:

Original Question: {original_question}
Enhanced Query: {enhanced_query}

Document Content:
{context}

Requirements:
- Use exact academic terminology and citations
- Reference specific studies, authors, and findings
- Provide confidence level for your answer
- Include relevant academic context and literature
- Focus on research accuracy and scholarly rigor"""
    
    def get_chunking_strategy(self) -> Dict[str, Any]:
        """Get the chunking strategy for academic documents."""
        return {
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "preserve_citations": True,
            "extract_academic_entities": True,
            "academic_metadata": True
        }
    
    def get_query_enhancement_prompt(self) -> str:
        """Get query enhancement prompt for academic domain."""
        return """You are an academic researcher. Enhance the following query to be more specific and effective for academic document analysis.

Consider:
- Academic terminology and concepts
- Specific research methodologies
- Theoretical frameworks and models
- Research gaps and opportunities
- Scholarly rigor and evidence-based analysis

Provide the enhanced query on the first line, followed by a brief explanation of the enhancements made."""


class PipelineManager:
    """Manages specialized pipelines and automatic pipeline selection."""
    
    def __init__(self, ollama_client: OllamaClient, base_data_dir: Path = None):
        """Initialize the pipeline manager."""
        self.ollama_client = ollama_client
        
        # Initialize domain index manager
        self.domain_manager = DomainIndexManager(base_data_dir)
        self.domain_manager.initialize_domain_indexes()
        
        # Initialize specialized pipelines with domain-specific indexers
        self.pipelines = {}
        for domain in ["financial", "legal", "medical", "academic"]:
            domain_indexer = self.domain_manager.get_domain_indexer(domain)
            if domain_indexer:
                if domain == "financial":
                    self.pipelines[domain] = FinancialPipeline(domain_indexer, ollama_client)
                elif domain == "legal":
                    self.pipelines[domain] = LegalPipeline(domain_indexer, ollama_client)
                elif domain == "medical":
                    self.pipelines[domain] = MedicalPipeline(domain_indexer, ollama_client)
                elif domain == "academic":
                    self.pipelines[domain] = AcademicPipeline(domain_indexer, ollama_client)
        
        logger.info(f"Pipeline manager initialized with {len(self.pipelines)} specialized pipelines")
    
    def get_pipeline(self, domain: str) -> Optional[BaseSpecializedPipeline]:
        """Get a specific pipeline by domain."""
        return self.pipelines.get(domain.lower())
    
    def list_available_pipelines(self) -> List[str]:
        """List all available pipeline domains."""
        return list(self.pipelines.keys())
    
    
    def query_with_specific_pipeline(self, question: str, domain: str, n_chunks: int = 5, 
                                   interaction_mode: str = "precise_question") -> PipelineResponse:
        """Query with a specific pipeline."""
        pipeline = self.get_pipeline(domain)
        
        if pipeline:
            return pipeline.query(question, n_chunks, interaction_mode)
        else:
            raise ValueError(f"Pipeline '{domain}' not found. Available: {self.list_available_pipelines()}")
    
    def index_document(self, file_path: Path, force_domain: str = None) -> Tuple[bool, str]:
        """Index a document in the appropriate domain index."""
        return self.domain_manager.index_document(file_path, force_domain)
    
    def index_directory(self, directory_path: Path, force_domain: str = None) -> Dict[str, int]:
        """Index all documents in a directory, routing to appropriate domains."""
        return self.domain_manager.index_directory(directory_path, force_domain)
    
    def detect_document_domain(self, file_path: Path, content: str = None) -> str:
        """Detect the appropriate domain for a document."""
        return self.domain_manager.detect_document_domain(file_path, content)
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get statistics for all domain indexes."""
        return self.domain_manager.get_domain_statistics()
    
    def reset_domain_index(self, domain: str) -> bool:
        """Reset a specific domain index."""
        return self.domain_manager.reset_domain_index(domain)
    
    def rebuild_domain_index(self, domain: str) -> bool:
        """Rebuild a specific domain index."""
        return self.domain_manager.rebuild_domain_index(domain)
    
    def reset_all_domain_indexes(self) -> Dict[str, bool]:
        """Reset all domain indexes."""
        return self.domain_manager.reset_all_domain_indexes()
    
    def rebuild_all_domain_indexes(self) -> Dict[str, bool]:
        """Rebuild all domain indexes."""
        return self.domain_manager.rebuild_all_domain_indexes()
