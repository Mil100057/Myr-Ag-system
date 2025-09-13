# Multi-Pipeline Architecture Design

## Overview

This document outlines the multi-pipeline RAG system implementation that provides specialized processing based on document type, topic domain, and interaction mode. The system is currently implemented and operational with domain-specific indexing and processing capabilities.

## Current Implementation Status

### ✅ Implemented Features

- **Domain-Specific Indexing**: Six specialized domains (General, Financial, Legal, Medical, Academic, Excel)
- **Document Processing**: Multi-format support with intelligent fallback mechanisms
- **Vector Storage**: LEANN for general documents, LlamaIndex for Excel files
- **Domain Indexing**: Automatic detection and manual assignment of documents to domains
- **Query Enhancement**: Domain-specific query enhancement for better results
- **System Management**: Comprehensive tools for managing domains and indexes
- **System Monitoring**: Real-time statistics and monitoring across all domain collections
- **Accurate Statistics**: Proper document and chunk counting across all domains
- **Collection Management**: Individual domain collection monitoring and management

### 🔧 Current Architecture

The system uses a simplified but effective approach:
- **Domain-Based Routing**: Documents are assigned to specific domains during upload
- **Isolated Indexes**: Each domain has its own vector index for optimal performance
- **Hybrid Processing**: LEANN for general documents, LlamaIndex for Excel files
- **User Control**: Manual domain assignment with automatic detection fallback

## Core Concepts

### Pipeline Specialization Dimensions

1. **Document Type**: PDF, Excel, Word, Technical Docs, Legal Docs, etc.
2. **Topic Domain**: Healthcare, Finance, Legal, Academic, Technical, etc.
3. **Interaction Mode**: Summary, Synthesis, Precise Question, Analysis

### Pipeline Selection Matrix

| Document Type | Topic Domain | Interaction Mode | Pipeline Name                 |
| ------------- | ------------ | ---------------- | ----------------------------- |
| PDF           | Legal        | Summary          | `legal_pdf_summary`         |
| PDF           | Legal        | Precise Question | `legal_pdf_precise`         |
| Excel         | Financial    | Synthesis        | `financial_excel_synthesis` |
| Word          | Technical    | Analysis         | `technical_word_analysis`   |
| PDF           | Academic     | Summary          | `academic_pdf_summary`      |

## Architecture Components

### 1. Pipeline Registry

```python
class PipelineRegistry:
    """Manages available pipelines and their configurations."""
  
    def __init__(self):
        self.pipelines = {}
        self.pipeline_configs = {}
  
    def register_pipeline(self, pipeline_id: str, pipeline_class: type, config: dict):
        """Register a new pipeline with its configuration."""
        pass
  
    def get_pipeline(self, document_type: str, topic: str, interaction_mode: str):
        """Get the appropriate pipeline based on criteria."""
        pass
  
    def list_available_pipelines(self):
        """List all available pipeline combinations."""
        pass
```

### 2. Base Pipeline Class

```python
class BasePipeline:
    """Base class for all specialized pipelines."""
  
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.document_processor = None
        self.vector_store = None
        self.llm_client = None
        self.chunking_strategy = None
        self.prompt_templates = None
  
    def process_document(self, file_path: Path) -> ProcessedDocument:
        """Process a document using pipeline-specific strategies."""
        pass
  
    def query(self, question: str, context: DocumentContext) -> PipelineResponse:
        """Execute a query using pipeline-specific logic."""
        pass
  
    def get_chunking_strategy(self) -> ChunkingStrategy:
        """Get the appropriate chunking strategy for this pipeline."""
        pass
  
    def get_prompt_template(self, interaction_mode: str) -> str:
        """Get the appropriate prompt template for this pipeline."""
        pass
```

### 3. Specialized Pipeline Classes

#### Financial Documents Pipeline

```python
class FinancialPipeline(BasePipeline):
    """Specialized pipeline for financial documents."""
  
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.chunking_strategy = FinancialChunkingStrategy()
        self.prompt_templates = FinancialPromptTemplates()
  
    def get_chunking_strategy(self):
        return FinancialChunkingStrategy(
            preserve_tables=True,
            extract_metadata=True,
            financial_entities=True
        )
  
    def get_prompt_template(self, interaction_mode: str):
        if interaction_mode == "summary":
            return "Summarize the financial document focusing on key metrics, trends, and financial health indicators..."
        elif interaction_mode == "synthesis":
            return "Analyze and synthesize financial data across multiple documents, identifying patterns and insights..."
        elif interaction_mode == "precise_question":
            return "Answer the specific financial question using exact data from the documents..."
```

#### Legal Documents Pipeline

```python
class LegalPipeline(BasePipeline):
    """Specialized pipeline for legal documents."""
  
    def get_chunking_strategy(self):
        return LegalChunkingStrategy(
            preserve_sections=True,
            extract_citations=True,
            legal_entities=True
        )
  
    def get_prompt_template(self, interaction_mode: str):
        if interaction_mode == "summary":
            return "Provide a legal summary focusing on key clauses, obligations, and legal implications..."
        elif interaction_mode == "synthesis":
            return "Compare and synthesize legal provisions across multiple documents..."
        elif interaction_mode == "precise_question":
            return "Answer the legal question with specific references to relevant clauses and provisions..."
```

#### Academic Papers Pipeline

```python
class AcademicPipeline(BasePipeline):
    """Specialized pipeline for academic papers."""
  
    def get_chunking_strategy(self):
        return AcademicChunkingStrategy(
            preserve_structure=True,
            extract_citations=True,
            methodology_sections=True
        )
  
    def get_prompt_template(self, interaction_mode: str):
        if interaction_mode == "summary":
            return "Summarize the academic paper focusing on research question, methodology, key findings, and conclusions..."
        elif interaction_mode == "synthesis":
            return "Synthesize findings across multiple academic papers, identifying common themes and differences..."
        elif interaction_mode == "precise_question":
            return "Answer the research question using specific data and findings from the academic literature..."
```

### 4. Chunking Strategies

#### Financial Chunking Strategy (example)

```python
class FinancialChunkingStrategy:
    """Specialized chunking for financial documents."""
  
    def __init__(self, preserve_tables=True, extract_metadata=True, financial_entities=True):
        self.preserve_tables = preserve_tables
        self.extract_metadata = extract_metadata
        self.financial_entities = financial_entities
  
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """Chunk financial document with specialized logic."""
        chunks = []
  
        # Extract financial tables
        if self.preserve_tables:
            table_chunks = self._extract_financial_tables(document)
            chunks.extend(table_chunks)
  
        # Extract financial entities
        if self.financial_entities:
            entity_chunks = self._extract_financial_entities(document)
            chunks.extend(entity_chunks)
  
        # Standard text chunking with financial context
        text_chunks = self._chunk_financial_text(document)
        chunks.extend(text_chunks)
  
        return chunks
  
    def _extract_financial_tables(self, document: Document) -> List[DocumentChunk]:
        """Extract and chunk financial tables."""
        pass
  
    def _extract_financial_entities(self, document: Document) -> List[DocumentChunk]:
        """Extract financial entities (amounts, dates, account numbers)."""
        pass
```

### 5. Prompt Templates

#### Financial Prompt Templates (example)

```python
class FinancialPromptTemplates:
    """Prompt templates for financial document processing."""
  
    SUMMARY_TEMPLATE = """
    You are a financial analyst. Summarize the following financial document:
  
    Document: {document_content}
  
    Focus on:
    - Key financial metrics and KPIs
    - Revenue and profit trends
    - Financial health indicators
    - Risk factors and opportunities
    - Executive summary of financial position
  
    Provide a structured summary with clear sections.
    """
  
    SYNTHESIS_TEMPLATE = """
    You are a senior financial analyst. Analyze and synthesize the following financial documents:
  
    Documents: {document_contents}
  
    Provide:
    - Comparative analysis across documents
    - Trend identification and patterns
    - Financial insights and recommendations
    - Risk assessment and opportunities
    - Executive summary with key findings
    """
  
    PRECISE_QUESTION_TEMPLATE = """
    You are a financial analyst. Answer the specific question using the provided financial documents:
  
    Question: {question}
    Documents: {document_contents}
  
    Requirements:
    - Use exact data and figures from the documents
    - Cite specific sections and page numbers
    - Provide confidence level for your answer
    - Include relevant context and assumptions
    """
```

### 6. Configuration System

```python
@dataclass
class PipelineConfig:
    """Configuration for a specialized pipeline."""
  
    # Pipeline identification
    pipeline_id: str
    document_types: List[str]
    topic_domains: List[str]
    interaction_modes: List[str]
  
    # Processing configuration
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
  
    # LLM configuration
    model_name: str
    temperature: float
    max_tokens: int
  
    # Pipeline-specific settings
    custom_settings: Dict[str, Any]
  
    # Prompt templates
    prompt_templates: Dict[str, str]
  
    # Metadata extraction
    extract_metadata: bool
    custom_metadata_fields: List[str]

# Example configurations
FINANCIAL_PDF_CONFIG = PipelineConfig(
    pipeline_id="financial_pdf",
    document_types=["pdf"],
    topic_domains=["finance", "accounting", "business"],
    interaction_modes=["summary", "synthesis", "precise_question"],
    chunking_strategy="financial",
    chunk_size=500,
    chunk_overlap=100,
    embedding_model="nomic-ai/nomic-embed-text-v2-moe",
    model_name="llama3.2:3b",
    temperature=0.3,  # Lower temperature for financial accuracy
    max_tokens=2048,
    custom_settings={
        "preserve_tables": True,
        "extract_financial_entities": True,
        "confidence_threshold": 0.8
    },
    prompt_templates={
        "summary": "financial_summary_template",
        "synthesis": "financial_synthesis_template",
        "precise_question": "financial_precise_template"
    },
    extract_metadata=True,
    custom_metadata_fields=["amount", "date", "account_number", "kpi"]
)
```
