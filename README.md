# Myr-Ag RAG System

A powerful RAG (Retrieval-Augmented Generation) system for document processing and intelligent querying, built with FastAPI, Gradio, LEANN vector database, and LlamaIndex for advanced Excel processing.

## Features

- **Multi-Pipeline Architecture**: Advanced system with specialized pipelines for different document types and domains
  - **Standard Method**: LEANN vector store for PDFs, Word docs, text files, images
  - **Excel Specific Method**: LlamaIndex for Excel files with advanced spreadsheet processing (⚠️ EXPERIMENTAL)
  - **Specialized Pipelines**: Domain-specific processing with user-controlled domain selection and query enhancement
    - **Financial Pipeline**: Optimized for financial documents, reports, and data analysis
    - **Legal Pipeline**: Specialized for legal documents, contracts, and compliance
    - **Medical Pipeline**: Designed for medical documents, research, and clinical data
    - **Academic Pipeline**: Tailored for academic papers, research, and scholarly content
    - **Domain-Specific Indexes**: Each pipeline has its own isolated vector index for optimal performance
    - **User-Controlled Domain Selection**: Manual domain assignment for precise document routing
- **Hybrid Document Chunking**: Advanced chunking strategy combining paragraph-based splitting with enhanced sentence splitting
- **Robust Document Processing**: Multi-format support with intelligent fallback mechanisms
  - **Docling**: Primary processor for PDF, DOCX, XLSX, PPTX, HTML, MD, images, and more
  - **OCR Support**: Automatic text extraction from scanned PDFs and images (PNG, JPEG, TIFF, BMP, WEBP)
  - **Enhanced Excel Processing**: Intelligent XLSX processing with smart chunking
    - **Row-based chunking**: Each row becomes a searchable chunk
    - **Column-aware processing**: Automatic detection of amounts, names, dates
    - **Natural language descriptions**: "Employee: John Doe | Salary: 50000"
    - **Multi-sheet support**: Processes all sheets in Excel files
    - **Pattern recognition**: Works with any Excel structure (named/unnamed columns)
  - **pypdf Fallback**: Automatic fallback for PDFs that Docling cannot process (e.g., web-printed PDFs)
  - **Direct Text Reading**: Optimized for simple text formats (TXT)
- **FastAPI Backend**: Robust REST API with optimized timeouts for document processing
- **Gradio Frontend**: Beautiful and intuitive web interface with modern theming and multi-language support (EN/FR/ES/DE)
- **LEANN Vector Database**: Ultra-efficient vector storage with 97% space savings and fast retrieval
- **LlamaIndex Integration**: Advanced Excel processing with persistent indexing (⚠️ EXPERIMENTAL)
- **LLM Integration**: Ollama support for local LLM inference with dynamic model selection
- **Query Enhancement**: Intelligent query enhancement for better domain-specific results
- **Smart Processing**: Automatic document indexing and chunking with progress tracking
- **System Management**: Comprehensive tools for managing documents, indexes, and system maintenance
  - **Document Management**: Upload, process, view, and delete documents with domain assignment
  - **Domain Indexing**: Index documents by domain with automatic detection or manual assignment
  - **Index Management**: Reset, rebuild, and clear indexes for both LEANN and LlamaIndex
    - **Individual Domain Control**: Manage each domain (General, Financial, Legal, Medical, Academic, Excel) separately
    - **Bulk Operations**: Reset/rebuild all indexes with a single operation
    - **Domain-Specific Operations**: Targeted operations for specific domains
  - **Vector Store Monitoring**: Real-time status monitoring for both vector stores
  - **System Maintenance**: Reset, rebuild, and clear operations with data preservation
    - **Granular Control**: Clear specific domains or all data
    - **Safety Features**: Confirmation required for destructive operations
- **Production Ready**: Comprehensive management scripts and tools

## Specialized Pipelines

The system includes specialized pipelines with user-controlled domain selection for precise document routing and optimal processing:

### Pipeline Types

#### 🏦 Financial Pipeline
- **Optimized for**: Financial reports, statements, budgets, invoices
- **Features**: 
  - Focus on financial metrics and KPIs
  - Revenue and profit trend analysis
  - Financial health indicators
  - Risk assessment and opportunities
- **Query Enhancement**: Automatically enhances queries with financial terminology

#### ⚖️ Legal Pipeline
- **Optimized for**: Legal documents, contracts, agreements, compliance
- **Features**:
  - Legal provision and clause analysis
  - Rights and obligations identification
  - Legal implications and consequences
  - Compliance and regulatory focus
- **Query Enhancement**: Enhances queries with legal terminology and concepts

#### 🏥 Medical Pipeline
- **Optimized for**: Medical documents, research papers, clinical data
- **Features**:
  - Medical findings and diagnosis focus
  - Treatment plan analysis
  - Patient safety considerations
  - Clinical recommendations
- **Query Enhancement**: Uses medical terminology and clinical context

#### 🎓 Academic Pipeline
- **Optimized for**: Research papers, academic articles, scholarly content
- **Features**:
  - Research methodology analysis
  - Key findings and results extraction
  - Academic citation and reference tracking
  - Theoretical implications
- **Query Enhancement**: Enhances queries with academic terminology and research concepts

### Auto-Detection

The system automatically detects the appropriate pipeline based on:
- Query content and keywords
- Document metadata and content
- Domain-specific terminology
- Context clues in the question

### Query Enhancement

Each pipeline includes intelligent query enhancement that:
- Rephrases queries for better domain-specific results
- Adds relevant terminology and concepts
- Improves search accuracy and relevance
- Provides explanations for enhancements

## Quick Start

### Prerequisites

- Python 3.11+
- Ollama server running locally
- Virtual environment (recommended)

### Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd Myr-Ag
   ```
2. **Setup the environment**

   ```bash
   make setup
   ```

   This will:

   - Create a virtual environment
   - Install dependencies
   - Create necessary directories

   **Alternative installation:**

   ```bash
   # Install dependencies
   pip install -r requirements.txt
   ```
3. **Start the application**

   ```bash
   make start
   ```

   Or use the shell script:

   ```bash
   ./myr-ag.sh start
   ```
4. **Access the application**

   - **API**: [http://localhost:8199](http://localhost:8199)
   - **UI**: [http://localhost:7860](http://localhost:7860)
   - **Health Check**: [http://localhost:8199/health](http://localhost:8199/health)

## Management Commands

### Using Makefile (Recommended)

```bash
# Start all services
make start

# Stop all services
make stop

# Restart all services
make restart

# Check service status
make status

# View logs
make logs

# Clean up temporary files
make clean

# Development mode (foreground)
make dev

# Production mode (background)
make prod

# Show help
make help
```

### Using Shell Script

```bash
# Start all services
./myr-ag.sh start

# Stop all services
./myr-ag.sh stop

# Restart all services
./myr-ag.sh restart

# Check service status
./myr-ag.sh status

# View logs
./myr-ag.sh logs

# Clean up
./myr-ag.sh clean

# Show help
./myr-ag.sh help
```

## Configuration

### Environment Variables (Optional)

The system works with default settings, but you can customize configuration by creating a `.env` file in the project root:

```env
# API Settings
API_HOST=0.0.0.0
API_PORT=8199

# UI Settings
UI_HOST=0.0.0.0
UI_PORT=7860

# Document Processing
CHUNK_SIZE=400
CHUNK_OVERLAP=100
MAX_FILE_SIZE_MB=100

# Timeouts (in seconds)
UPLOAD_TIMEOUT=900
PROCESSING_TIMEOUT=900
REQUEST_TIMEOUT=600

# LLM Settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# LEANN Vector Database
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LEANN_BACKEND=hnsw
```

### Key Settings (Default Values)

- **Chunk Size**: 400 characters (optimized for hybrid chunking)
- **Chunk Overlap**: 100 characters
- **Max File Size**: 100MB
- **Upload Timeout**: 15 minutes (for large documents)
- **Processing Timeout**: 15 minutes (for complex processing)

## Domain Management

### User-Controlled Domain Selection

The system provides full user control over document domain assignment, ensuring precise routing and optimal processing:

#### Domain Assignment Process
1. **Upload Document**: Upload your document using the web interface
2. **Select Domain**: Choose the appropriate domain from the dropdown:
   - **General**: For general-purpose documents
   - **Financial**: For financial reports, statements, budgets
   - **Legal**: For contracts, legal documents, compliance
   - **Medical**: For medical records, research, clinical data
   - **Academic**: For research papers, scholarly content
3. **Process**: The document is processed using the selected domain's specialized pipeline

#### Domain-Specific Benefits
- **Isolated Indexes**: Each domain has its own vector index for optimal performance
- **Specialized Processing**: Domain-specific chunking and query enhancement
- **Targeted Queries**: Queries are routed to the appropriate domain index
- **Granular Management**: Reset, rebuild, or clear individual domains

#### System Management
- **Index Management**: Select domain → Reset/Rebuild selected domain
- **Data Clearing**: Select domain → Clear selected domain data
- **Bulk Operations**: Reset/rebuild all domains at once
- **Safety Features**: Confirmation required for destructive operations

## Usage

### 1. Upload Documents

- Use the web interface to upload PDF, DOCX, TXT, or MD files
- **Upload Only**: Upload documents without immediate processing
- **Upload & Process**: Upload and immediately process documents with domain assignment
- **Process Existing**: Process documents already in the uploads directory
- **Process Uploaded Only**: Process only documents that haven't been processed yet
- **Index by Domain**: Index documents by domain with automatic detection or manual assignment
- Documents are automatically processed and chunked using intelligent fallback mechanisms
- The hybrid chunking strategy creates meaningful, searchable chunks

### 2. Query Documents

- Ask questions in natural language
- **Query Method Selection**: Choose between Standard (LEANN) or Excel Specific (LlamaIndex) methods
- **Dynamic Model Selection**: Choose from available Ollama models
- **Search Parameters**: Adjust chunk retrieval, temperature, and token limits
- **Progress Tracking**: Real-time progress indicators for long queries
- The system retrieves relevant chunks and generates answers
- View source documents and confidence scores

#### **Query Methods**

**Standard Method (LEANN only):**
- **Best for**: General documents (PDFs, Word docs, text files, images)
- **Features**: Uses LEANN vector store for all document types
- **Parameters**: All search parameters are used (chunks, temperature, max tokens)

**Excel Specific Method (LlamaIndex only) - EXPERIMENTAL:**
- **Best for**: Excel files and spreadsheet data
- **Features**: Uses LlamaIndex for Excel files only (⚠️ EXPERIMENTAL)
- **Parameters**: Ignores chunk retrieval (processes all data), uses temperature and max tokens
- **Note**: This feature is experimental and may have limitations or unexpected behavior

### 3. System Management

- **Document Management**: View, upload, process, and delete documents
- **Domain Indexing**: Index documents by domain with automatic detection or manual assignment
- **Index Management**: Reset, rebuild, and clear indexes for both LEANN and LlamaIndex
- **Vector Store Monitoring**: Real-time status monitoring for both vector stores
- **System Maintenance**: Comprehensive tools for system upkeep

#### **System Maintenance Operations**

**Reset Operations (Index Only):**
- **Reset LEANN Index**: Rebuilds LEANN index, preserves all data
- **Reset LlamaIndex Excel**: Rebuilds LlamaIndex index, preserves all data

**Rebuild Operations (Fast, No Reprocessing):**
- **Rebuild LEANN Index**: Rebuilds from existing processed documents
- **Rebuild LlamaIndex Excel**: Rebuilds from existing Excel processed files

**Clear Operations (Index + Data):**
- **Clear LEANN Documents**: Removes LEANN index + non-Excel processed documents
- **Clear LlamaIndex Excel**: Removes LlamaIndex index + processed Excel files
- **Clear Everything**: Removes all indexes and all data

### 4. Monitor Processing

- Check service status: `make status`
- View logs: `make logs`
- Monitor processing progress in real-time

## Architecture

```
Myr-Ag RAG System
├── Frontend (Gradio) - Port 7860
│   ├── Modern UI with progress tracking
│   ├── Real-time document processing status
│   ├── Query method selection (Standard/Excel Specific)
│   └── System management interface
├── Backend (FastAPI) - Port 8199
│   ├── REST API with optimized timeouts
│   ├── Document upload & processing endpoints
│   ├── RAG query endpoints (dual processing)
│   └── System management endpoints
├── Document Processor (Hybrid Chunking)
│   ├── Docling (primary) - PDF, DOCX, XLSX, PPTX, HTML, images
│   ├── pypdf (fallback) - Web-printed PDFs
│   └── OCR support - Scanned documents & images
├── Dual Vector Storage Architecture
│   ├── LEANN Vector Database (General Documents)
│   │   ├── HNSW backend for fast retrieval
│   │   ├── 97% space savings vs traditional vector DBs
│   │   └── Multilingual embeddings (768 dimensions)
│   └── LlamaIndex (Excel Files) - EXPERIMENTAL
│       ├── Persistent Excel indexing
│       ├── Advanced Excel chunking
│       └── Row-based processing
├── LLM Integration (Ollama)
│   ├── Local LLM server (llama3.2:3b default)
│   ├── Dynamic model selection
│   └── MPS acceleration for Apple Silicon
└── Management Tools
    ├── Makefile commands
    ├── Service management scripts
    ├── Index management (Reset/Rebuild/Clear)
    └── Logging & monitoring
```

## Enhanced Excel Processing (EXPERIMENTAL)

Myr-Ag features intelligent Excel processing that transforms tabular data into searchable, queryable content. **⚠️ Note: This feature is experimental and may have limitations or unexpected behavior.**

### **Smart Chunking Strategy**

- **Row-based processing**: Each row becomes an individual searchable chunk
- **Column-aware detection**: Automatically identifies amounts, names, dates, and categories
- **Natural language descriptions**: Converts tabular data to readable text

### **Example Transformations**

**Before (Raw Excel):**

```
| Employee | Department | Salary | Hire_Date |
|----------|------------|--------|-----------|
| John Doe | Sales      | 50000  | 2020-01-15|
```

**After (Searchable Chunks):**

```
Row 1: Employee: John Doe, Department: Sales | Salary: 50000
```

### **Supported Excel Patterns**

- **Any column structure**: Works with named columns, unnamed columns, or mixed scenarios
- **Flexible naming**: Works with any column name or no column name at all
- **Mixed data types**: Text, numbers, dates, formulas, empty cells
- **Multiple sheets**: Processes all sheets in the workbook automatically
- **Any language**: English, French, and other language column names supported
- **Empty columns**: Automatically skips empty columns and handles missing data

### **Query Examples**

- "What is John Doe's salary?" → Finds specific employee data
- "Show me all sales department employees" → Filters by department
- "What's the total salary for 2020?" → Aggregates by date
- "Find employees with salary over 60000" → Numeric comparisons

### **Benefits**

- **Granular search**: Find specific rows, not entire tables
- **Natural queries**: Ask questions in plain language
- **Context preservation**: Maintains relationships between columns
- **Multi-sheet support**: Search across all sheets simultaneously

## Recent Updates & Fixes

### System Improvements (Latest)

- **✅ System Status & Statistics Fixed**: Complete overhaul of system monitoring and statistics
  - **System Status**: Now shows accurate total documents and chunks across all domains
  - **LEANN Vector Store Management**: Displays comprehensive information for all 7 collections (main + 6 domains)
  - **Domain Statistics**: Real-time updates with correct document and chunk counts
  - **Aggregated Totals**: Proper summation of data across all domain collections

- **✅ Domain Indexing Fixed**: The "Index Documents by Domain" functionality now works correctly
  - Fixed domain extraction logic in domain index manager
  - Proper domain assignment and counting
  - Support for both automatic detection and manual domain assignment

- **✅ Document Processing Fixed**: Documents are now properly processed and indexed
  - Fixed UI to use correct upload endpoint for immediate processing
  - Proper domain assignment during upload
  - Improved error handling and user feedback

- **✅ Embedding Model Stability**: Switched to stable embedding model for reliable processing
  - Current model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - 384-dimensional embeddings for optimal performance
  - Multilingual support for English, French, and other languages

- **✅ UI Improvements**: Enhanced user interface and experience
  - Removed domain change functionality (replaced with domain indexing)
  - Removed "Index Documents by Domain" and "Individual Domain Management" subsections from frontend
  - Simplified document management workflow
  - Better error messages and status indicators
  - Automatic domain statistics loading on startup

- **✅ API Performance**: Optimized API startup and response times
  - Fixed slow startup due to unnecessary embedding model loading
  - Streamlined initialization process
  - Improved error handling and logging

### Known Issues Resolved

- **System Status Mismatch**: Fixed incorrect total_documents and total_chunks in system status
- **LEANN Vector Store Info**: Now shows all domain collections instead of just main collection
- **Domain Statistics Mismatch**: Fixed incorrect document/chunk counts in statistics
- **Document Processing Failures**: Resolved issues with document not being processed after upload
- **UI Startup Errors**: Fixed frontend startup issues after domain functionality changes
- **LEANN Indexing Issues**: Resolved embedding model compatibility problems
- **API Startup Performance**: Fixed slow startup due to multiple embedding model loading

## Troubleshooting

### Common Issues

1. **Port conflicts**

   - API uses port 8199 (configurable)
   - UI uses port 7860 (configurable)
2. **Timeout errors**

   - Increase timeouts in `.env` (optional) for large documents
   - Check system resources
3. **Service not starting**

   - Check logs: `make logs`
   - Verify virtual environment: `source venv/bin/activate`
   - Check dependencies: `make install`
4. **LEANN Searcher Initialization Issues**

   - **Error**: `Index not built or searcher not initialized`
   - **Cause**: LEANN searcher cannot find metadata files
   - **Solution**: The system automatically handles searcher initialization
   - **Verification**: Check `/system/vector-store` endpoint for searcher status
   - **Manual Fix**: Use `/system/force-initialize-searcher` endpoint if needed
5. **No Search Results**

   - **Symptom**: Queries return "No relevant chunks found"
   - **Possible Causes**:
     - Documents not properly indexed
     - Embedding model mismatch
     - Index corruption
   - **Solutions**:
     - Reprocess documents: `/system/reprocess-all-documents`
     - Check document count in vector store status
     - Verify LEANN index files exist in `.leann/indexes/`
6. **Document Processing Issues**

   - **Excel files not processed**: Ensure file is valid XLSX format
   - **OCR not working**: Check image quality and file format support
   - **Large files timeout**: Increase processing timeout in settings
   - **Unsupported format**: Check [Supported Formats](docs/supported_formats.md) documentation

### Debug Commands

```bash
# Check service status
make status

# View recent logs
make logs

# Check API health
curl http://localhost:8199/health

# Check UI availability
curl http://localhost:7860

# View process information
ps aux | grep "python run_"
```

## Logs

Logs are stored in the `logs/` directory:

- `logs/api.log` - API backend logs
- `logs/ui.log` - UI frontend logs

Log rotation is automatic (keeps last 1000 lines).

## Performance

### Hybrid Chunking Results

- **Strategy**: Paragraph-based + enhanced sentence splitting
- **Chunk Size**: 400 characters (optimized for sentence boundaries)
- **Chunk Overlap**: 100 characters (25% overlap for context preservation)
- **Typical Results**: 100-1000+ meaningful chunks per document
- **Processing Time**: 1-15 minutes depending on document size and complexity
- **Timeout**: 15 minutes (configurable)

### System Requirements

- **Memory**: 8GB+ RAM minimum, 16GB+ recommended for optimal performance
- **Storage**: 10GB+ free space for models and vector database
- **GPU**: MPS support for Apple Silicon (highly recommended)
- **LLM Models**:
  - **llama3.2:3b**: ~2GB RAM (minimum viable)
  - **llama3.2:7b**: ~4-6GB RAM (recommended)
  - **llama3.2:13b**: ~8-10GB RAM (high quality)
  - **llama3.2:70b**: ~40GB+ RAM (best quality, requires significant resources)

## Environment Configuration

### Hardware & OS

Myr-Ag has been tested on:

- **Operating System**: macOS 24.6.0 (Darwin)
- **Architecture**: Apple Silicon (M-series chip)
- **Memory**: 16GB+ RAM
- **Storage**: SSD with sufficient free space
- **GPU**: MPS (Metal Performance Shaders) acceleration enabled

### Software Stack

- **Python**: 3.11.7 (pyenv managed)
- **Virtual Environment**: venv with isolated dependencies
- **Package Manager**: pip with requirements.txt

### Key Dependencies

- **LEANN**: Ultra-efficient vector database with 97% space savings
- **FastAPI**: Backend API framework
- **Gradio**: Frontend interface with modern theming
- **Docling**: Advanced document processing for complex formats
- **pypdf**: PDF processing fallback for web-printed documents
- **Sentence Transformers**: paraphrase-multilingual-MiniLM-L12-v2 embedding model
- **PyTorch**: MPS-enabled for Apple Silicon
- **Ollama**: Local LLM server with llama3.2:3b model

### Network Configuration

- **API Backend**: [http://localhost:8199](http://localhost:8199)
- **UI Frontend**: [http://localhost:7860](http://localhost:7860)
- **Ollama Server**: [http://localhost:11434](http://localhost:11434)
- **Ports**: 8199 (API), 7860 (UI), 11434 (Ollama)

### Document Processing Settings

- **Chunk Size**: 400 characters (optimized for sentence-based strategy)
- **Chunk Overlap**: 100 characters
- **Supported Formats**: PDF, DOCX, XLSX, PPTX, TXT, MD, HTML, XHTML, CSV, PNG, JPEG, TIFF, BMP, WEBP, AsciiDoc, XML
- **Max File Size**: 100MB
- **Processing Strategy**: Sentence-based chunking with paragraph awareness
- **Fallback Mechanisms**: pypdf for problematic PDFs, direct text reading for simple formats
- **OCR Capabilities**: Automatic text extraction from scanned documents and images
- **Excel Processing**: Native support for XLSX files with table structure preservation (⚠️ EXPERIMENTAL)

### Performance Metrics (Expected)

- **Chunk Size**: 400 characters (optimized for sentence-based strategy)
- **Chunk Overlap**: 100 characters
- **Vector Dimensions**: 384 (paraphrase-multilingual-MiniLM-L12-v2)
- **Memory Usage**: ~4-8GB during processing (depends on LLM model size)
- **GPU Acceleration**: MPS enabled for embeddings

### Performance Expectations

- **Small Documents** (< 1MB): Processing in seconds
- **Medium Documents** (1-10MB): Processing in 1-5 minutes
- **Large Documents** (10-100MB): Processing in 5-15 minutes
- **Chunk Generation**: Typically 100-1000+ meaningful chunks depending on document size
- **Memory Usage**: Scales with document size and chunk count

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Deploy and monitor
5. Submit a pull request

## Research & References

### LEANN Vector Database

This project uses [LEANN (A Low-Storage Vector Index)](https://github.com/yichuan-w/LEANN) for ultra-efficient vector storage with 97% space savings. LEANN is a research project from Berkeley Sky Computing Lab.

**Citation:**

```bibtex
@misc{wang2025leannlowstoragevectorindex,
      title={LEANN: A Low-Storage Vector Index},
      author={Yichuan Wang and Shu Liu and Zhifei Li and Yongji Wu and Ziming Mao and Yilong Zhao and Xiao Yan and Zhiying Xu and Yang Zhou and Ion Stoica and Sewon Min and Matei Zaharia and Joseph E. Gonzalez},
      year={2025},
      eprint={2506.08276},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2506.08276},
}
```

**Key Benefits:**

- **97% storage savings** compared to traditional vector databases
- **Fast retrieval** with graph-based selective recomputation
- **Privacy-first** - runs entirely on your local device
- **Multiple backends** - HNSW (default) and DiskANN support

### Docling Document Processing

This project uses [Docling](https://github.com/docling-project/docling) for advanced document processing and parsing. Docling is developed by the Deep Search Team at IBM Research Zurich and provides comprehensive document understanding capabilities.

**Citation:**

```bibtex
@techreport{Docling,
  author = {Deep Search Team},
  month = {8},
  title = {Docling Technical Report},
  url = {https://arxiv.org/abs/2408.09869},
  eprint = {2408.09869},
  doi = {10.48550/arXiv.2408.09869},
  version = {1.0.0},
  year = {2024}
}
```

**Key Features:**

- **Multi-format support** - PDF, DOCX, PPTX, XLSX, HTML, images, audio, CSV, AsciiDoc, XML
- **Advanced PDF understanding** - page layout, reading order, table structure, formulas
- **OCR capabilities** - comprehensive support for scanned documents and images
- **Excel processing** - native XLSX support with table structure preservation (⚠️ EXPERIMENTAL)
- **Local execution** - privacy-first processing for sensitive data

📄 **Complete format support**: See [Supported Formats Documentation](docs/supported_formats.md) for detailed information about all supported document types and processing capabilities.

- **AI integrations** - seamless integration with LangChain, LlamaIndex, and other frameworks

## Documentation

- **[System Monitoring](docs/system_monitoring.md)** - Comprehensive monitoring and status documentation
- **[Multi-Pipeline Architecture](docs/multi_pipeline_architecture.md)** - Detailed architecture and pipeline design
- **[Data Structure Explanation](docs/data_structure_explanation.md)** - Data directory structure and organization
- **[Architecture Diagrams](docs/architecture_diagrams.md)** - Visual system architecture and flow diagrams
- **[Supported Formats](docs/supported_formats.md)** - Complete list of supported document types

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### What MIT License Means for Contributors

**You can:**

- Use the software for any purpose (commercial, personal, etc.)
- Modify and distribute the software
- Use it in proprietary software
- Sublicense it

**You must:**

- Include the original copyright notice
- Include the MIT license text

**We are not liable for:**

- Any damages or issues from using the software
- Any warranty claims

### Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to:

- Report bugs
- Suggest new features
- Submit code changes
- Join our community

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review logs: `make logs`
3. Check service status: `make status`
4. Open an issue on GitHub

---

**Happy Document Processing! 🚀📚**
