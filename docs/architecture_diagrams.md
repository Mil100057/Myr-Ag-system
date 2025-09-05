# Myr-Ag Architecture & Processing Diagrams

## General Application Architecture

```mermaid
graph TB
    subgraph UILayer["User Interface Layer"]
        UI["Gradio Web Interface<br/>Port 7860"]
        API["FastAPI Backend<br/>Port 8199"]
    end
    
    subgraph CoreLayer["Core Processing Layer"]
        DP["Document Processor"]
        VS["Vector Store<br/>LEANN"]
        LIX["LlamaIndex<br/>Excel Processing<br/>(EXPERIMENTAL)"]
        DI["Document Indexer"]
        RP["RAG Pipeline"]
        OC["Ollama Client"]
    end
    
    subgraph DataLayer["Data Storage Layer"]
        UPLOADS["Uploads Directory"]
        PROCESSED["Processed Documents"]
        VECTOR_DB["LEANN Vector Database<br/>Ultra-efficient Storage"]
        LIX_DB["LlamaIndex Excel Database<br/>(EXPERIMENTAL)"]
    end
    
    subgraph ExternalLayer["External Services"]
        OLLAMA["Ollama Server<br/>Port 11434"]
        LLM["LLM Models<br/>llama3.2:3b"]
    end
    
    UI -->|"HTTP Requests"| API
    API -->|"HTTP Responses"| UI
    
    API -->|"Document Upload"| DP
    API -->|"Query Requests"| RP
    API -->|"System Info"| VS
    API -->|"Excel Info"| LIX
    
    DP -->|"Processed Text"| DI
    DI -->|"Chunks + Metadata"| VS
    DI -->|"Excel Chunks"| LIX
    VS -->|"Embeddings"| VECTOR_DB
    LIX -->|"Excel Index"| LIX_DB
    
    RP -->|"Search Query"| VS
    RP -->|"Excel Query"| LIX
    VS -->|"Relevant Chunks"| RP
    LIX -->|"Excel Chunks"| RP
    RP -->|"Context + Question"| OC
    OC -->|"LLM Request"| OLLAMA
    OLLAMA -->|"Generated Response"| OC
    OC -->|"Answer"| RP
    RP -->|"Final Response"| API
    
    UPLOADS -->|"Raw Documents"| DP
    DP -->|"Processed Data"| PROCESSED
    VS -->|"Vector Data"| VECTOR_DB
    LIX -->|"Excel Data"| LIX_DB
    
    classDef uiLayer fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef coreLayer fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef dataLayer fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef externalLayer fill:#fff3e0,stroke:#e65100,stroke-width:2px
    
    class UI,API uiLayer
    class DP,VS,LIX,DI,RP,OC coreLayer
    class UPLOADS,PROCESSED,VECTOR_DB,LIX_DB dataLayer
    class OLLAMA,LLM externalLayer
```

## Dual Processing Architecture

```mermaid
graph TB
    subgraph DocumentInput["Document Input"]
        UPLOAD["Document Upload<br/>PDF/DOCX/XLSX/TXT/MD"]
    end
    
    subgraph ProcessingDecision["Processing Decision"]
        UPLOAD --> CHECK{"File Type?"}
        CHECK -->|"Excel (.xlsx)"| EXCEL_PATH["Excel Processing Path<br/>(EXPERIMENTAL)"]
        CHECK -->|"Other Formats"| GENERAL_PATH["General Processing Path"]
    end
    
    subgraph ExcelProcessing["Excel Processing (LlamaIndex)"]
        EXCEL_PATH --> EXCEL_EXTRACT["Excel Text Extraction<br/>Docling"]
        EXCEL_EXTRACT --> EXCEL_CHUNK["Row-based Chunking<br/>Column-aware Processing"]
        EXCEL_CHUNK --> EXCEL_EMBED["Embedding Generation<br/>Sentence Transformers"]
        EXCEL_EMBED --> EXCEL_STORE["LlamaIndex Storage<br/>Persistent Index"]
    end
    
    subgraph GeneralProcessing["General Processing (LEANN)"]
        GENERAL_PATH --> GENERAL_EXTRACT["Text Extraction<br/>Docling/pypdf"]
        GENERAL_EXTRACT --> GENERAL_CHUNK["Sentence-based Chunking<br/>400 chars, 100 overlap"]
        GENERAL_CHUNK --> GENERAL_EMBED["Embedding Generation<br/>Sentence Transformers"]
        GENERAL_EMBED --> GENERAL_STORE["LEANN Storage<br/>Ultra-efficient Index"]
    end
    
    subgraph QueryProcessing["Query Processing"]
        QUERY["User Query"] --> METHOD{"Query Method?"}
        METHOD -->|"Standard"| LEANN_SEARCH["LEANN Search<br/>General Documents"]
        METHOD -->|"Excel Specific"| LIX_SEARCH["LlamaIndex Search<br/>Excel Files Only"]
        
        LEANN_SEARCH --> LEANN_RESULTS["LEANN Results"]
        LIX_SEARCH --> LIX_RESULTS["LlamaIndex Results"]
        
        LEANN_RESULTS --> LLM_GEN["LLM Generation<br/>Ollama"]
        LIX_RESULTS --> LLM_GEN
        LLM_GEN --> RESPONSE["Final Response"]
    end
    
    %% Styling
    classDef inputLayer fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef decisionLayer fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef excelLayer fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef generalLayer fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef queryLayer fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    
    class UPLOAD inputLayer
    class CHECK,METHOD decisionLayer
    class EXCEL_PATH,EXCEL_EXTRACT,EXCEL_CHUNK,EXCEL_EMBED,EXCEL_STORE,LIX_SEARCH,LIX_RESULTS excelLayer
    class GENERAL_PATH,GENERAL_EXTRACT,GENERAL_CHUNK,GENERAL_EMBED,GENERAL_STORE,LEANN_SEARCH,LEANN_RESULTS generalLayer
    class QUERY,LLM_GEN,RESPONSE queryLayer
```

## Document Processing Flow

```mermaid
flowchart TD
    subgraph UploadProcessing["Document Upload & Processing"]
        A["Document Upload<br/>PDF/DOCX/TXT/MD"] --> B{"File Validation"}
        B -->|"Valid"| C["Text Extraction<br/>via Docling"]
        B -->|"Invalid"| D["Error Response"]
        
        C --> E["Text Chunking<br/>Size: 400 chars<br/>Overlap: 100 chars<br/>Sentence-based strategy"]
        E --> F["Metadata Extraction<br/>File info, timestamps"]
        
        F --> G["Save Processed Document<br/>JSON format"]
        G --> H["Document Indexing"]
    end
    
    subgraph VectorProcessing["Vector Database Processing"]
        H --> I["Generate Embeddings<br/>sentence-transformers<br/>paraphrase-multilingual-MiniLM-L12-v2"]
        I --> J["Store in LEANN<br/>Index: main_collection<br/>Ultra-efficient storage"]
        J --> K["Build Index<br/>HNSW + CSR format<br/>97% space savings"]
    end
    
    subgraph RAGProcessing["RAG Query Processing"]
        L["User Query"] --> M["Query Preprocessing"]
        M --> N["Semantic Search<br/>Find relevant chunks"]
        N --> O["Retrieve Top Chunks<br/>Default: 5 chunks"]
        
        O --> P["Context Assembly<br/>Chunk content + metadata"]
        P --> Q["LLM Prompt Construction<br/>Context + Question"]
        
        Q --> R["Ollama API Call<br/>Model: llama3.2:3b"]
        R --> S["Response Generation<br/>Temperature: 0.7"]
        
        S --> T["Confidence Scoring<br/>Based on context relevance"]
        T --> U["Response Assembly<br/>Answer + Sources + Metadata"]
    end
    
    subgraph ResponseDelivery["Response Delivery"]
        U --> V["API Response<br/>JSON format"]
        V --> W["User Interface Display"]
    end
    
    %% Styling
    classDef uploadFlow fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef vectorFlow fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    classDef ragFlow fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef responseFlow fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    
    class A,B,C,D,E,F,G uploadFlow
    class H,I,J,K vectorFlow
    class L,M,N,O,P,Q,R,S,T ragFlow
    class U,V,W responseFlow
```

## System Components Detail

```mermaid
graph LR
    subgraph DocumentModule["Document Processing Module"]
        DP["DocumentProcessor"]
        DE["Docling Extractor"]
        TC["Text Chunker"]
        ME["Metadata Extractor"]
        
        DP --> DE
        DP --> TC
        DP --> ME
    end
    
    subgraph VectorModule["Vector Database Module"]
        VS["VectorStore"]
        EM["Embedding Model"]
        LEANN["LEANN Client"]
        SC["Search Controller"]
        
        VS --> EM
        VS --> LEANN
        VS --> SC
    end
    
    subgraph LLMModule["LLM Integration Module"]
        RP["RAGPipeline"]
        OC["OllamaClient"]
        QC["Query Controller"]
        RC["Response Controller"]
        
        RP --> OC
        RP --> QC
        RP --> RC
    end
    
    subgraph APILayer["API Layer"]
        FA["FastAPI App"]
        DR["Document Routes"]
        QR["Query Routes"]
        SR["System Routes"]
        
        FA --> DR
        FA --> QR
        FA --> SR
    end
    
    %% Connections between modules
    DP --> VS
    VS --> RP
    RP --> FA
    
    %% Styling
    classDef module fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef component fill:#e8f5e8,stroke:#1b5e20,stroke-width:1px
    
    class DP,VS,RP,FA module
    class DE,TC,ME,EM,LEANN,SC,OC,QC,RC,DR,QR,SR component
```

## Data Flow Architecture

```mermaid
graph TD
    subgraph InputLayer["Input Layer"]
        UPLOAD["Document Upload<br/>HTTP POST"]
        QUERY["User Query<br/>HTTP POST"]
    end
    
    subgraph ProcessingPipeline["Processing Pipeline"]
        EXTRACT["Text Extraction<br/>Docling"]
        CHUNK["Text Chunking<br/>LlamaIndex"]
        EMBED["Embedding Generation<br/>Sentence Transformers"]
        STORE["Vector Storage<br/>LEANN"]
        SEARCH["Semantic Search<br/>Vector Similarity"]
        GENERATE["LLM Generation<br/>Ollama"]
    end
    
    subgraph OutputLayer["Output Layer"]
        RESPONSE["API Response<br/>JSON"]
        UI_UPDATE["UI Update<br/>Real-time"]
    end
    
    subgraph DataStores["Data Stores"]
        RAW["Raw Documents<br/>File System"]
        PROCESSED["Processed Data<br/>JSON Files"]
        VECTORS["Vector Embeddings<br/>LEANN"]
        METADATA["Document Metadata<br/>LEANN"]
    end
    
    %% Data Flow
    UPLOAD --> RAW
    RAW --> EXTRACT
    EXTRACT --> CHUNK
    CHUNK --> EMBED
    EMBED --> STORE
    STORE --> VECTORS
    STORE --> METADATA
    
    QUERY --> SEARCH
    SEARCH --> VECTORS
    SEARCH --> GENERATE
    GENERATE --> RESPONSE
    RESPONSE --> UI_UPDATE
    
    %% Styling
    classDef inputLayer fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef processingLayer fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef outputLayer fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef dataLayer fill:#fff3e0,stroke:#e65100,stroke-width:2px
    
    class UPLOAD,QUERY inputLayer
    class EXTRACT,CHUNK,EMBED,STORE,SEARCH,GENERATE processingLayer
    class RESPONSE,UI_UPDATE outputLayer
    class RAW,PROCESSED,VECTORS,METADATA dataLayer
```

## Key Technical Specifications

### Document Processing

- **Dual Processing Architecture**: LEANN for general documents, LlamaIndex for Excel files (⚠️ EXPERIMENTAL)
- **Chunk Size**: 400 characters (general), row-based (Excel)
- **Chunk Overlap**: 100 characters (general), column-aware (Excel)
- **Supported Formats**: PDF, DOCX, XLSX, PPTX, TXT, MD, HTML, XHTML, CSV, PNG, JPEG, TIFF, BMP, WEBP, AsciiDoc, XML
- **Max File Size**: 100MB
- **Processing Strategy**: 
  - **General Documents**: Sentence-based chunking with paragraph awareness
  - **Excel Files**: Row-based chunking with column-aware processing (⚠️ EXPERIMENTAL)
- **OCR Support**: Automatic text extraction from scanned PDFs and images
- **Excel Support**: Native processing of XLSX files with table structure preservation (⚠️ EXPERIMENTAL)
- **Fallback Mechanisms**: pypdf for problematic PDFs, direct text reading for simple formats

### Vector Database

- **Dual Storage Architecture**: LEANN for general documents, LlamaIndex for Excel files (⚠️ EXPERIMENTAL)
- **Embedding Model**: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- **Vector Dimension**: 384
- **LEANN Database**: Ultra-efficient storage for general documents
  - **Index**: main_collection
  - **Storage Efficiency**: 97% space savings vs traditional vector databases
  - **Backend**: HNSW with CSR format for optimal performance
  - **Searcher Initialization**: Automatic initialization with fallback mechanisms
  - **Metadata Files**: main_collection.meta.json for searcher configuration
- **LlamaIndex Database**: Persistent Excel indexing (⚠️ EXPERIMENTAL)
  - **Index**: data/llamaindex_excel_index
  - **Storage**: Persistent disk-based storage
  - **Processing**: Row-based chunking with column awareness
  - **Query Method**: Excel-specific queries only

### LLM Integration

- **Model**: llama3.2:3b
- **Server**: Ollama (localhost:11434)
- **Temperature**: 0.7 (configurable)
- **Max Tokens**: 2048 (configurable)

### Performance

- **MPS Acceleration**: Enabled for Apple Silicon
- **Batch Processing**: Supported for multiple documents
- **Real-time Indexing**: Immediate after processing
- **Async Operations**: Non-blocking API calls

## Research References

### LEANN Vector Database

This system uses [LEANN (A Low-Storage Vector Index)](https://github.com/yichuan-w/LEANN) for ultra-efficient vector storage. LEANN is a research project from Berkeley Sky Computing Lab that provides 97% storage savings compared to traditional vector databases.

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

**Key Technical Features:**

- **Graph-based selective recomputation**: Only compute embeddings for nodes in the search path
- **High-degree preserving pruning**: Keep important "hub" nodes while removing redundant connections
- **Dynamic batching**: Efficiently batch embedding computations for GPU utilization
- **Two-level search**: Smart graph traversal that prioritizes promising nodes

### Docling Document Processing

This system uses [Docling](https://github.com/docling-project/docling) for advanced document processing and parsing. Docling is developed by the Deep Search Team at IBM Research Zurich and provides comprehensive document understanding capabilities.

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

**Key Technical Features:**

- **Multi-format parsing**: PDF, DOCX, PPTX, XLSX, HTML, images, audio files
- **Advanced PDF understanding**: Page layout analysis, reading order detection, table structure recognition
- **OCR capabilities**: Comprehensive support for scanned documents and images
- **Local execution**: Privacy-first processing for sensitive data
- **Unified document representation**: Expressive DoclingDocument format for consistent processing

---

*These diagrams can be viewed in any Markdown viewer that supports Mermaid diagrams (GitHub, GitLab, VS Code with Mermaid extension, etc.)*
