# Myr-Ag Architecture & Processing Diagrams

## System Overview

```mermaid
graph LR
    subgraph User["User"]
        U["Web Interface"]
    end
    
    subgraph App["Myr-Ag Application"]
        API["API Server"]
        PROC["Document Processor"]
        RAG["RAG Engine"]
        MON["System Monitor"]
    end
    
    subgraph Data["Data Storage"]
        DOCS["Documents"]
        VECTORS["Vector Indexes"]
        COLLECTIONS["Domain Collections"]
    end
    
    subgraph AI["AI Model"]
        LLM["Ollama LLM"]
    end
    
    U -->|"Upload & Query"| API
    API -->|"Process"| PROC
    API -->|"Search"| RAG
    API -->|"Monitor"| MON
    PROC -->|"Store"| DOCS
    PROC -->|"Index"| VECTORS
    VECTORS -->|"Domain Collections"| COLLECTIONS
    RAG -->|"Search"| VECTORS
    RAG -->|"Generate"| LLM
    LLM -->|"Answer"| RAG
    RAG -->|"Response"| API
    MON -->|"Statistics"| COLLECTIONS
    API -->|"Result"| U
    
    classDef user fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef app fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef data fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef ai fill:#fff3e0,stroke:#e65100,stroke-width:2px
    
    class U user
    class API,PROC,RAG,MON app
    class DOCS,VECTORS,COLLECTIONS data
    class LLM ai
```

## System Monitoring Architecture

```mermaid
graph TB
    subgraph UI["User Interface"]
        STATUS["System Status"]
        VECTOR["LEANN Vector Store Management"]
        DOMAIN["Domain Statistics"]
    end
    
    subgraph API["API Endpoints"]
        SYS_INFO["/system/info"]
        VECTOR_INFO["/system/vector-store"]
        DOMAIN_STATS["/domains/statistics"]
    end
    
    subgraph COLLECTIONS["Domain Collections"]
        MAIN["main_collection"]
        FINANCIAL["financial_collection"]
        LEGAL["legal_collection"]
        MEDICAL["medical_collection"]
        ACADEMIC["academic_collection"]
        EXCEL["excel_collection"]
        GENERAL["general_collection"]
    end
    
    subgraph STATS["Statistics Engine"]
        AGGREGATOR["Statistics Aggregator"]
        COUNTER["Document/Chunk Counter"]
        MONITOR["Collection Monitor"]
    end
    
    UI -->|"Display"| API
    API -->|"Query"| STATS
    STATS -->|"Monitor"| COLLECTIONS
    COLLECTIONS -->|"Data"| STATS
    STATS -->|"Aggregated Stats"| API
    API -->|"Real-time Data"| UI
    
    classDef ui fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef api fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef collections fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef stats fill:#fff3e0,stroke:#e65100,stroke-width:2px
    
    class STATUS,VECTOR,DOMAIN ui
    class SYS_INFO,VECTOR_INFO,DOMAIN_STATS api
    class MAIN,FINANCIAL,LEGAL,MEDICAL,ACADEMIC,EXCEL,GENERAL collections
    class AGGREGATOR,COUNTER,MONITOR stats
```

## Document Processing Flow

```mermaid
flowchart TD
    subgraph Upload["Upload"]
        A["Document Upload<br/>PDF/DOCX/XLSX"]
    end
  
    subgraph Decision["File Type Decision"]
        B{"File Type?"}
    end
  
    subgraph GeneralProcess["General Documents Process"]
        C1["Extract Text<br/>Docling"]
        C2["Create Chunks<br/>400 chars, sentence-based"]
        C3["Generate Embeddings<br/>Sentence Transformers"]
        C4["Store in LEANN<br/>Domain-specific index"]
    end
  
    subgraph ExcelProcess["Excel Files Process"]
        D1["Parse Excel<br/>LlamaIndex"]
        D2["Row-based Chunking<br/>Column-aware processing"]
        D3["Generate Embeddings<br/>LlamaIndex embeddings"]
        D4["Store in LlamaIndex<br/>Persistent database"]
    end
  
    subgraph Storage["Storage Result"]
        E["Documents Ready<br/>for Query"]
    end
  
    A --> B
    B -->|"PDF/DOCX/TXT/MD"| C1
    B -->|"XLSX"| D1
  
    C1 --> C2
    C2 --> C3
    C3 --> C4
  
    D1 --> D2
    D2 --> D3
    D3 --> D4
  
    C4 --> E
    D4 --> E
  
    classDef upload fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef decision fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef generalProcess fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef excelProcess fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef storage fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
  
    class A upload
    class B decision
    class C1,C2,C3,C4 generalProcess
    class D1,D2,D3,D4 excelProcess
    class E storage
```

## Domain Management

```mermaid
graph TB
    subgraph User["User"]
        UPLOAD["Upload Document"]
        SELECT["Choose Domain<br/>General, Financial, Legal, Medical, Academic"]
    end
  
    subgraph System["System"]
        PROCESS["Process Document"]
        ROUTE["Route to Domain Index"]
    end
  
    subgraph Indexes["Domain Indexes"]
        GENERAL[".leann_main_collection<br/>General Documents"]
        FINANCIAL[".leann_financial_collection<br/>Financial Documents"]
        LEGAL[".leann_legal_collection<br/>Legal Documents"]
        MEDICAL[".leann_medical_collection<br/>Medical Documents"]
        ACADEMIC[".leann_academic_collection<br/>Academic Documents"]
        EXCEL["LlamaIndex Database<br/>Excel Files"]
    end
  
    UPLOAD --> PROCESS
    SELECT --> PROCESS
    PROCESS --> ROUTE
  
    ROUTE -->|"General"| GENERAL
    ROUTE -->|"Financial"| FINANCIAL
    ROUTE -->|"Legal"| LEGAL
    ROUTE -->|"Medical"| MEDICAL
    ROUTE -->|"Academic"| ACADEMIC
    ROUTE -->|"Excel"| EXCEL
  
    classDef user fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef system fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef indexes fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
  
    class UPLOAD,SELECT user
    class PROCESS,ROUTE system
    class GENERAL,FINANCIAL,LEGAL,MEDICAL,ACADEMIC,EXCEL indexes
```

## Data Storage Structure

```mermaid
graph TB
    subgraph DataDir["Data Directory"]
        subgraph Uploads["Uploads"]
            UPLOADS["uploads/<br/>Raw documents"]
        end
   
        subgraph Processed["Processed"]
            PROCESSED["processed/<br/>JSON files with metadata"]
        end
   
        subgraph Indexes["Vector Indexes"]
            LEANN_INDEXES[".leann_*_collection/<br/>Domain-specific LEANN indexes<br/>General, Financial, Legal, Medical, Academic"]
            EXCEL_INDEX["llamaindex_excel_index/<br/>Excel files database"]
        end
    end
    
    UPLOADS -->|"Process"| PROCESSED
    PROCESSED -->|"Index by domain"| LEANN_INDEXES
    PROCESSED -->|"Index Excel"| EXCEL_INDEX
    
    classDef uploads fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef processed fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef indexes fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    
    class UPLOADS uploads
    class PROCESSED processed
    class LEANN_INDEXES,EXCEL_INDEX indexes
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
- **Embedding Model**: nomic-ai/nomic-embed-text-v2-moe
- **Vector Dimension**: 768
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
