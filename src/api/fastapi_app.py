"""
FastAPI backend for Myr-Ag RAG System.
"""
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from loguru import logger

# Configure loguru to write to file
logger.add("logs/api.log", rotation="10 MB", retention="7 days", level="INFO")

from config.settings import settings
from src.document_processing.document_processor import DocumentProcessor

from src.vector_db.document_indexer import DocumentIndexer
from src.vector_db.unified_vector_store import UnifiedVectorStore, unified_manager
from src.llm_integration.ollama_client import OllamaClient
from src.llm_integration.rag_pipeline import RAGPipeline
from src.llm_integration.enhanced_rag_pipeline import EnhancedRAGPipeline
from src.llm_integration.enhanced_rag_with_pipelines import EnhancedRAGWithPipelines


# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    question: str
    n_chunks: int = 5
    temperature: float = 0.7
    max_tokens: int = 2048
    model: str = None


class ModelChangeRequest(BaseModel):
    model_name: str



class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence_score: float
    processing_time: float
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class SystemInfo(BaseModel):
    ollama_health: bool
    available_models: List[Dict[str, Any]]
    default_model: str
    index_info: Dict[str, Any]
    total_documents: int
    total_chunks: int


class DocumentInfo(BaseModel):
    file_name: str
    file_path: str
    file_size: int
    file_extension: str
    content_length: int
    chunk_count: int
    processing_timestamp: str
    domain: Optional[str] = None


class FastAPIBackend:
    """FastAPI backend for Myr-Ag RAG System."""
    
    def __init__(self):
        """Initialize the FastAPI backend."""
        self.app = FastAPI(
            title="Myr-Ag RAG System API",
            description="Document Processing and Local LLM Querying System",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize components
        self._initialize_components()
        
        # Setup routes
        self._setup_routes()
        
        logger.info("FastAPI backend initialized successfully")
    
    def _initialize_components(self):
        """Initialize system components."""
        try:
            # Initialize LEANN vector store
            self.vector_store = unified_manager.get_or_create_store("main_collection")
            
            # Force initialize LEANN searcher if index exists
            if hasattr(self.vector_store, 'force_initialize_searcher'):
                logger.info("Attempting to force initialize LEANN searcher...")
                self.vector_store.force_initialize_searcher()
                logger.info("LEANN searcher initialization completed")
            else:
                logger.warning("Vector store does not have force_initialize_searcher method")
            
            # Initialize document processor
            self.doc_processor = DocumentProcessor()
            
            # Initialize document indexer with differentiation
            self.indexer = DocumentIndexer(self.vector_store, self.doc_processor)
            
            # Initialize Ollama client
            self.ollama_client = OllamaClient()
            
            # Initialize RAG pipeline
            self.rag_pipeline = RAGPipeline(self.indexer, self.ollama_client)
            
            # Initialize enhanced RAG pipeline with LlamaIndex
            self.enhanced_rag_pipeline = EnhancedRAGPipeline(
                self.indexer, 
                self.ollama_client, 
                enable_llamaindex=True
            )
            
            # Initialize enhanced RAG pipeline with specialized pipelines
            self.specialized_rag_pipeline = EnhancedRAGWithPipelines(
                self.indexer,
                self.ollama_client,
                Path(settings.DATA_DIR)
            )
            
            logger.info(f"All system components initialized successfully with LEANN + LlamaIndex for Excel + Specialized Pipelines")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _setup_routes(self):
        """Setup API routes."""
        from src.utils.device_manager import device_manager, get_mps_info
        
        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "message": "Myr-Ag RAG System API",
                "version": "1.0.0",
                "status": "running"
            }
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            try:
                ollama_health = self.ollama_client.health_check()
                return {
                    "status": "healthy" if ollama_health else "unhealthy",
                    "ollama_server": "running" if ollama_health else "down",
                    "timestamp": str(Path(__file__).stat().st_mtime)
                }
            except Exception as e:
                return {"status": "error", "error": str(e)}
        
        @self.app.get("/system/mps-info")
        async def get_mps_info_endpoint():
            """Get MPS device information and performance metrics."""
            try:
                mps_info = get_mps_info()
                return {
                    "status": "success",
                    "mps_info": mps_info,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"status": "error", "error": str(e)}
        
        @self.app.post("/system/cleanup-mps")
        async def cleanup_mps_memory():
            """Clean up MPS memory."""
            try:
                device_manager.cleanup_memory("mps")
                return {
                    "status": "success",
                    "message": "MPS memory cleaned successfully",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"status": "error", "error": str(e)}
        
        @self.app.get("/system/info", response_model=SystemInfo)
        async def get_system_info():
            """Get system information."""
            try:
                pipeline_info = self.rag_pipeline.get_pipeline_info()
                index_info = self.indexer.get_index_info()
                
                # Count documents in uploads directory
                uploads_dir = Path(settings.DATA_DIR) / "uploads"
                total_documents = len([f for f in uploads_dir.iterdir() if f.is_file()]) if uploads_dir.exists() else 0
                
                # Get total chunks from all domains
                domain_stats = self.specialized_rag_pipeline.get_domain_statistics()
                total_chunks = sum(domain_info.get("chunk_count", 0) for domain_info in domain_stats.values() if "error" not in domain_info)
                
                return SystemInfo(
                    ollama_health=pipeline_info.get("ollama_health", False),
                    available_models=pipeline_info.get("available_models", []),
                    default_model=pipeline_info.get("default_model", "unknown"),
                    index_info=index_info,
                    total_documents=total_documents,
                    total_chunks=total_chunks
                )
            except Exception as e:
                logger.error(f"Error getting system info: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/documents", response_model=List[DocumentInfo])
        async def list_documents(refresh: bool = Query(False, description="Force refresh of document data")):
            """List all documents (both uploaded and processed) with real-time data."""
            try:
                # Get real processed document data
                processed_dir = Path(settings.DATA_DIR) / "processed"
                uploads_dir = Path(settings.DATA_DIR) / "uploads"
                documents = []
                processed_files = set()  # Track which files have been processed
                
                # First, collect all processed documents
                if processed_dir.exists():
                    for file_path in processed_dir.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() == ".json" and file_path.name.endswith("_processed.json"):
                            try:
                                import json
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    doc_data = json.load(f)
                                
                                # Get domain from the corresponding metadata file
                                metadata_file = processed_dir / f"{file_path.stem.replace('_processed', '')}.json"
                                domain = "general"  # default
                                if metadata_file.exists():
                                    try:
                                        with open(metadata_file, 'r', encoding='utf-8') as f:
                                            metadata = json.load(f)
                                        domain = metadata.get('domain', 'general')
                                    except Exception as e:
                                        logger.warning(f"Error reading domain from {metadata_file}: {e}")
                                
                                # Get original file info from the processed data
                                original_file_path = doc_data.get('file_path', '')
                                if original_file_path:
                                    original_file = Path(original_file_path)
                                    if original_file.exists():
                                        stat = original_file.stat()
                                        processed_files.add(original_file.name)  # Track as processed
                                        
                                        documents.append(DocumentInfo(
                                            file_name=original_file.name,
                                            file_path=str(original_file),
                                            file_size=stat.st_size,
                                            file_extension=original_file.suffix.lower(),
                                            content_length=doc_data.get('content_length', 0),
                                            chunk_count=doc_data.get('metadata', {}).get('chunk_count', 0),
                                            processing_timestamp=doc_data.get('metadata', {}).get('processing_timestamp', ''),
                                            domain=domain
                                        ))
                                    else:
                                        logger.warning(f"Original file not found: {original_file_path}")
                                else:
                                    logger.warning(f"No file_path in processed document: {file_path}")
                            except Exception as e:
                                logger.warning(f"Error reading processed document {file_path}: {e}")
                                continue
                
                # Then, add all uploaded documents that haven't been processed yet
                if uploads_dir.exists():
                    for file_path in uploads_dir.iterdir():
                        if file_path.is_file() and file_path.name not in processed_files:
                            try:
                                stat = file_path.stat()
                                documents.append(DocumentInfo(
                                    file_name=file_path.name,
                                    file_path=str(file_path),
                                    file_size=stat.st_size,
                                    file_extension=file_path.suffix.lower(),
                                    content_length=0,
                                    chunk_count=0,  # Not processed yet
                                    processing_timestamp="",
                                    domain="general"  # Default domain for unprocessed documents
                                ))
                            except Exception as e:
                                logger.warning(f"Error processing file {file_path}: {e}")
                                continue
                
                logger.info(f"Returning {len(documents)} documents (processed + uploaded)")
                for doc in documents:
                    status = "PROCESSED" if doc.chunk_count > 0 else "UPLOADED"
                    logger.info(f"Document: {doc.file_name}, Status: {status}, Chunks: {doc.chunk_count}, Content: {doc.content_length}")
                return documents
                
            except Exception as e:
                logger.error(f"Error listing documents: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/documents/upload")
        async def upload_documents(files: List[UploadFile] = File(...)):
            """Upload multiple documents."""
            try:
                if not files:
                    raise HTTPException(status_code=400, detail="No files provided")
                
                uploads_dir = Path(settings.DATA_DIR) / "uploads"
                uploads_dir.mkdir(parents=True, exist_ok=True)
                
                uploaded_files = []
                for file in files:
                    if file.filename:
                        # Save uploaded file
                        file_path = uploads_dir / file.filename
                        try:
                            # Read file content and save it
                            content = await file.read()
                            with open(file_path, "wb") as buffer:
                                buffer.write(content)
                            
                            uploaded_files.append(str(file_path))
                            logger.info(f"Successfully uploaded file: {file.filename}")
                        except Exception as e:
                            logger.error(f"Error uploading file {file.filename}: {e}")
                            continue
                
                # Process uploaded documents using the general indexer (for backward compatibility)
                processed_count = self.indexer.index_directory(uploads_dir)
                
                return {
                    "message": f"Successfully uploaded and processed {len(uploaded_files)} documents",
                    "uploaded_files": uploaded_files,
                    "processed_count": processed_count
                }
                
            except Exception as e:
                logger.error(f"Error uploading documents: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/documents/upload-only")
        async def upload_only_documents(files: List[UploadFile] = File(...)):
            """Upload documents without processing them."""
            try:
                if not files:
                    raise HTTPException(status_code=400, detail="No files provided")
                
                uploads_dir = Path(settings.DATA_DIR) / "uploads"
                uploads_dir.mkdir(parents=True, exist_ok=True)
                
                uploaded_files = []
                for file in files:
                    if file.filename:
                        # Save uploaded file
                        file_path = uploads_dir / file.filename
                        try:
                            # Read file content and save it
                            content = await file.read()
                            with open(file_path, "wb") as buffer:
                                buffer.write(content)
                            
                            uploaded_files.append(str(file_path))
                            logger.info(f"Successfully uploaded file (no processing): {file.filename}")
                        except Exception as e:
                            logger.error(f"Error uploading file {file.filename}: {e}")
                            continue
                
                return {
                    "message": f"Successfully uploaded {len(uploaded_files)} documents (no processing)",
                    "uploaded_files": uploaded_files,
                    "processed_count": 0
                }
                
            except Exception as e:
                logger.error(f"Error uploading documents: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/documents/upload-with-domain")
        async def upload_documents_with_domain(files: List[UploadFile] = File(...), domain: str = Form("general")):
            """Upload documents and process them in a specific domain."""
            try:
                if not files:
                    raise HTTPException(status_code=400, detail="No files provided")
                
                uploads_dir = Path(settings.DATA_DIR) / "uploads"
                uploads_dir.mkdir(parents=True, exist_ok=True)
                
                uploaded_files = []
                processed_count = 0
                
                for file in files:
                    if file.filename:
                        # Save uploaded file
                        file_path = uploads_dir / file.filename
                        try:
                            # Read file content and save it
                            content = await file.read()
                            with open(file_path, "wb") as buffer:
                                buffer.write(content)
                            
                            uploaded_files.append(str(file_path))
                            logger.info(f"Successfully uploaded file: {file.filename}")
                            
                            # Process the document in the specified domain
                            success, message = self.specialized_rag_pipeline.index_document(file_path, domain)
                            if success:
                                processed_count += 1
                                logger.info(f"Successfully processed {file.filename} in {domain} domain")
                            else:
                                logger.warning(f"Failed to process {file.filename} in {domain} domain: {message}")
                                
                        except Exception as e:
                            logger.error(f"Error uploading file {file.filename}: {e}")
                            continue
                
                return {
                    "message": f"Successfully uploaded and processed {processed_count}/{len(uploaded_files)} documents in {domain} domain",
                    "uploaded_files": uploaded_files,
                    "processed_count": processed_count,
                    "domain": domain
                }
                
            except Exception as e:
                logger.error(f"Error uploading documents with domain: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/documents/upload-only-with-domain")
        async def upload_only_documents_with_domain(files: List[UploadFile] = File(...), domain: str = Form("general")):
            """Upload documents without processing them, but store domain metadata."""
            try:
                if not files:
                    raise HTTPException(status_code=400, detail="No files provided")
                
                uploads_dir = Path(settings.DATA_DIR) / "uploads"
                uploads_dir.mkdir(parents=True, exist_ok=True)
                
                uploaded_files = []
                for file in files:
                    if file.filename:
                        # Save uploaded file
                        file_path = uploads_dir / file.filename
                        try:
                            # Read file content and save it
                            content = await file.read()
                            with open(file_path, "wb") as buffer:
                                buffer.write(content)
                            
                            uploaded_files.append(str(file_path))
                            logger.info(f"Successfully uploaded file: {file.filename}")
                            
                            # Store domain metadata for the uploaded file
                            self.specialized_rag_pipeline.pipeline_manager.domain_manager._store_domain_metadata(file_path, domain)
                            logger.info(f"Stored domain metadata for {file.filename}: {domain}")
                                
                        except Exception as e:
                            logger.error(f"Error uploading file {file.filename}: {e}")
                            continue
                
                return {
                    "message": f"Successfully uploaded {len(uploaded_files)} documents with domain {domain}",
                    "uploaded_files": uploaded_files,
                    "domain": domain
                }
                
            except Exception as e:
                logger.error(f"Error uploading documents with domain: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/documents/process")
        async def process_documents():
            """Process all documents in uploads directory."""
            try:
                uploads_dir = Path(settings.DATA_DIR) / "uploads"
                
                if not uploads_dir.exists():
                    raise HTTPException(status_code=400, detail="Uploads directory not found")
                
                # Process documents
                processed_count = self.indexer.index_directory(uploads_dir)
                
                return {
                    "message": f"Successfully processed {processed_count} documents",
                    "processed_count": processed_count
                }
                
            except Exception as e:
                logger.error(f"Error processing documents: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/documents/process-uploaded-only")
        async def process_uploaded_only_documents():
            """Process only documents that are uploaded but not yet processed."""
            try:
                uploads_dir = Path(settings.DATA_DIR) / "uploads"
                processed_dir = Path(settings.DATA_DIR) / "processed"
                
                if not uploads_dir.exists():
                    raise HTTPException(status_code=400, detail="Uploads directory not found")
                
                # Get list of already processed files
                processed_files = set()
                if processed_dir.exists():
                    for file_path in processed_dir.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() == ".json":
                            try:
                                import json
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    doc_data = json.load(f)
                                original_file_path = doc_data.get('file_path', '')
                                if original_file_path:
                                    original_file = Path(original_file_path)
                                    if original_file.exists():
                                        processed_files.add(original_file.name)
                            except Exception as e:
                                logger.warning(f"Error reading processed document {file_path}: {e}")
                                continue
                
                # Find uploaded files that haven't been processed
                unprocessed_files = []
                for file_path in uploads_dir.iterdir():
                    if file_path.is_file() and file_path.name not in processed_files:
                        unprocessed_files.append(file_path)
                
                if not unprocessed_files:
                    return {
                        "message": "No unprocessed documents found - all uploaded documents are already processed",
                        "processed_count": 0,
                        "unprocessed_files": []
                    }
                
                # Process only the unprocessed files
                processed_count = 0
                for file_path in unprocessed_files:
                    try:
                        # Use the indexer to process and index the document
                        logger.info(f"Processing unprocessed document: {file_path.name}")
                        
                        success = self.indexer.index_document(file_path)
                        if success:
                            processed_count += 1
                            logger.info(f"Successfully processed unprocessed document: {file_path.name}")
                        else:
                            logger.warning(f"Failed to process document: {file_path.name}")
                    except Exception as e:
                        logger.error(f"Error processing document {file_path.name}: {e}")
                        continue
                
                # Prepare detailed response
                if processed_count == 0 and unprocessed_files:
                    message = f"No documents could be processed. {len(unprocessed_files)} documents failed processing (possibly due to format issues or empty content)."
                else:
                    message = f"Successfully processed {processed_count} unprocessed documents"
                
                return {
                    "message": message,
                    "processed_count": processed_count,
                    "unprocessed_files": [f.name for f in unprocessed_files],
                    "failed_count": len(unprocessed_files) - processed_count
                }
                
            except Exception as e:
                logger.error(f"Error processing uploaded-only documents: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/query", response_model=QueryResponse)
        async def query_documents(request: QueryRequest):
            """Query documents using RAG pipeline."""
            try:
                logger.info(f"Processing RAG query: {request.question}")
                
                # Change model if specified
                if request.model and request.model != self.ollama_client.current_model:
                    logger.info(f"Changing model from {self.ollama_client.current_model} to {request.model}")
                    self.ollama_client.change_model(request.model)
                
                # Execute RAG query
                response = self.rag_pipeline.query(
                    question=request.question,
                    n_chunks=request.n_chunks,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                )
                
                return QueryResponse(
                    question=response.query,
                    answer=response.answer,
                    confidence_score=response.confidence_score,
                    processing_time=response.processing_time,
                    sources=response.sources,
                    metadata=response.metadata
                )
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/query-enhanced")
        async def query_documents_enhanced(request: QueryRequest):
            """Query documents using enhanced RAG pipeline with LlamaIndex."""
            try:
                logger.info(f"Processing enhanced RAG query: {request.question}")
                
                # Change model if specified
                if request.model and request.model != self.ollama_client.current_model:
                    logger.info(f"Changing model from {self.ollama_client.current_model} to {request.model}")
                    self.ollama_client.change_model(request.model)
                
                # Execute enhanced RAG query (Excel-only mode)
                response = self.enhanced_rag_pipeline.query(
                    question=request.question,
                    n_chunks=request.n_chunks,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    use_direct_answer=True,
                    excel_only=True
                )
                
                return {
                    "question": response.query,
                    "answer": response.answer,
                    "confidence_score": response.confidence_score,
                    "processing_time": response.processing_time,
                    "sources": response.sources,
                    "excel_sources": response.excel_sources,
                    "direct_answer": response.direct_answer,
                    "metadata": response.metadata
                }
                
            except Exception as e:
                logger.error(f"Error processing enhanced query: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/query-specialized")
        async def query_documents_specialized(request: QueryRequest):
            """Query documents using specialized domain pipelines with auto-detection."""
            try:
                logger.info(f"Processing specialized RAG query: {request.question}")
                
                # Change model if specified
                if request.model and request.model != self.ollama_client.current_model:
                    self.ollama_client.switch_model(request.model)
                
                # Execute specialized RAG query with auto-detection
                response = self.specialized_rag_pipeline.query(
                    question=request.question,
                    n_chunks=request.n_chunks,
                    interaction_mode="precise_question"
                )
                
                return {
                    "question": response.query,
                    "answer": response.answer,
                    "confidence_score": response.confidence_score,
                    "processing_time": response.processing_time,
                    "sources": response.sources,
                    "pipeline_used": response.pipeline_used,
                    "domain": response.domain,
                    "enhanced_query": response.enhanced_query,
                    "metadata": response.metadata
                }
                
            except Exception as e:
                logger.error(f"Error processing specialized query: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/query-specialized/{domain}")
        async def query_documents_specific_pipeline(domain: str, request: QueryRequest):
            """Query documents using a specific specialized pipeline."""
            try:
                logger.info(f"Processing {domain} pipeline query: {request.question}")
                
                # Change model if specified
                if request.model and request.model != self.ollama_client.current_model:
                    self.ollama_client.switch_model(request.model)
                
                # Execute specialized RAG query with specific pipeline
                response = self.specialized_rag_pipeline.query_with_specific_pipeline(
                    question=request.question,
                    domain=domain,
                    n_chunks=request.n_chunks,
                    interaction_mode="precise_question"
                )
                
                return {
                    "question": response.query,
                    "answer": response.answer,
                    "confidence_score": response.confidence_score,
                    "processing_time": response.processing_time,
                    "sources": response.sources,
                    "pipeline_used": response.pipeline_used,
                    "domain": response.domain,
                    "enhanced_query": response.enhanced_query,
                    "metadata": response.metadata
                }
                
            except Exception as e:
                logger.error(f"Error processing {domain} pipeline query: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/pipelines")
        async def list_pipelines():
            """List all available specialized pipelines."""
            try:
                pipelines = self.specialized_rag_pipeline.list_available_pipelines()
                return {
                    "available_pipelines": pipelines,
                    "auto_detection_enabled": self.specialized_rag_pipeline.auto_detect_pipeline
                }
            except Exception as e:
                logger.error(f"Error listing pipelines: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/pipelines/{domain}/info")
        async def get_pipeline_info(domain: str):
            """Get information about a specific pipeline."""
            try:
                info = self.specialized_rag_pipeline.get_pipeline_info(domain)
                return info
            except Exception as e:
                logger.error(f"Error getting pipeline info for {domain}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/pipelines/test-detection")
        async def test_pipeline_detection(request: dict):
            """Test pipeline detection with sample queries."""
            try:
                test_queries = request.get("queries", [])
                if not test_queries:
                    raise HTTPException(status_code=400, detail="No queries provided")
                
                results = self.specialized_rag_pipeline.test_pipeline_detection(test_queries)
                return {"detection_results": results}
                
            except Exception as e:
                logger.error(f"Error testing pipeline detection: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/documents/index-domain")
        async def index_document_domain(request: dict):
            """Index a document in a specific domain."""
            try:
                file_path = request.get("file_path")
                force_domain = request.get("force_domain")
                
                if not file_path:
                    raise HTTPException(status_code=400, detail="file_path is required")
                
                file_path = Path(file_path)
                if not file_path.exists():
                    raise HTTPException(status_code=404, detail="File not found")
                
                success, message = self.specialized_rag_pipeline.index_document(file_path, force_domain)
                
                return {
                    "success": success,
                    "message": message,
                    "file_path": str(file_path),
                    "domain": force_domain or "auto-detected"
                }
                
            except Exception as e:
                logger.error(f"Error indexing document in domain: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/documents/index-directory-domain")
        async def index_directory_domain(request: dict):
            """Index all documents in a directory, routing to appropriate domains."""
            try:
                directory_path = request.get("directory_path")
                force_domain = request.get("force_domain")
                
                if not directory_path:
                    raise HTTPException(status_code=400, detail="directory_path is required")
                
                directory_path = Path(directory_path)
                if not directory_path.exists():
                    raise HTTPException(status_code=404, detail="Directory not found")
                
                results = self.specialized_rag_pipeline.index_directory(directory_path, force_domain)
                
                return {
                    "success": True,
                    "message": f"Indexed documents in {len(results)} domains",
                    "directory_path": str(directory_path),
                    "domain_results": results
                }
                
            except Exception as e:
                logger.error(f"Error indexing directory in domains: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/domains/statistics")
        async def get_domain_statistics():
            """Get statistics for all domain indexes."""
            try:
                stats = self.specialized_rag_pipeline.get_domain_statistics()
                return stats
                
            except Exception as e:
                logger.error(f"Error getting domain statistics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/domains/{domain}/reset")
        async def reset_domain_index(domain: str):
            """Reset a specific domain index."""
            try:
                success = self.specialized_rag_pipeline.reset_domain_index(domain)
                
                if success:
                    return {"message": f"Domain '{domain}' index reset successfully"}
                else:
                    raise HTTPException(status_code=500, detail=f"Failed to reset domain '{domain}' index")
                    
            except Exception as e:
                logger.error(f"Error resetting domain {domain} index: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/domains/{domain}/rebuild")
        async def rebuild_domain_index(domain: str):
            """Rebuild a specific domain index."""
            try:
                success = self.specialized_rag_pipeline.rebuild_domain_index(domain)
                
                if success:
                    return {"message": f"Domain '{domain}' index rebuilt successfully"}
                else:
                    raise HTTPException(status_code=500, detail=f"Failed to rebuild domain '{domain}' index")
                    
            except Exception as e:
                logger.error(f"Error rebuilding domain {domain} index: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/domains/reset-all")
        async def reset_all_domain_indexes():
            """Reset all domain indexes."""
            try:
                results = self.specialized_rag_pipeline.reset_all_domain_indexes()
                
                return {
                    "message": "Domain indexes reset completed",
                    "results": results
                }
                
            except Exception as e:
                logger.error(f"Error resetting all domain indexes: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/domains/rebuild-all")
        async def rebuild_all_domain_indexes():
            """Rebuild all domain indexes."""
            try:
                results = self.specialized_rag_pipeline.rebuild_all_domain_indexes()
                
                return {
                    "message": "Domain indexes rebuild completed",
                    "results": results
                }
                
            except Exception as e:
                logger.error(f"Error rebuilding all domain indexes: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/system/rebuild-all")
        async def rebuild_all_indexes():
            """Rebuild ALL indexes: general + specialized domains + excel."""
            try:
                results = {}
                
                # 1. Rebuild general LEANN index
                try:
                    general_success = self.indexer.rebuild_leann_index()
                    results["general"] = general_success
                except Exception as e:
                    logger.error(f"Error rebuilding general index: {e}")
                    results["general"] = False
                
                # 2. Rebuild all specialized domain indexes
                try:
                    domain_results = self.specialized_rag_pipeline.rebuild_all_domain_indexes()
                    results.update(domain_results)
                except Exception as e:
                    logger.error(f"Error rebuilding domain indexes: {e}")
                    results["domains"] = False
                
                # 3. Rebuild LlamaIndex Excel
                try:
                    from src.document_processing.llamaindex_persistent_processor import LlamaIndexExcelProcessor
                    llamaindex_processor = LlamaIndexExcelProcessor()
                    excel_success = llamaindex_processor.rebuild_index_from_processed_files()
                    results["excel"] = excel_success
                except Exception as e:
                    logger.error(f"Error rebuilding Excel index: {e}")
                    results["excel"] = False
                
                # Count successes
                success_count = sum(1 for success in results.values() if success)
                total_count = len(results)
                
                return {
                    "message": f"All indexes rebuild completed ({success_count}/{total_count} successful)",
                    "results": results
                }
                
            except Exception as e:
                logger.error(f"Error rebuilding all indexes: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        
        @self.app.get("/documents/{file_name}/domain")
        async def get_document_domain(file_name: str):
            """Get the current domain of a document."""
            try:
                # Use the domain manager to get document domain
                domain = self.specialized_rag_pipeline.pipeline_manager.domain_manager.get_document_domain(file_name)
                
                return {
                    "file_name": file_name,
                    "current_domain": domain
                }
                
            except Exception as e:
                logger.error(f"Error getting document domain: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        
        @self.app.post("/system/reset-index")
        async def reset_index():
            """Reset only the LEANN vector index (preserves all processed documents and uploads)."""
            try:
                success = self.indexer.reset_index()
                
                if success:
                    return {"message": "LEANN index reset successfully (all processed documents and uploads preserved)"}
                else:
                    raise HTTPException(status_code=500, detail="Failed to reset index")
                    
            except Exception as e:
                logger.error(f"Error resetting index: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/system/clear-documents")
        async def clear_documents():
            """Clear LEANN index and all processed documents (preserves uploads)."""
            try:
                # Reset LEANN index only
                self.indexer.reset_index()
                
                # Clear only non-Excel processed documents (preserve Excel files for LlamaIndex)
                processed_dir = Path(settings.DATA_DIR) / "processed"
                non_excel_files_deleted = []
                if processed_dir.exists():
                    for file_path in processed_dir.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() == ".json":
                            try:
                                import json
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    doc_data = json.load(f)
                                # Only delete non-Excel files
                                if not doc_data.get('metadata', {}).get('filename', '').lower().endswith('.xlsx'):
                                    file_path.unlink()
                                    non_excel_files_deleted.append(file_path.name)
                            except Exception:
                                # If we can't read the file, delete it to be safe
                                file_path.unlink()
                                non_excel_files_deleted.append(file_path.name)
                
                return {
                    "message": "LEANN index and non-Excel processed documents cleared successfully (Excel files preserved for LlamaIndex)",
                    "deleted_files": non_excel_files_deleted,
                    "preserved_excel_files": "Excel files preserved for LlamaIndex management"
                }
                
            except Exception as e:
                logger.error(f"Error clearing documents: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/system/rebuild-index")
        async def rebuild_index():
            """Rebuild LEANN index from existing processed documents (no reprocessing)."""
            try:
                # Rebuild LEANN index from existing processed documents
                success = self.indexer.rebuild_leann_index()
                
                if success:
                    return {"message": "LEANN index rebuilt successfully from existing processed documents"}
                else:
                    raise HTTPException(status_code=500, detail="Failed to rebuild LEANN index")
                    
            except Exception as e:
                logger.error(f"Error rebuilding LEANN index: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/system/rebuild-llamaindex")
        async def rebuild_llamaindex():
            """Rebuild LlamaIndex Excel index from existing processed Excel files (no reprocessing)."""
            try:
                from src.document_processing.llamaindex_persistent_processor import LlamaIndexExcelProcessor
                
                # Rebuild LlamaIndex from existing processed Excel files
                excel_processor = LlamaIndexExcelProcessor()
                success = excel_processor.rebuild_index_from_processed_files()
                
                if success:
                    return {"message": "LlamaIndex Excel index rebuilt successfully from existing processed Excel files"}
                else:
                    raise HTTPException(status_code=500, detail="Failed to rebuild LlamaIndex index")
                    
            except Exception as e:
                logger.error(f"Error rebuilding LlamaIndex: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/system/reset-llamaindex")
        async def reset_llamaindex():
            """Reset only the LlamaIndex Excel index (preserves all processed Excel files and uploads)."""
            try:
                from src.document_processing.llamaindex_persistent_processor import LlamaIndexExcelProcessor
                
                # Reset LlamaIndex Excel index only
                excel_index_path = Path("data/llamaindex_excel_index")
                if excel_index_path.exists():
                    shutil.rmtree(excel_index_path)
                    excel_index_path.mkdir(exist_ok=True)
                
                return {
                    "message": "LlamaIndex Excel index reset successfully (all processed Excel files and uploads preserved)",
                    "deleted_processed_files": [],
                    "deleted_upload_files": []
                }
                
            except Exception as e:
                logger.error(f"Error resetting LlamaIndex: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/system/clear-llamaindex")
        async def clear_llamaindex():
            """Clear LlamaIndex Excel index and processed Excel files (preserves uploads)."""
            try:
                from src.document_processing.llamaindex_persistent_processor import LlamaIndexExcelProcessor
                
                # Clear LlamaIndex Excel index
                excel_index_path = Path("data/llamaindex_excel_index")
                if excel_index_path.exists():
                    shutil.rmtree(excel_index_path)
                    excel_index_path.mkdir(exist_ok=True)
                
                # Clear processed Excel files only
                processed_dir = Path(settings.DATA_DIR) / "processed"
                excel_files_deleted = []
                if processed_dir.exists():
                    for file_path in processed_dir.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() == ".json":
                            try:
                                import json
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    doc_data = json.load(f)
                                if doc_data.get('metadata', {}).get('filename', '').lower().endswith('.xlsx'):
                                    file_path.unlink()
                                    excel_files_deleted.append(file_path.name)
                            except Exception:
                                continue
                
                return {
                    "message": "LlamaIndex Excel index and all processed Excel files cleared successfully (uploads preserved)",
                    "deleted_processed_files": excel_files_deleted,
                    "deleted_upload_files": []  # No uploads deleted
                }
                
            except Exception as e:
                logger.error(f"Error clearing LlamaIndex: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/system/llamaindex-status")
        async def get_llamaindex_status():
            """Get LlamaIndex Excel index status and statistics."""
            try:
                from src.document_processing.llamaindex_persistent_processor import LlamaIndexExcelProcessor
                
                excel_index_path = Path("data/llamaindex_excel_index")
                index_exists = excel_index_path.exists() and any(excel_index_path.iterdir())
                
                # Count Excel files in uploads
                uploads_dir = Path(settings.DATA_DIR) / "uploads"
                excel_upload_count = 0
                excel_upload_files = []
                if uploads_dir.exists():
                    for file_path in uploads_dir.iterdir():
                        if file_path.suffix.lower() == '.xlsx':
                            excel_upload_count += 1
                            excel_upload_files.append(file_path.name)
                
                # Count processed Excel files
                processed_dir = Path(settings.DATA_DIR) / "processed"
                excel_processed_count = 0
                excel_processed_files = []
                if processed_dir.exists():
                    for file_path in processed_dir.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() == ".json":
                            try:
                                import json
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    doc_data = json.load(f)
                                if doc_data.get('metadata', {}).get('filename', '').lower().endswith('.xlsx'):
                                    excel_processed_count += 1
                                    excel_processed_files.append(file_path.name)
                            except Exception:
                                continue
                
                return {
                    "index_exists": index_exists,
                    "index_path": str(excel_index_path),
                    "excel_upload_count": excel_upload_count,
                    "excel_upload_files": excel_upload_files,
                    "excel_processed_count": excel_processed_count,
                    "excel_processed_files": excel_processed_files,
                    "total_excel_files": excel_upload_count + excel_processed_count
                }
                
            except Exception as e:
                logger.error(f"Error getting LlamaIndex status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/documents/{file_name}")
        async def delete_document(file_name: str):
            """Delete a specific document and its processed data."""
            try:
                from urllib.parse import unquote
                file_name = unquote(file_name)
                
                # Find and delete the original file
                uploads_dir = Path(settings.DATA_DIR) / "uploads"
                processed_dir = Path(settings.DATA_DIR) / "processed"
                original_file = None
                if uploads_dir.exists():
                    for file_path in uploads_dir.iterdir():
                        if file_path.name == file_name:
                            original_file = file_path
                            break
                
                # Check if we have either the original file or a processed file
                processed_file_exists = False
                if processed_dir.exists():
                    for file_path in processed_dir.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() == ".json":
                            try:
                                import json
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    doc_data = json.load(f)
                                if doc_data.get('metadata', {}).get('filename') == file_name:
                                    processed_file_exists = True
                                    break
                            except Exception:
                                continue
                
                if (original_file is None or not original_file.exists()) and not processed_file_exists:
                    raise HTTPException(status_code=404, detail=f"Document '{file_name}' not found")
                
                # Find the processed file (we already searched above)
                processed_file = None
                if processed_dir.exists():
                    for file_path in processed_dir.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() == ".json":
                            try:
                                import json
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    doc_data = json.load(f)
                                if doc_data.get('metadata', {}).get('filename') == file_name:
                                    processed_file = file_path
                                    break
                            except Exception:
                                continue
                
                # Delete files
                deleted_files = []
                if original_file and original_file.exists():
                    original_file.unlink()
                    deleted_files.append(f"Original file: {file_name}")
                
                if processed_file and processed_file.exists():
                    processed_file.unlink()
                    deleted_files.append(f"Processed file: {processed_file.name}")
                
                # Remove from appropriate index based on file type
                try:
                    if file_name.lower().endswith('.xlsx'):
                        # Remove from LlamaIndex Excel index
                        logger.info(f"Removing Excel file from LlamaIndex: {file_name}")
                        # For now, we'll need to rebuild the LlamaIndex since it doesn't have individual document removal
                        # In the future, this could be improved with proper document removal
                        from src.document_processing.llamaindex_persistent_processor import LlamaIndexExcelProcessor
                        excel_processor = LlamaIndexExcelProcessor()
                        # Clear and rebuild Excel index
                        import shutil
                        excel_index_path = Path("data/llamaindex_excel_index")
                        if excel_index_path.exists():
                            shutil.rmtree(excel_index_path)
                            excel_index_path.mkdir(exist_ok=True)
                        
                        # Rebuild Excel index with remaining files
                        uploads_dir = Path(settings.DATA_DIR) / "uploads"
                        if uploads_dir.exists():
                            for remaining_file in uploads_dir.iterdir():
                                if remaining_file.suffix.lower() == '.xlsx' and remaining_file.name != file_name:
                                    try:
                                        # Process and index the remaining Excel file
                                        from src.document_processing.document_processor import DocumentProcessor
                                        doc_processor = DocumentProcessor()
                                        processed_doc = doc_processor.process_document(remaining_file)
                                        excel_processor.process_and_index_excel(remaining_file, processed_doc.content, processed_doc.metadata)
                                    except Exception as e:
                                        logger.warning(f"Error re-indexing Excel file {remaining_file}: {e}")
                                        continue
                        
                        deleted_files.append("Removed from LlamaIndex Excel index")
                    else:
                        # Remove from LEANN index by rebuilding it
                        logger.info(f"Removing non-Excel file from LEANN index: {file_name}")
                        self.indexer.reset_index()
                        # Re-index remaining non-Excel documents
                        if processed_dir.exists():
                            for file_path in processed_dir.iterdir():
                                if file_path.is_file() and file_path.suffix.lower() == ".json":
                                    try:
                                        # Check if it's not an Excel file
                                        with open(file_path, 'r', encoding='utf-8') as f:
                                            doc_data = json.load(f)
                                        if not doc_data.get('filename', '').lower().endswith('.xlsx'):
                                            self.indexer.index_processed_document(file_path)
                                    except Exception as e:
                                        logger.warning(f"Error re-indexing {file_path}: {e}")
                                        continue
                        
                        deleted_files.append("Removed from LEANN index")
                        
                except Exception as e:
                    logger.warning(f"Error updating index after deletion: {e}")
                    deleted_files.append(f"Warning: Index update failed - {str(e)}")
                
                return {
                    "message": f"Document '{file_name}' deleted successfully",
                    "deleted_files": deleted_files
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error deleting document {file_name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/system/processing-status")
        async def get_processing_status():
            """Get current processing status."""
            try:
                # Check if any processing is currently happening
                # For now, return a simple status
                # In the future, this could track actual processing state
                return {
                    "status": "idle",
                    "message": "No documents currently being processed",
                    "timestamp": str(datetime.now()),
                    "processed_documents": len([f for f in (Path(settings.DATA_DIR) / "processed").iterdir() if f.suffix == ".json"]) if (Path(settings.DATA_DIR) / "processed").exists() else 0,
                    "uploaded_documents": len([f for f in (Path(settings.DATA_DIR) / "uploads").iterdir() if f.is_file()]) if (Path(settings.DATA_DIR) / "uploads").exists() else 0
                }
            except Exception as e:
                logger.error(f"Error getting processing status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/system/models")
        async def get_available_models():
            """Get list of available Ollama models."""
            try:
                models = self.ollama_client.list_models()
                return {
                    "models": models,
                    "default_model": self.ollama_client.default_model,
                    "current_model": self.ollama_client.current_model if hasattr(self.ollama_client, 'current_model') else self.ollama_client.default_model
                }
            except Exception as e:
                logger.error(f"Error getting available models: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/system/change-model")
        async def change_model(request: ModelChangeRequest):
            """Change the current Ollama model."""
            try:
                success = self.ollama_client.change_model(request.model_name)
                if success:
                    return {
                        "message": f"Model changed successfully to {request.model_name}",
                        "current_model": request.model_name
                    }
                else:
                    raise HTTPException(status_code=400, detail=f"Failed to change model to {request.model_name}")
            except Exception as e:
                logger.error(f"Error changing model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/system/vector-store")
        async def get_vector_store_info():
            """Get information about all LEANN vector stores."""
            try:
                # Get main collection info
                main_store_info = self.vector_store.get_collection_info()
                main_store_info.update({
                    "store_type": "LEANN",
                    "is_leann": True,
                    "use_leann_setting": True
                })
                
                # Get domain statistics for all collections
                domain_stats = self.specialized_rag_pipeline.get_domain_statistics()
                
                # Create comprehensive store info
                all_stores = {
                    "main_collection": main_store_info,
                    "domain_collections": {}
                }
                
                # Add domain collection info
                for domain, stats in domain_stats.items():
                    if "error" not in stats:
                        all_stores["domain_collections"][domain] = {
                            "index_name": f"{domain}_collection",
                            "document_count": stats.get("document_count", 0),
                            "chunk_count": stats.get("chunk_count", 0),
                            "is_initialized": stats.get("is_initialized", False),
                            "store_type": "LEANN",
                            "is_leann": True,
                            "use_leann_setting": True
                        }
                
                # Calculate totals
                total_documents = main_store_info.get("document_count", 0) + sum(
                    stats.get("document_count", 0) for stats in all_stores["domain_collections"].values()
                )
                total_chunks = main_store_info.get("chunk_count", 0) + sum(
                    stats.get("chunk_count", 0) for stats in all_stores["domain_collections"].values()
                )
                
                all_stores["summary"] = {
                    "total_collections": 1 + len(all_stores["domain_collections"]),
                    "total_documents": total_documents,
                    "total_chunks": total_chunks,
                    "embedding_model": main_store_info.get("embedding_model", "unknown"),
                    "backend": main_store_info.get("backend", "unknown")
                }
                
                return all_stores
            except Exception as e:
                logger.error(f"Error getting LEANN vector store info: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        

        
        @self.app.delete("/system/clear-all")
        async def clear_all():
            """Clear everything: LEANN index + LlamaIndex + all documents + all uploads."""
            try:
                # Reset LEANN index
                self.indexer.reset_index()
                
                # Clear LlamaIndex Excel index
                excel_index_path = Path("data/llamaindex_excel_index")
                if excel_index_path.exists():
                    shutil.rmtree(excel_index_path)
                    excel_index_path.mkdir(exist_ok=True)
                
                # Clear all data directories
                data_dir = Path(settings.DATA_DIR)
                for subdir in ["processed", "uploads"]:
                    subdir_path = data_dir / subdir
                    if subdir_path.exists():
                        shutil.rmtree(subdir_path)
                        subdir_path.mkdir(parents=True, exist_ok=True)
                
                return {"message": "Everything cleared: LEANN + LlamaIndex + all processed documents + all uploads"}
                
            except Exception as e:
                logger.error(f"Error clearing all data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/system/clear-general")
        async def clear_general():
            """Clear general LEANN index and all processed documents."""
            try:
                # Reset general LEANN index
                self.indexer.reset_index()
                
                # Clear all processed documents
                processed_dir = Path(settings.DATA_DIR) / "processed"
                files_deleted = []
                if processed_dir.exists():
                    for file_path in processed_dir.iterdir():
                        if file_path.is_file():
                            file_path.unlink()
                            files_deleted.append(file_path.name)
                
                return {"message": f"General index cleared: {len(files_deleted)} processed files deleted"}
                
            except Exception as e:
                logger.error(f"Error clearing general index: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/system/clear-domain/{domain}")
        async def clear_domain(domain: str):
            """Clear specific domain index and its processed documents."""
            try:
                # Reset domain index
                success = self.specialized_rag_pipeline.reset_domain_index(domain)
                if not success:
                    raise HTTPException(status_code=500, detail=f"Failed to reset {domain} domain index")
                
                # Clear domain-specific processed documents
                processed_dir = Path(settings.DATA_DIR) / "processed"
                files_deleted = []
                if processed_dir.exists():
                    for file_path in processed_dir.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() == ".json":
                            try:
                                import json
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    doc_data = json.load(f)
                                
                                # Check if document belongs to this domain
                                doc_domain = doc_data.get('domain', 'general')
                                if doc_domain == domain:
                                    file_path.unlink()
                                    files_deleted.append(file_path.name)
                            except Exception as e:
                                logger.warning(f"Could not read domain from {file_path}: {e}")
                                continue
                
                return {"message": f"{domain.title()} domain cleared: {len(files_deleted)} processed files deleted"}
                
            except Exception as e:
                logger.error(f"Error clearing {domain} domain: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/system/clear-excel")
        async def clear_excel():
            """Clear Excel index and processed Excel files."""
            try:
                # Clear LlamaIndex Excel index
                excel_index_path = Path("data/llamaindex_excel_index")
                if excel_index_path.exists():
                    shutil.rmtree(excel_index_path)
                    excel_index_path.mkdir(exist_ok=True)
                
                # Clear Excel processed files
                processed_dir = Path(settings.DATA_DIR) / "processed"
                files_deleted = []
                if processed_dir.exists():
                    for file_path in processed_dir.iterdir():
                        if file_path.is_file() and file_path.suffix.lower() == ".json":
                            try:
                                import json
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    doc_data = json.load(f)
                                
                                # Check if it's an Excel file
                                filename = doc_data.get('metadata', {}).get('filename', '')
                                if filename.lower().endswith('.xlsx'):
                                    file_path.unlink()
                                    files_deleted.append(file_path.name)
                            except Exception as e:
                                logger.warning(f"Could not read file {file_path}: {e}")
                                continue
                
                return {"message": f"Excel index cleared: {len(files_deleted)} Excel processed files deleted"}
                
            except Exception as e:
                logger.error(f"Error clearing Excel index: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        
        @self.app.post("/system/reprocess-all-documents")
        async def reprocess_all_documents():
            """Force reprocessing of all uploaded documents to LEANN vector store."""
            try:
                logger.info("Starting reprocessing of all documents")
                
                # Get all uploaded files
                uploaded_files = []
                for ext in settings.SUPPORTED_FORMATS:
                    uploaded_files.extend(settings.UPLOADS_DIR.glob(f"*{ext}"))
                
                if not uploaded_files:
                    return {
                        "message": "No uploaded documents found",
                        "processed_count": 0
                    }
                
                logger.info(f"Found {len(uploaded_files)} uploaded documents to reprocess")
                
                processed_count = 0
                failed_count = 0
                
                for file_path in uploaded_files:
                    try:
                        logger.info(f"Reprocessing document: {file_path.name}")
                        
                        # Process and index the document
                        success = self.indexer.index_document(file_path)
                        if success:
                            processed_count += 1
                            logger.info(f"✅ Successfully reprocessed: {file_path.name}")
                        else:
                            failed_count += 1
                            logger.warning(f"⚠️ Failed to reprocess: {file_path.name}")
                            
                    except Exception as e:
                        failed_count += 1
                        logger.error(f"Error reprocessing {file_path.name}: {e}")
                        continue
                
                logger.info(f"Reprocessing completed. Processed: {processed_count}, Failed: {failed_count}")
                
                return {
                    "message": f"Reprocessing completed. Processed {processed_count} documents, {failed_count} failed",
                    "processed_count": processed_count,
                    "failed_count": failed_count,
                    "total_files": len(uploaded_files)
                }
                
            except Exception as e:
                logger.error(f"Error during reprocessing: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/system/force-initialize-searcher")
        async def force_initialize_searcher():
            """Force initialization of the LEANN searcher."""
            try:
                self.vector_store.force_initialize_searcher()
                return {"message": "LEANN searcher initialization forced"}
            except Exception as e:
                logger.error(f"Error forcing searcher initialization: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(self, host: str = "0.0.0.0", port: int = 8199):
        """Run the FastAPI server."""
        logger.info(f"Starting FastAPI server on {host}:{port}")
        uvicorn.run(
            self.app, 
            host=host, 
            port=port,
            timeout_keep_alive=settings.UPLOAD_TIMEOUT,  # Use configured timeout
            timeout_graceful_shutdown=settings.SYSTEM_OPERATION_TIMEOUT  # Use configured timeout
        )


# Create global instance
api_backend = FastAPIBackend()
app = api_backend.app
