#!/usr/bin/env python3
"""
LlamaIndex Excel processor with persistence for enhanced Excel document processing.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from llama_index.core import Settings, Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.document_processing.document_processor import ProcessedDocument

logger = logging.getLogger(__name__)


class LlamaIndexExcelProcessor:
    """LlamaIndex-based Excel processor with persistence."""
    
    def __init__(self):
        """Initialize the LlamaIndex Excel processor."""
        self.index_path = Path("data/llamaindex_excel_index")
        self.index_path.mkdir(exist_ok=True)
        
        # Initialize LlamaIndex settings
        self._setup_llamaindex()
        
        logger.info(f"LlamaIndex Excel processor initialized with persistence at: {self.index_path}")
    
    def _setup_llamaindex(self):
        """Setup LlamaIndex with Ollama and embedding model."""
        # Set up LlamaIndex with Ollama
        Settings.llm = Ollama(
            model='llama3.2:3b',
            base_url='http://localhost:11434',
            request_timeout=120.0,
            temperature=0.1,
            max_tokens=150,
            top_p=0.9
        )
        Settings.embed_model = HuggingFaceEmbedding(
            model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            trust_remote_code=False
        )
    
    def process_and_index_excel(self, file_path: Path, content: str, metadata: Dict[str, Any]) -> bool:
        """Process and index Excel file with LlamaIndex persistence."""
        try:
            logger.info(f"Processing and indexing Excel with LlamaIndex: {file_path}")
            
            # Load or create index
            index = self._load_or_create_index()
            
            # Process Excel content using the existing Excel processor
            excel_chunks = self._process_excel_content_with_existing_processor(file_path, content, metadata)
            
            # Add to index
            for chunk in excel_chunks:
                index.insert(chunk)
            
            # Save index to disk
            self._save_index(index)
            
            logger.info(f"Successfully indexed Excel with LlamaIndex: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing Excel with LlamaIndex {file_path}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _load_or_create_index(self):
        """Load existing index or create new one."""
        try:
            if self._index_exists():
                return self._load_persistent_index()
            else:
                return self._create_new_index()
        except Exception as e:
            logger.error(f"Error loading/creating index: {e}")
            return self._create_new_index()
    
    def _index_exists(self) -> bool:
        """Check if persistent index exists."""
        return (self.index_path / "index.json").exists()
    
    def _load_persistent_index(self):
        """Load index from disk."""
        storage_context = StorageContext.from_defaults(persist_dir=str(self.index_path))
        index = load_index_from_storage(storage_context)
        logger.info(f"Loaded persistent LlamaIndex from: {self.index_path}")
        return index
    
    def _create_new_index(self):
        """Create new LlamaIndex."""
        # Create empty index
        index = VectorStoreIndex([])
        logger.info("Created new LlamaIndex")
        return index
    
    def _save_index(self, index):
        """Save index to disk."""
        index.storage_context.persist(persist_dir=str(self.index_path))
        logger.info(f"Saved LlamaIndex to: {self.index_path}")
    
    def rebuild_index_from_processed_files(self) -> bool:
        """Rebuild LlamaIndex from existing processed Excel files (no reprocessing)."""
        try:
            logger.info("Rebuilding LlamaIndex from existing processed Excel files...")
            
            # Load all processed Excel files
            processed_dir = Path("data/processed")
            if not processed_dir.exists():
                logger.warning("No processed documents directory found")
                return False
            
            # Find all Excel processed files
            excel_processed_files = []
            for file_path in processed_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == ".json":
                    try:
                        import json
                        with open(file_path, 'r', encoding='utf-8') as f:
                            doc_data = json.load(f)
                        # Only process Excel files for LlamaIndex
                        if doc_data.get('metadata', {}).get('filename', '').lower().endswith('.xlsx'):
                            excel_processed_files.append((file_path, doc_data))
                    except Exception as e:
                        logger.warning(f"Could not load processed Excel file {file_path}: {e}")
                        continue
            
            if not excel_processed_files:
                logger.warning("No processed Excel files found for LlamaIndex rebuild")
                return False
            
            # Create new index
            index = self._create_new_index()
            
            # Rebuild index from processed Excel files
            success_count = 0
            for file_path, doc_data in excel_processed_files:
                try:
                    # Convert processed data back to ProcessedDocument format
                    processed_doc = ProcessedDocument(
                        file_path=Path(doc_data['file_path']),
                        content=doc_data['content'],
                        metadata=doc_data['metadata'],
                        chunks=doc_data.get('chunks', [])
                    )
                    
                    # Process Excel content using the existing Excel processor
                    excel_chunks = self._process_excel_content_with_existing_processor(
                        processed_doc.file_path, 
                        processed_doc.content, 
                        processed_doc.metadata
                    )
                    
                    # Add to index
                    for chunk in excel_chunks:
                        index.insert(chunk)
                    
                    success_count += 1
                    logger.info(f"Rebuilt LlamaIndex for: {processed_doc.file_path}")
                    
                except Exception as e:
                    logger.error(f"Error rebuilding LlamaIndex for {file_path}: {e}")
                    continue
            
            # Save the rebuilt index
            self._save_index(index)
            
            logger.info(f"LlamaIndex rebuilt successfully: {success_count}/{len(excel_processed_files)} Excel files indexed")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error rebuilding LlamaIndex: {e}")
            return False
    
    def _process_excel_content_with_existing_processor(self, file_path: Path, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Process Excel content using the existing Excel processor."""
        from src.document_processing.excel_processor import ExcelProcessor
        
        # Use the existing Excel processor with the content already extracted
        excel_processor = ExcelProcessor()
        excel_chunks = excel_processor.process_excel_file(file_path, content)
        
        # Convert Excel chunks to LlamaIndex documents
        documents = []
        for i, chunk in enumerate(excel_chunks):
            doc = Document(
                text=chunk.text,
                metadata={
                    **chunk.metadata,
                    'chunk_index': i,
                    'file_type': 'excel',
                    'processing_method': 'llamaindex',
                    'file_path': str(file_path)
                }
            )
            documents.append(doc)
        
        return documents
    
    def _process_excel_content(self, file_path: Path, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Process Excel content into LlamaIndex documents."""
        documents = []
        
        # Split content into chunks
        chunks = content.split('\n\n')  # Simple chunking for now
        
        for i, chunk_text in enumerate(chunks):
            if chunk_text.strip():
                doc = Document(
                    text=chunk_text,
                    metadata={
                        **metadata,
                        'chunk_index': i,
                        'file_type': 'excel',
                        'processing_method': 'llamaindex'
                    }
                )
                documents.append(doc)
        
        return documents
    
    def rebuild_excel_index(self) -> bool:
        """Rebuild the Excel index from existing processed Excel documents."""
        try:
            self.logger.info("Rebuilding Excel index from processed documents...")
            
            # Find all processed Excel documents
            processed_dir = Path("data/processed")
            if not processed_dir.exists():
                self.logger.warning("No processed documents directory found")
                return False
            
            excel_files = []
            for file_path in processed_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == ".json" and file_path.name.endswith("_processed.json"):
                    try:
                        import json
                        with open(file_path, 'r', encoding='utf-8') as f:
                            doc_data = json.load(f)
                        
                        # Only process Excel files
                        if doc_data.get('metadata', {}).get('filename', '').lower().endswith('.xlsx'):
                            excel_files.append((file_path, doc_data))
                    except Exception as e:
                        self.logger.warning(f"Could not load processed Excel file {file_path}: {e}")
                        continue
            
            if not excel_files:
                self.logger.info("No processed Excel files found")
                return True
            
            # Rebuild index with all Excel files
            success = self._rebuild_index_from_files(excel_files)
            
            if success:
                self.logger.info(f"Successfully rebuilt Excel index with {len(excel_files)} files")
            else:
                self.logger.error("Failed to rebuild Excel index")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error rebuilding Excel index: {e}")
            return False
    
    def _rebuild_index_from_files(self, excel_files: List[Tuple[Path, Dict]]) -> bool:
        """Rebuild index from a list of processed Excel files."""
        try:
            # Clear existing index
            if hasattr(self, 'vector_store') and self.vector_store:
                # Clear the existing index
                self.logger.info("Clearing existing Excel index...")
                # Note: This would need to be implemented based on the specific vector store
            
            # Re-index all Excel files
            for file_path, doc_data in excel_files:
                try:
                    # Create ProcessedDocument from stored data
                    from src.document_processing.document_processor import ProcessedDocument
                    processed_doc = ProcessedDocument(
                        file_path=Path(doc_data['file_path']),
                        content=doc_data['content'],
                        metadata=doc_data['metadata'],
                        chunks=doc_data.get('chunks', [])
                    )
                    
                    # Process and index the document
                    success = self.process_and_index_excel(
                        file_path=processed_doc.file_path,
                        content=processed_doc.content,
                        metadata=processed_doc.metadata
                    )
                    
                    if not success:
                        self.logger.warning(f"Failed to re-index Excel file: {processed_doc.file_path}")
                        
                except Exception as e:
                    self.logger.warning(f"Error re-indexing Excel file {file_path}: {e}")
                    continue
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error rebuilding index from files: {e}")
            return False