"""
Document Indexer for processing and indexing documents into the vector store.
"""
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from loguru import logger

from src.document_processing.document_processor import ProcessedDocument, DocumentProcessor
from .unified_vector_store import UnifiedVectorStore
from config.settings import settings
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata for vector storage."""
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class DocumentIndexer:
    """Handles document indexing into the vector store."""
    
    def __init__(self, vector_store: UnifiedVectorStore, document_processor: DocumentProcessor):
        """Initialize the document indexer."""
        self.vector_store = vector_store
        self.document_processor = document_processor
        
        # Initialize LlamaIndex Excel processor for Excel files
        from src.document_processing.llamaindex_persistent_processor import LlamaIndexExcelProcessor
        self.llamaindex_processor = LlamaIndexExcelProcessor()
        
        # Determine store type for logging
        if hasattr(vector_store, 'get_store_type'):
            store_type = vector_store.get_store_type()
        else:
            store_type = "LEANN"
        
        logger.info(f"Document indexer initialized with {store_type} + LlamaIndex for Excel")
    
    def index_document(self, file_path: Path) -> bool:
        """Index a single document into the appropriate vector store."""
        try:
            logger.info(f"Indexing document: {file_path}")
            
            # Process the document
            processed_doc = self.document_processor.process_document(file_path)
            
            # Check if it's an Excel file
            if file_path.suffix.lower() == '.xlsx':
                logger.info(f"Excel file detected, using LlamaIndex: {file_path}")
                return self._index_excel_with_llamaindex(processed_doc)
            else:
                logger.info(f"Non-Excel file, using LEANN: {file_path}")
                return self._index_with_leann(processed_doc)
                
        except Exception as e:
            logger.error(f"Error indexing document {file_path}: {e}")
            return False
    
    def _index_excel_with_llamaindex(self, processed_doc: ProcessedDocument) -> bool:
        """Index Excel document using LlamaIndex with persistence."""
        try:
            logger.info(f"Indexing Excel with LlamaIndex: {processed_doc.file_path}")
            
            # Use LlamaIndex Excel processor
            success = self.llamaindex_processor.process_and_index_excel(
                file_path=processed_doc.file_path,
                content=processed_doc.content,
                metadata=processed_doc.metadata
            )
            
            if success:
                # Save processed document to disk
                try:
                    processed_dir = Path(settings.DATA_DIR) / "processed"
                    self.document_processor.save_processed_document(processed_doc, processed_dir)
                    logger.info(f"Excel indexed with LlamaIndex and saved to disk: {processed_doc.file_path}")
                except Exception as save_error:
                    logger.warning(f"Failed to save processed Excel to disk: {save_error}")
                
                return True
            else:
                logger.error(f"Failed to index Excel with LlamaIndex: {processed_doc.file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error indexing Excel with LlamaIndex {processed_doc.file_path}: {e}")
            return False
    
    def _index_with_leann(self, processed_doc: ProcessedDocument) -> bool:
        """Index non-Excel document using LEANN."""
        try:
            logger.info(f"Indexing with LEANN: {processed_doc.file_path}")
            
            # Convert to document chunks
            chunks = self._create_document_chunks(processed_doc)
            
            # Convert DocumentChunk objects to dictionaries for LEANN
            chunk_dicts = []
            for chunk in chunks:
                chunk_dict = {
                    'text': chunk.text,
                    'metadata': chunk.metadata
                }
                chunk_dicts.append(chunk_dict)
            
            # Add to LEANN vector store
            success = self.vector_store.add_documents(chunk_dicts)
            
            # Build LEANN index
            if success and hasattr(self.vector_store, 'build_index'):
                build_success = self.vector_store.build_index()
                if not build_success:
                    logger.warning("Failed to build LEANN index")
            
            if success:
                # Save processed document to disk
                try:
                    processed_dir = Path(settings.DATA_DIR) / "processed"
                    self.document_processor.save_processed_document(processed_doc, processed_dir)
                    logger.info(f"Document indexed with LEANN and saved to disk: {processed_doc.file_path}")
                except Exception as save_error:
                    logger.warning(f"Failed to save processed document to disk: {save_error}")
                
                return True
            else:
                logger.error(f"Failed to index document with LEANN: {processed_doc.file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error indexing with LEANN {processed_doc.file_path}: {e}")
            return False
    
    def index_processed_document(self, processed_doc: ProcessedDocument) -> bool:
        """Index an already processed document into the vector store."""
        try:
            logger.info(f"Indexing processed document: {processed_doc.file_path}")
            
            # Convert to document chunks
            chunks = self._create_document_chunks(processed_doc)
            
            # Convert DocumentChunk objects to dictionaries for LEANN
            chunk_dicts = []
            for chunk in chunks:
                chunk_dict = {
                    'text': chunk.text,
                    'metadata': chunk.metadata
                }
                chunk_dicts.append(chunk_dict)
            
            # Add to vector store
            success = self.vector_store.add_documents(chunk_dicts)
            
            # Build index if using LEANN
            if success and hasattr(self.vector_store, 'build_index'):
                build_success = self.vector_store.build_index()
                if not build_success:
                    logger.warning("Failed to build LEANN index")
            
            if success:
                # Save processed document to disk for the documents endpoint
                try:
                    processed_dir = Path(settings.DATA_DIR) / "processed"
                    self.document_processor.save_processed_document(processed_doc, processed_dir)
                    logger.info(f"Saved processed document to disk: {processed_doc.file_path}")
                except Exception as save_error:
                    logger.warning(f"Failed to save processed document to disk: {save_error}")
                
                logger.info(f"Successfully indexed processed document: {processed_doc.file_path}")
                return True
            else:
                logger.error(f"Failed to index processed document: {processed_doc.file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error indexing processed document {processed_doc.file_path}: {e}")
            return False
    
    def index_directory(self, directory_path: Path) -> Dict[str, bool]:
        """Index all documents in a directory."""
        try:
            logger.info(f"Indexing directory: {directory_path} (type: {type(directory_path)})")
            
            # Ensure directory_path is a Path object
            if not isinstance(directory_path, Path):
                directory_path = Path(directory_path)
                logger.info(f"Converted to Path: {directory_path}")
            
            # Process all documents
            processed_docs = self.document_processor.process_directory(directory_path)
            
            # Index each processed document using the main index_document method
            results = {}
            for processed_doc in processed_docs:
                # Use index_document to ensure proper routing (Excel -> LlamaIndex, others -> LEANN)
                success = self.index_document(processed_doc.file_path)
                results[str(processed_doc.file_path)] = success
            
            # Log summary
            successful = sum(results.values())
            total = len(results)
            logger.info(f"Directory indexing complete: {successful}/{total} documents indexed successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Error indexing directory {directory_path}: {e}")
            return {}
    
    def _create_document_chunks(self, processed_doc: ProcessedDocument) -> List[DocumentChunk]:
        """Convert a processed document to document chunks for vector storage."""
        chunks = []
        
        for i, chunk_text in enumerate(processed_doc.chunks):
            # Create unique ID for the chunk
            chunk_id = f"{processed_doc.file_path.stem}_{i}_{uuid.uuid4().hex[:8]}"
            
            # Prepare metadata
            metadata = {
                "file_name": processed_doc.file_path.name,
                "file_path": str(processed_doc.file_path),
                "file_extension": processed_doc.file_path.suffix.lower(),
                "file_size": processed_doc.metadata.get("file_size", 0),
                "chunk_index": i,
                "total_chunks": len(processed_doc.chunks),
                "chunk_size": len(chunk_text),
                "content_length": processed_doc.metadata.get("content_length", 0),
                "indexing_timestamp": datetime.now().isoformat(),
                "source_document": str(processed_doc.file_path)
            }
            
            # Create document chunk
            chunk = DocumentChunk(
                id=chunk_id,
                text=chunk_text,
                metadata=metadata
            )
            
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} document chunks from {processed_doc.file_path}")
        return chunks
    
    def search_documents(self, query: str, n_results: int = 5, 
                        filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for documents in the vector store."""
        return self.vector_store.search(query, n_results, filter_metadata=filter_metadata)
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the indexed documents."""
        if hasattr(self.vector_store, 'get_collection_info'):
            return self.vector_store.get_collection_info()
        else:
            # Unified vector store
            return self.vector_store.get_collection_info()
    
    def reset_index(self) -> bool:
        """Reset the entire index."""
        logger.warning("Resetting document index - this will delete all indexed documents!")
        if hasattr(self.vector_store, 'reset_collection'):
            return self.vector_store.reset_collection()
        else:
            # Unified vector store
            return self.vector_store.reset_collection()
    
    def rebuild_leann_index(self) -> bool:
        """Rebuild LEANN index from existing processed documents (no reprocessing)."""
        try:
            logger.info("Rebuilding LEANN index from existing processed documents...")
            
            # Load all processed documents
            processed_dir = Path(settings.DATA_DIR) / "processed"
            if not processed_dir.exists():
                logger.warning("No processed documents directory found")
                return False
            
            # Find all non-Excel processed documents
            processed_docs = []
            for file_path in processed_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == ".json":
                    try:
                        import json
                        with open(file_path, 'r', encoding='utf-8') as f:
                            doc_data = json.load(f)
                        # Only process non-Excel files for LEANN
                        if not doc_data.get('metadata', {}).get('filename', '').lower().endswith('.xlsx'):
                            processed_doc = ProcessedDocument(
                                file_path=Path(doc_data['file_path']),
                                content=doc_data['content'],
                                metadata=doc_data['metadata'],
                                chunks=doc_data.get('chunks', [])
                            )
                            processed_docs.append(processed_doc)
                    except Exception as e:
                        logger.warning(f"Could not load processed document {file_path}: {e}")
                        continue
            
            if not processed_docs:
                logger.warning("No non-Excel processed documents found for LEANN rebuild")
                return False
            
            # Reset LEANN index first
            self.vector_store.reset_collection()
            
            # Rebuild index from processed documents
            success_count = 0
            for processed_doc in processed_docs:
                if self._index_with_leann(processed_doc):
                    success_count += 1
            
            logger.info(f"LEANN index rebuilt successfully: {success_count}/{len(processed_docs)} documents indexed")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error rebuilding LEANN index: {e}")
            return False
    
    def remove_document(self, file_path: str) -> bool:
        """Remove a specific document from the index."""
        try:
            # This would require implementing deletion by metadata filtering
            # For now, we'll log that this feature needs to be implemented
            logger.warning(f"Document removal not yet implemented for: {file_path}")
            return False
        except Exception as e:
            logger.error(f"Error removing document {file_path}: {e}")
            return False
