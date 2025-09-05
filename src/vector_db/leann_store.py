"""
LEANN Vector Store Implementation for Myr-Ag RAG System.

This module provides a LEANN-based vector store that can be used as an alternative
for document storage and retrieval with significant storage savings.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from leann.api import LeannBuilder, LeannSearcher
    LEANN_AVAILABLE = True
except ImportError:
    LEANN_AVAILABLE = False
    logging.warning("LEANN not available. Install with: pip install leann")

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class LeannSearchResult:
    """Search result from LEANN."""
    text: str
    score: float
    metadata: Dict[str, Any]


class LeannVectorStore:
    """
    LEANN-based vector store implementation.
    
    Provides 97% storage savings compared to traditional vector databases
    by using graph-based selective recomputation instead of storing all embeddings.
    """
    
    def __init__(self, 
                 index_name: str = None,
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 backend: str = "hnsw",
                 graph_degree: int = 32,
                 complexity: int = 64,
                 use_compact: bool = True,
                 use_recompute: bool = True):
        """
        Initialize LEANN vector store.
        
        Args:
            index_name: Name of the index
            embedding_model: Embedding model to use
            backend: Backend type ('hnsw' or 'diskann')
            graph_degree: Graph degree parameter
            complexity: Build complexity parameter
            use_compact: Use compact storage
            use_recompute: Enable recomputation
        """
        if not LEANN_AVAILABLE:
            raise ImportError("LEANN not available. Install with: pip install leann-core")
        
        # Use settings default if no index_name provided
        if index_name is None:
            from config.settings import settings
            index_name = settings.LEANN_INDEX_NAME
        
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.backend = backend
        self.graph_degree = graph_degree
        self.complexity = complexity
        self.use_compact = use_compact
        self.use_recompute = use_recompute
        
        self.builder = None
        self.searcher = None
        # LEANN expects the .leann directory for index files
        self.leann_index_path = Path(".leann")
        self.index_path = self.leann_index_path / index_name
        
        logger.info(f"LEANN Vector Store initialized with index: {index_name}")
        logger.info(f"Backend: {backend}, Embedding model: {embedding_model}")
    
    def _ensure_index_directory(self):
        """Ensure the index directory exists."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_existing_documents(self):
        """Load existing documents from processed files."""
        try:
            from config.settings import settings
            processed_dir = Path(settings.DATA_DIR) / "processed"
            
            if not processed_dir.exists():
                logger.info("No processed directory found")
                return
            
            existing_count = 0
            for file_path in processed_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() == ".json":
                    try:
                        import json
                        with open(file_path, 'r', encoding='utf-8') as f:
                            doc_data = json.load(f)
                        
                        # Add existing document to builder
                        chunks = doc_data.get('chunks', [])
                        for chunk_text in chunks:
                            if chunk_text.strip():
                                self.builder.add_text(chunk_text, metadata={
                                    'source_file': doc_data.get('file_path', ''),
                                    'file_name': Path(doc_data.get('file_path', '')).name,
                                    'content_length': doc_data.get('content_length', 0)
                                })
                                existing_count += 1
                                
                    except Exception as e:
                        logger.warning(f"Error loading existing document {file_path}: {e}")
                        continue
            
            logger.info(f"Loaded {existing_count} existing document chunks")
            
        except Exception as e:
            logger.error(f"Error loading existing documents: {e}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the LEANN index.
        
        Args:
            documents: List of documents with 'text' and 'metadata' keys
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._ensure_index_directory()
            
            # Initialize builder if not already done
            if self.builder is None:
                self.builder = LeannBuilder(
                    backend_name=self.backend,
                    embedding_model=self.embedding_model,
                    graph_degree=self.graph_degree,
                    complexity=self.complexity,
                    compact=self.use_compact,
                    recompute=self.use_recompute
                )
                
                # Load existing documents if index exists
                if self.index_path.exists():
                    logger.info("Loading existing documents from index...")
                    # Load existing documents from processed files
                    self._load_existing_documents()
            
            # Add documents to builder
            for doc in documents:
                text = doc.get('text', '')
                metadata = doc.get('metadata', {})
                
                if text.strip():
                    self.builder.add_text(text, metadata=metadata)
                    logger.debug(f"Added document with {len(text)} characters")
            
            logger.info(f"Added {len(documents)} documents to LEANN builder")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to LEANN: {e}")
            return False
    
    def build_index(self) -> bool:
        """
        Build the LEANN index from added documents.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.builder is None:
                logger.error("No documents added to builder")
                return False
            
            logger.info("Building LEANN index...")
            self.builder.build_index(str(self.index_path))
            
            # Initialize searcher
            self.searcher = LeannSearcher(index_path=str(self.index_path))
            
            logger.info(f"LEANN index built successfully at {self.index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error building LEANN index: {e}")
            return False
    
    def search(self, query: str, top_k: int = 20, complexity: int = None) -> List[LeannSearchResult]:
        """
        Search the LEANN index.
        
        Args:
            query: Search query
            top_k: Number of results to return
            complexity: Search complexity (uses default if None)
            
        Returns:
            List of search results
        """
        try:
            if self.searcher is None:
                # Try to load existing index before failing
                logger.warning("Searcher not initialized, attempting to load existing index...")
                if not self.load_existing_index():
                    logger.error("Index not built or searcher not initialized")
                    return []
            
            search_complexity = complexity or self.complexity
            
            # Perform search
            results = self.searcher.search(
                query=query,
                top_k=top_k,
                complexity=search_complexity,
                recompute=self.use_recompute
            )
            
            # Convert to our format
            search_results = []
            for result in results:
                search_results.append(LeannSearchResult(
                    text=result.text,
                    score=result.score,
                    metadata=result.metadata
                ))
            
            logger.info(f"LEANN search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching LEANN index: {e}")
            return []
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the current index.
        
        Returns:
            Dictionary with index information
        """
        info = {
            "index_name": self.index_name,
            "index_path": str(self.index_path),
            "embedding_model": self.embedding_model,
            "backend": self.backend,
            "graph_degree": self.graph_degree,
            "complexity": self.complexity,
            "use_compact": self.use_compact,
            "use_recompute": self.use_recompute,
            "index_exists": (self.leann_index_path / f"{self.index_name}.index").exists(),
            "searcher_initialized": self.searcher is not None,
            "document_count": len(self.builder.chunks) if self.builder else 0
        }
        
        # Get index size if it exists
        if (self.leann_index_path / f"{self.index_name}.index").exists():
            try:
                total_size = sum(f.stat().st_size for f in self.leann_index_path.glob(f"{self.index_name}.*") if f.is_file())
                info["index_size_bytes"] = total_size
                info["index_size_mb"] = round(total_size / (1024 * 1024), 2)
            except Exception as e:
                logger.warning(f"Could not calculate index size: {e}")
                info["index_size_bytes"] = 0
                info["index_size_mb"] = 0.0
        else:
            info["index_size_bytes"] = 0
            info["index_size_mb"] = 0.0
        
        return info
    
    def reset_index(self) -> bool:
        """
        Reset the LEANN index.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import shutil
            import os
            
            # Remove the entire .leann directory to clean up all index files
            if self.leann_index_path.exists():
                shutil.rmtree(self.leann_index_path)
                logger.info(f"Removed LEANN index directory at {self.leann_index_path}")
            
            # Also clean up any LEANN files in the root directory
            root_files_to_remove = [
                ".leann.index",
                ".leann.csr.tmp", 
                ".leann.meta.json",
                ".leann.passages.idx",
                ".leann.passages.jsonl"
            ]
            
            for file_name in root_files_to_remove:
                if os.path.exists(file_name):
                    os.remove(file_name)
                    logger.info(f"Removed LEANN file: {file_name}")
            
            # Reset builder and searcher
            self.builder = None
            self.searcher = None
            
            return True
            
        except Exception as e:
            logger.error(f"Error resetting LEANN index: {e}")
            return False
    
    def load_existing_index(self) -> bool:
        """
        Load an existing LEANN index.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if the metadata file exists in .leann directory
            meta_file = self.leann_index_path / f"{self.index_name}.meta.json"
            if not meta_file.exists():
                logger.warning(f"LEANN metadata file not found at {meta_file}")
                return False
            
            # Initialize searcher with the specific index path
            from leann.api import LeannSearcher
            self.searcher = LeannSearcher(index_path=str(self.index_path))
            logger.info(f"Loaded existing LEANN index from {self.index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading existing LEANN index: {e}")
            return False
    
    def force_initialize_searcher(self):
        """Force initialization of the searcher."""
        logger.info("Force initializing LEANN searcher...")
        try:
            if self.searcher is None:
                logger.info(f"Checking for existing index at {self.leann_index_path}")
                # Try to load existing index
                if self.load_existing_index():
                    logger.info("Successfully initialized searcher from existing index")
                else:
                    logger.warning("No existing index found to initialize searcher")
            else:
                logger.info("Searcher already initialized")
        except Exception as e:
            logger.error(f"Error force initializing searcher: {e}")


class LeannVectorStoreManager:
    """
    Manager class for LEANN vector stores.
    
    Provides high-level operations and integration with the existing Myr-Ag system.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize LEANN vector store manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.stores: Dict[str, LeannVectorStore] = {}
        
        logger.info("LEANN Vector Store Manager initialized")
    
    def create_store(self, 
                    index_name: str,
                    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    backend: str = "hnsw") -> LeannVectorStore:
        """
        Create a new LEANN vector store.
        
        Args:
            index_name: Name of the index
            embedding_model: Embedding model to use
            backend: Backend type
            
        Returns:
            LeannVectorStore instance
        """
        store = LeannVectorStore(
            index_name=index_name,
            embedding_model=embedding_model,
            backend=backend
        )
        
        self.stores[index_name] = store
        logger.info(f"Created LEANN store: {index_name}")
        return store
    
    def get_store(self, index_name: str) -> Optional[LeannVectorStore]:
        """
        Get an existing LEANN vector store.
        
        Args:
            index_name: Name of the index
            
        Returns:
            LeannVectorStore instance or None
        """
        return self.stores.get(index_name)
    
    def list_stores(self) -> List[Dict[str, Any]]:
        """
        List all available LEANN stores.
        
        Returns:
            List of store information dictionaries
        """
        stores_info = []
        
        for name, store in self.stores.items():
            info = store.get_index_info()
            stores_info.append(info)
        
        return stores_info
    
    def benchmark_storage_savings(self, 
                                 documents: List[Dict[str, Any]],
                                 ) -> Dict[str, Any]:
        """
        Benchmark storage savings compared to traditional vector databases.
        
        Args:
            documents: List of documents to benchmark
            
        Returns:
            Dictionary with benchmark results
        """
        results = {
            "document_count": len(documents),
            "total_text_length": sum(len(doc.get('text', '')) for doc in documents),
            "leann_index_size": 0,
            "estimated_traditional_size": 0,
            "storage_savings_percent": 0
        }
        
        try:
            # Create temporary LEANN index
            temp_store = self.create_store("benchmark_temp")
            temp_store.add_documents(documents)
            temp_store.build_index()
            
            # Get LEANN index size
            leann_info = temp_store.get_index_info()
            results["leann_index_size"] = leann_info.get("index_size_bytes", 0)
            results["leann_index_size_mb"] = leann_info.get("index_size_mb", 0)
            
            # Estimate traditional vector database size
            # Rough estimation: 384 dimensions * 4 bytes * document_count
            embedding_dim = 384  # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
            estimated_size = embedding_dim * 4 * len(documents)
            results["estimated_traditional_size"] = estimated_size
            results["estimated_traditional_size_mb"] = round(estimated_size / (1024 * 1024), 2)
            
            # Calculate savings
            if estimated_size > 0:
                savings = ((estimated_size - results["leann_index_size"]) / estimated_size) * 100
                results["storage_savings_percent"] = round(savings, 2)
            
            # Cleanup
            temp_store.reset_index()
            
            logger.info(f"Benchmark completed: {results['storage_savings_percent']}% storage savings")
            
        except Exception as e:
            logger.error(f"Error in benchmark: {e}")
            results["error"] = str(e)
        
        return results
