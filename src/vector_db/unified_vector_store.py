"""
LEANN Vector Store Manager for Myr-Ag RAG System.

This module provides a unified interface for LEANN vector store operations.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from config.settings import settings
from src.vector_db.leann_store import LeannVectorStore, LeannVectorStoreManager

logger = logging.getLogger(__name__)


class UnifiedVectorStore:
    """
    LEANN-based vector store implementation.
    
    This class provides a consistent interface for LEANN vector operations.
    """
    
    def __init__(self, collection_name: str = "documents"):
        """
        Initialize the LEANN vector store.
        
        Args:
            collection_name: Name of the collection/index
        """
        self.collection_name = collection_name
        self.vector_store = None
        self.store_type = "LEANN"
        
        # Initialize LEANN vector store
        self._initialize_vector_store()
        
        logger.info(f"LEANN Vector Store initialized: {collection_name}")
    
    def _initialize_vector_store(self):
        """Initialize the LEANN vector store."""
        try:
            leann_manager = LeannVectorStoreManager()
            self.vector_store = leann_manager.create_store(
                index_name=self.collection_name,
                embedding_model=settings.EMBEDDING_MODEL,
                backend=settings.LEANN_BACKEND
            )
            logger.info(f"LEANN vector store initialized: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize LEANN vector store: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents with 'text' and 'metadata' keys
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # LEANN requires building the index after adding documents
            success = self.vector_store.add_documents(documents)
            if success:
                success = self.vector_store.build_index()
            return success
                
        except Exception as e:
            logger.error(f"Error adding documents to LEANN: {e}")
            return False
    
    def force_initialize_searcher(self):
        """Force initialize the searcher if available."""
        logger.info(f"Unified store attempting to force initialize searcher...")
        if hasattr(self.vector_store, 'force_initialize_searcher'):
            logger.info("Calling vector store force_initialize_searcher method")
            self.vector_store.force_initialize_searcher()
        else:
            logger.warning("Vector store does not have force_initialize_searcher method")
    
    def search(self, query: str, n_results: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
        """
        try:
            # LEANN search
            results = self.vector_store.search(query, top_k=n_results)
            # Convert LEANN results to unified format
            return [
                {
                    'text': result.text,
                    'metadata': result.metadata,
                    'score': float(result.score),  # Convert numpy.float32 to Python float
                    'distance': 1 - float(result.score)  # Convert similarity to distance
                }
                for result in results
            ]
                
        except Exception as e:
            logger.error(f"Error searching in LEANN: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection/index.
        
        Returns:
            Dictionary with collection information
        """
        try:
            return self.vector_store.get_index_info()
                
        except Exception as e:
            logger.error(f"Error getting collection info from LEANN: {e}")
            return {}
    
    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the index (alias for get_collection_info).
        
        Returns:
            Dictionary with index information
        """
        return self.get_collection_info()
    
    
    def force_initialize_searcher(self):
        """Force initialization of the LEANN searcher."""
        if self.vector_store:
            self.vector_store.force_initialize_searcher()
    
    def reset_collection(self) -> bool:
        """
        Reset/clear the collection/index.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # For LEANN, we need to recreate the index
            return self.vector_store.reset_index()
                
        except Exception as e:
            logger.error(f"Error resetting LEANN collection: {e}")
            return False
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents from the collection.
        
        Returns:
            List of all documents
        """
        try:
            # LEANN doesn't have a direct "get all" method
            # This would need to be implemented based on LEANN's API
            logger.warning("get_all_documents not implemented for LEANN")
            return []
                
        except Exception as e:
            logger.error(f"Error getting all documents from LEANN: {e}")
            return []
    
    def get_store_type(self) -> str:
        """Get the current store type."""
        return self.store_type
    
    def is_leann(self) -> bool:
        """Check if using LEANN."""
        return True
    


class UnifiedVectorStoreManager:
    """
    Manager for unified vector stores.
    
    Provides high-level operations and manages multiple collections/indexes.
    """
    
    def __init__(self):
        """Initialize the unified vector store manager."""
        self.stores: Dict[str, UnifiedVectorStore] = {}
        logger.info("Unified Vector Store Manager initialized")
    
    def get_or_create_store(self, collection_name: str = "documents") -> UnifiedVectorStore:
        """
        Get existing store or create a new one.
        
        Args:
            collection_name: Name of the collection/index
            
        Returns:
            UnifiedVectorStore instance
        """
        if collection_name not in self.stores:
            self.stores[collection_name] = UnifiedVectorStore(collection_name)
            logger.info(f"Created new unified store: {collection_name}")
        else:
            logger.info(f"Retrieved existing unified store: {collection_name}")
        
        return self.stores[collection_name]
    
    def list_stores(self) -> List[Dict[str, Any]]:
        """
        List all available stores.
        
        Returns:
            List of store information
        """
        stores_info = []
        for name, store in self.stores.items():
            info = {
                'name': name,
                'type': store.get_store_type(),
                'is_leann': store.is_leann()
            }
            stores_info.append(info)
        
        return stores_info
    
    def get_store_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific store.
        
        Args:
            collection_name: Name of the collection/index
            
        Returns:
            Dictionary with store information
        """
        if collection_name not in self.stores:
            return {}
        
        store = self.stores[collection_name]
        info = store.get_collection_info()
        info.update({
            'name': collection_name,
            'type': store.get_store_type(),
            'is_leann': store.is_leann()
        })
        
        return info


# Global manager instance
unified_manager = UnifiedVectorStoreManager()
