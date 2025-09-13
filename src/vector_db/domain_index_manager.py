"""
Domain-specific index manager for specialized pipelines.

This module manages separate vector indexes for each pipeline domain,
providing better performance and more focused search results.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from .unified_vector_store import UnifiedVectorStore, unified_manager
from .document_indexer import DocumentIndexer
from ..document_processing.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


@dataclass
class DomainIndexInfo:
    """Information about a domain-specific index."""
    
    domain: str
    index_name: str
    document_count: int
    chunk_count: int
    last_updated: str
    is_initialized: bool


class DomainIndexManager:
    """Manages domain-specific vector indexes for specialized pipelines."""
    
    def __init__(self, base_data_dir: Path = None):
        """Initialize the domain index manager."""
        self.base_data_dir = base_data_dir or Path("data")
        self.domains = ["financial", "legal", "medical", "academic", "excel", "general"]
        
        # Domain-specific index names
        self.domain_indexes = {
            "financial": "financial_collection",
            "legal": "legal_collection", 
            "medical": "medical_collection",
            "academic": "academic_collection",
            "excel": "excel_collection",
            "general": "main_collection"  # Keep existing general index
        }
        
        # Initialize domain-specific vector stores
        self.vector_stores = {}
        self.document_indexers = {}
        
        # Initialize document processor
        self.document_processor = DocumentProcessor()
        
        logger.info("Domain index manager initialized")
    
    def initialize_domain_indexes(self):
        """Initialize all domain-specific indexes."""
        try:
            for domain in self.domains:
                self._initialize_domain_index(domain)
            
            logger.info("All domain indexes initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing domain indexes: {e}")
            return False
    
    def _initialize_domain_index(self, domain: str):
        """Initialize a specific domain index."""
        try:
            index_name = self.domain_indexes[domain]
            
            # Create domain-specific vector store
            vector_store = unified_manager.get_or_create_store(index_name)
            self.vector_stores[domain] = vector_store
            
            # Create domain-specific document indexer
            document_indexer = DocumentIndexer(vector_store, self.document_processor)
            self.document_indexers[domain] = document_indexer
            
            logger.info(f"Initialized {domain} index: {index_name}")
            
        except Exception as e:
            logger.error(f"Error initializing {domain} index: {e}")
            raise
    
    def get_domain_indexer(self, domain: str) -> Optional[DocumentIndexer]:
        """Get the document indexer for a specific domain."""
        return self.document_indexers.get(domain)
    
    def get_domain_vector_store(self, domain: str) -> Optional[UnifiedVectorStore]:
        """Get the vector store for a specific domain."""
        return self.vector_stores.get(domain)
    
    def detect_document_domain(self, file_path: Path, content: str = None) -> str:
        """Detect the appropriate domain for a document based on content analysis."""
        try:
            # Check if it's an Excel file first
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                logger.info(f"Excel file detected, assigning to excel domain: {file_path.name}")
                return "excel"
            
            # If content is not provided, extract it
            if content is None:
                content = self.document_processor.extract_text(file_path)
            
            content_lower = content.lower()
            
            # Domain-specific keyword detection
            domain_scores = {
                "financial": self._calculate_financial_score(content_lower),
                "legal": self._calculate_legal_score(content_lower),
                "medical": self._calculate_medical_score(content_lower),
                "academic": self._calculate_academic_score(content_lower)
            }
            
            # Find the domain with the highest score
            best_domain = max(domain_scores, key=domain_scores.get)
            
            # If no domain scores high enough, use general
            if domain_scores[best_domain] < 0.1:
                best_domain = "general"
            
            logger.info(f"Detected domain '{best_domain}' for {file_path.name} (score: {domain_scores[best_domain]:.2f})")
            return best_domain
            
        except Exception as e:
            logger.warning(f"Error detecting domain for {file_path.name}: {e}")
            return "general"
    
    def _calculate_financial_score(self, content: str) -> float:
        """Calculate financial domain score based on content."""
        financial_keywords = [
            "revenue", "profit", "income", "expense", "budget", "cost", "price",
            "financial", "accounting", "balance sheet", "income statement",
            "cash flow", "investment", "roi", "kpi", "quarterly", "annual",
            "earnings", "loss", "assets", "liabilities", "equity", "debt",
            "currency", "dollar", "euro", "yen", "pound", "franc",
            "tax", "audit", "compliance", "regulatory", "fiscal"
        ]
        
        score = 0.0
        for keyword in financial_keywords:
            if keyword in content:
                score += 1.0
        
        # Normalize by content length
        return min(score / max(len(content.split()), 1), 1.0)
    
    def _calculate_legal_score(self, content: str) -> float:
        """Calculate legal domain score based on content."""
        legal_keywords = [
            "contract", "agreement", "legal", "law", "clause", "provision",
            "liability", "obligation", "right", "duty", "compliance",
            "regulation", "statute", "court", "litigation", "lawsuit",
            "attorney", "lawyer", "counsel", "jurisdiction", "precedent",
            "terms", "conditions", "warranty", "indemnity", "breach",
            "damages", "penalty", "fine", "violation", "infringement"
        ]
        
        score = 0.0
        for keyword in legal_keywords:
            if keyword in content:
                score += 1.0
        
        return min(score / max(len(content.split()), 1), 1.0)
    
    def _calculate_medical_score(self, content: str) -> float:
        """Calculate medical domain score based on content."""
        medical_keywords = [
            "patient", "medical", "health", "diagnosis", "treatment", "therapy",
            "symptom", "disease", "illness", "clinical", "hospital", "doctor",
            "physician", "nurse", "medication", "drug", "prescription",
            "surgery", "procedure", "examination", "test", "lab", "laboratory",
            "blood", "pressure", "temperature", "pulse", "heart", "lung",
            "cancer", "diabetes", "hypertension", "infection", "virus",
            "bacteria", "antibiotic", "vaccine", "immunization"
        ]
        
        score = 0.0
        for keyword in medical_keywords:
            if keyword in content:
                score += 1.0
        
        return min(score / max(len(content.split()), 1), 1.0)
    
    def _calculate_academic_score(self, content: str) -> float:
        """Calculate academic domain score based on content."""
        academic_keywords = [
            "research", "study", "academic", "university", "college", "thesis",
            "dissertation", "paper", "article", "journal", "publication",
            "methodology", "hypothesis", "analysis", "findings", "conclusion",
            "abstract", "introduction", "literature", "review", "citation",
            "reference", "author", "co-author", "peer", "review", "scholarly",
            "empirical", "experimental", "data", "statistics", "sample",
            "population", "variable", "correlation", "significance", "p-value"
        ]
        
        score = 0.0
        for keyword in academic_keywords:
            if keyword in content:
                score += 1.0
        
        return min(score / max(len(content.split()), 1), 1.0)
    
    def index_document(self, file_path: Path, force_domain: str = None) -> Tuple[bool, str]:
        """Index a document in the appropriate domain index."""
        try:
            # Detect domain if not forced
            if force_domain:
                domain = force_domain
                logger.info(f"Using forced domain '{force_domain}' for {file_path.name}")
            else:
                domain = self.detect_document_domain(file_path)
                logger.info(f"Detected domain '{domain}' for {file_path.name}")
            
            # Get the appropriate indexer
            indexer = self.get_domain_indexer(domain)
            if not indexer:
                logger.error(f"No indexer found for domain: {domain}")
                return False, f"Domain {domain} not available"
            
            # Index the document
            success = indexer.index_document(file_path, force_domain)
            
            if success:
                # Store domain metadata with the document
                self._store_domain_metadata(file_path, domain)
                logger.info(f"Successfully indexed {file_path.name} in {domain} domain")
                return True, f"Indexed in {domain} domain"
            else:
                logger.error(f"Failed to index {file_path.name} in {domain} domain")
                return False, f"Failed to index in {domain} domain"
                
        except Exception as e:
            logger.error(f"Error indexing document {file_path.name}: {e}")
            return False, f"Error: {str(e)}"
    
    def _store_domain_metadata(self, file_path: Path, domain: str):
        """Store domain metadata with the processed document."""
        try:
            processed_dir = Path("data/processed")
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Create metadata file
            metadata_file = processed_dir / f"{file_path.stem}.json"
            
            # Load existing metadata or create new
            metadata = {}
            if metadata_file.exists():
                try:
                    import json
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except Exception as e:
                    logger.warning(f"Error loading existing metadata: {e}")
            
            # Update with domain information
            metadata.update({
                "file_name": file_path.name,
                "file_path": str(file_path),
                "domain": domain,
                "domain_updated": str(datetime.now()),
                "processing_timestamp": metadata.get("processing_timestamp", str(datetime.now()))
            })
            
            # Save metadata
            import json
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Stored domain metadata for {file_path.name}: {domain}")
            
        except Exception as e:
            logger.warning(f"Error storing domain metadata: {e}")
    
    def get_document_domain(self, file_name: str) -> str:
        """Get the domain of a specific document."""
        try:
            processed_dir = Path("data/processed")
            metadata_file = processed_dir / f"{Path(file_name).stem}.json"
            
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                return metadata.get("domain", "general")
            else:
                return "general"
                
        except Exception as e:
            logger.warning(f"Error getting document domain: {e}")
            return "general"
    
    def index_directory(self, directory_path: Path, force_domain: str = None) -> Dict[str, int]:
        """Index all documents in a directory, routing to appropriate domains."""
        try:
            results = {domain: 0 for domain in self.domains}
            
            if not directory_path.exists():
                logger.error(f"Directory not found: {directory_path}")
                return results
            
            # Process all files in the directory
            for file_path in directory_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in [
                    ".pdf", ".docx", ".txt", ".md", ".html", ".xhtml", 
                    ".csv", ".png", ".jpeg", ".jpg", ".tiff", ".bmp", 
                    ".webp", ".adoc", ".xml"
                ]:
                    # Check if document already has a domain assignment (unless forcing)
                    if not force_domain:
                        existing_domain = self._get_existing_domain(file_path)
                        if existing_domain and existing_domain != "general":
                            logger.info(f"Skipping {file_path.name} - already assigned to {existing_domain} domain")
                            results[existing_domain] += 1
                            continue
                    
                    success, message = self.index_document(file_path, force_domain)
                    if success:
                        # Extract domain from message (format: "Indexed in {domain} domain")
                        if "Indexed in" in message and "domain" in message:
                            domain = message.split()[2]  # Get the domain name (3rd word)
                        else:
                            domain = "general"
                        results[domain] += 1
                    else:
                        logger.warning(f"Failed to index {file_path.name}: {message}")
            
            logger.info(f"Directory indexing completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error indexing directory {directory_path}: {e}")
            return {domain: 0 for domain in self.domains}
    
    def _get_existing_domain(self, file_path: Path) -> Optional[str]:
        """Get the existing domain assignment for a document."""
        try:
            processed_dir = Path("data/processed")
            metadata_file = processed_dir / f"{file_path.stem}.json"
            
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                return metadata.get('domain')
            return None
        except Exception as e:
            logger.warning(f"Error reading existing domain for {file_path.name}: {e}")
            return None
    
    def search_domain(self, query: str, domain: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search within a specific domain index."""
        try:
            indexer = self.get_domain_indexer(domain)
            if not indexer:
                logger.error(f"No indexer found for domain: {domain}")
                return []
            
            results = indexer.search_documents(query, n_results)
            logger.info(f"Found {len(results)} results in {domain} domain")
            return results
            
        except Exception as e:
            logger.error(f"Error searching {domain} domain: {e}")
            return []
    
    def search_all_domains(self, query: str, n_results_per_domain: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """Search across all domain indexes."""
        try:
            results = {}
            
            for domain in self.domains:
                domain_results = self.search_domain(query, domain, n_results_per_domain)
                if domain_results:
                    results[domain] = domain_results
            
            logger.info(f"Search completed across {len(results)} domains")
            return results
            
        except Exception as e:
            logger.error(f"Error searching all domains: {e}")
            return {}
    
    def get_domain_index_info(self, domain: str = None) -> Dict[str, Any]:
        """Get information about domain indexes."""
        try:
            if domain:
                # Get info for specific domain
                indexer = self.get_domain_indexer(domain)
                if not indexer:
                    return {"error": f"Domain {domain} not found"}
                
                index_info = indexer.get_index_info()
                return {
                    "domain": domain,
                    "index_name": self.domain_indexes[domain],
                    "document_count": index_info.get("document_count", 0),
                    "chunk_count": index_info.get("chunk_count", 0),
                    "is_initialized": True
                }
            else:
                # Get info for all domains
                all_info = {}
                for domain in self.domains:
                    all_info[domain] = self.get_domain_index_info(domain)
                return all_info
                
        except Exception as e:
            logger.error(f"Error getting domain index info: {e}")
            return {"error": str(e)}
    
    def reset_domain_index(self, domain: str) -> bool:
        """Reset a specific domain index."""
        try:
            indexer = self.get_domain_indexer(domain)
            if not indexer:
                logger.error(f"No indexer found for domain: {domain}")
                return False
            
            success = indexer.reset_index()
            if success:
                logger.info(f"Reset {domain} domain index successfully")
            else:
                logger.error(f"Failed to reset {domain} domain index")
            
            return success
            
        except Exception as e:
            logger.error(f"Error resetting {domain} domain index: {e}")
            return False
    
    def rebuild_domain_index(self, domain: str) -> bool:
        """Rebuild a specific domain index from existing processed documents."""
        try:
            indexer = self.get_domain_indexer(domain)
            if not indexer:
                logger.error(f"No indexer found for domain: {domain}")
                return False
            
            # For LEANN domains, use rebuild_leann_index with domain filter
            if domain in ["financial", "legal", "medical", "academic", "general"]:
                success = indexer.rebuild_leann_index(domain)
            elif domain == "excel":
                # For Excel domain, use LlamaIndex rebuild
                success = indexer.rebuild_llamaindex_excel()
            else:
                # For other domains, use reset_index as fallback
                success = indexer.reset_index()
            
            if success:
                logger.info(f"Rebuilt {domain} domain index successfully")
            else:
                logger.error(f"Failed to rebuild {domain} domain index")
            
            return success
            
        except Exception as e:
            logger.error(f"Error rebuilding {domain} domain index: {e}")
            return False
    
    def reset_all_domain_indexes(self) -> Dict[str, bool]:
        """Reset all domain indexes."""
        try:
            results = {}
            
            for domain in self.domains:
                results[domain] = self.reset_domain_index(domain)
            
            logger.info(f"Reset all domain indexes: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error resetting all domain indexes: {e}")
            return {domain: False for domain in self.domains}
    
    def rebuild_all_domain_indexes(self) -> Dict[str, bool]:
        """Rebuild all domain indexes."""
        try:
            results = {}
            
            for domain in self.domains:
                results[domain] = self.rebuild_domain_index(domain)
            
            logger.info(f"Rebuild all domain indexes: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error rebuilding all domain indexes: {e}")
            return {domain: False for domain in self.domains}
    
    def list_available_domains(self) -> List[str]:
        """List all available domains."""
        return self.domains.copy()
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get statistics for all domain indexes."""
        try:
            stats = {}
            
            for domain in self.domains:
                domain_info = self.get_domain_index_info(domain)
                if "error" not in domain_info:
                    stats[domain] = {
                        "document_count": domain_info.get("document_count", 0),
                        "chunk_count": domain_info.get("chunk_count", 0),
                        "is_initialized": domain_info.get("is_initialized", False)
                    }
                else:
                    stats[domain] = {"error": domain_info["error"]}
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting domain statistics: {e}")
            return {"error": str(e)}
