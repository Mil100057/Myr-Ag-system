"""
Document Processor for handling document ingestion, extraction, and chunking.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from docling.document_converter import DocumentConverter
from docling.exceptions import ConversionError
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from loguru import logger

from config.settings import settings
from .excel_processor import ExcelProcessor, ExcelChunk


@dataclass
class ProcessedDocument:
    """Represents a processed document with metadata."""
    file_path: Path
    content: str
    chunks: List[str]
    metadata: Dict[str, Any]
    chunk_embeddings: Optional[List[List[float]]] = None


class DocumentProcessor:
    """Handles document processing including extraction and chunking."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.supported_formats = settings.SUPPORTED_FORMATS
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        
        # Initialize Docling converter
        self.docling_converter = DocumentConverter()
        
        # Initialize enhanced chunking strategy with better SentenceSplitter configuration
        self.enhanced_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            paragraph_separator="\n\n",  # Better paragraph detection
            secondary_chunking_regex=r'[^,.;。？！]+[,.;。？！]?|[,.;。？！]',  # More granular sentence splitting
            separator=" "  # Use space as primary separator
        )
        
        # Initialize Excel processor with smaller grouping threshold
        self.excel_processor = ExcelProcessor(max_rows_per_chunk=20, enable_grouping=True)
        
        logger.info(f"Document processor initialized with chunk size: {self.chunk_size}, overlap: {self.chunk_overlap}")
        logger.info("Using enhanced hybrid chunking strategy: paragraph-based + enhanced SentenceSplitter")
    
    def validate_file(self, file_path: Path) -> bool:
        """Validate if the file can be processed."""
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False
        
        if file_path.suffix.lower() not in self.supported_formats:
            logger.error(f"Unsupported file format: {file_path.suffix}")
            return False
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > settings.MAX_FILE_SIZE_MB:
            logger.error(f"File too large: {file_size_mb:.2f}MB > {settings.MAX_FILE_SIZE_MB}MB")
            return False
        
        return True
    
    def extract_text_with_docling(self, file_path: Path) -> str:
        """Extract text from document using Docling with PyPDF2 fallback."""
        try:
            logger.info(f"Extracting text from: {file_path}")
            
            # Use Docling for text extraction
            result = self.docling_converter.convert(file_path)
            
            if result.status.value != "success":
                logger.error(f"Docling conversion failed: {result.status}")
                return self._extract_text_with_pypdf(file_path)
            
            # Extract text from the Docling document
            doc = result.document
            
            # Use export_to_markdown() for better compatibility with all formats
            # This works for PDF, DOCX, XLSX, PPTX, HTML, and other formats
            try:
                full_text = doc.export_to_markdown()
                if not full_text or not full_text.strip():
                    logger.warning(f"No content extracted with Docling from: {file_path}, trying PyPDF2 fallback")
                    return self._extract_text_with_pypdf(file_path)
            except Exception as e:
                logger.warning(f"Markdown export failed for {file_path}: {e}, trying PyPDF2 fallback")
                return self._extract_text_with_pypdf(file_path)
            
            logger.info(f"Successfully extracted {len(full_text)} characters from: {file_path}")
            return full_text
            
        except ConversionError as e:
            logger.error(f"Docling conversion error for {file_path}: {str(e)}, trying pypdf fallback")
            return self._extract_text_with_pypdf(file_path)
        except Exception as e:
            logger.error(f"Unexpected error with Docling for {file_path}: {str(e)}, trying pypdf fallback")
            return self._extract_text_with_pypdf(file_path)
    
    def _extract_text_with_pypdf(self, file_path: Path) -> str:
        """Extract text using pypdf as fallback for PDF files."""
        try:
            # Only use pypdf for PDF files
            if file_path.suffix.lower() != ".pdf":
                logger.warning(f"pypdf fallback only supports PDF files, not: {file_path.suffix}")
                return ""
            
            logger.info(f"Using pypdf fallback for: {file_path}")
            
            import pypdf
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text_parts = []
                
                # Extract text from all pages
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_parts.append(page_text)
                    except Exception as page_error:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {page_error}")
                        continue
                
                full_text = "\n".join(text_parts)
                
                if not full_text.strip():
                    logger.warning(f"No text extracted with pypdf from: {file_path}")
                    return ""
                
                logger.info(f"Successfully extracted {len(full_text)} characters with pypdf from: {file_path}")
                return full_text
                
        except ImportError:
            logger.error("pypdf not available for fallback extraction")
            return ""
        except Exception as e:
            logger.error(f"pypdf fallback error for {file_path}: {str(e)}")
            return ""
    
    def extract_text(self, file_path: Path) -> str:
        """Extract text from document using the appropriate method."""
        # Use Docling for complex formats (PDF, DOCX, XLSX, PPTX, HTML, images, etc.)
        if file_path.suffix.lower() in [".pdf", ".docx", ".xlsx", ".pptx", ".html", ".xhtml", ".md", ".csv", ".png", ".jpeg", ".jpg", ".tiff", ".bmp", ".webp", ".adoc", ".xml"]:
            return self.extract_text_with_docling(file_path)
        else:
            # For simple text formats (txt), read directly
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {str(e)}")
                return ""
    
    def extract_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Extract metadata from the document."""
        metadata = {
            "filename": file_path.name,
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "file_extension": file_path.suffix.lower(),
            "content_length": len(content),
            "chunk_count": 0,
            "processing_timestamp": str(datetime.now()),
        }
        
        # Add file-specific metadata
        if file_path.suffix.lower() == ".pdf":
            try:
                result = self.docling_converter.convert(file_path)
                if result.status.value == "success" and hasattr(result.document, 'origin'):
                    origin = result.document.origin
                    if hasattr(origin, 'mimetype'):
                        metadata["mimetype"] = origin.mimetype
            except Exception as e:
                logger.warning(f"Could not extract PDF metadata: {e}")
        
        return metadata
    
    def _split_with_sentence_boundaries(self, text: str) -> List[str]:
        """Split text respecting sentence boundaries and preserving proper names."""
        try:
            logger.info(f"Splitting text with sentence boundaries for length: {len(text)}")
            
            # Split by sentences first
            sentences = self._split_into_sentences(text)
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                # Check if adding this sentence would exceed chunk size
                if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                    current_chunk += (sentence + " ") if current_chunk else sentence
                else:
                    # Save current chunk if it has content
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    # Start new chunk with current sentence
                    current_chunk = sentence
            
            # Add the last chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            logger.info(f"Sentence boundary splitting created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error with sentence boundary splitting: {str(e)}")
            # Fallback to enhanced splitter
            return self._split_with_enhanced_splitter(text)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving proper names."""
        import re
        
        # Enhanced sentence splitting that preserves proper names
        # Look for sentence endings but avoid splitting on abbreviations
        sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
        
        # Split by sentence endings
        sentences = re.split(sentence_endings, text)
        
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _split_with_enhanced_splitter(self, text: str) -> List[str]:
        """Use enhanced SentenceSplitter as fallback chunking method."""
        try:
            logger.info(f"Using enhanced SentenceSplitter for text of length: {len(text)}")
            
            # Create a LlamaIndex Document
            doc = Document(text=text)
            
            # Parse into nodes/chunks using enhanced SentenceSplitter
            nodes = self.enhanced_splitter.get_nodes_from_documents([doc])
            
            # Extract text from nodes
            chunks = [node.text for node in nodes]
            
            logger.info(f"Enhanced SentenceSplitter created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error with enhanced SentenceSplitter: {str(e)}")
            # Ultimate fallback: simple character-based splitting
            return self._fallback_split(text)
    
    def _fallback_split(self, text: str) -> List[str]:
        """Ultimate fallback: simple character-based splitting."""
        logger.warning("Using fallback character-based splitting")
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
            
            if start >= len(text):
                break
        
        logger.info(f"Fallback splitting created {len(chunks)} chunks")
        return chunks
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text to improve chunking quality."""
        # Clean up excessive whitespace
        text = " ".join(text.split())
        
        # Add line breaks after sentences for better paragraph detection
        import re
        text = re.sub(r'([.!?])\s+', r'\1\n\n', text)
        
        # Clean up multiple line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """Improved hybrid chunking approach with better sentence boundary respect."""
        try:
            logger.info(f"Starting hybrid chunking for text of length: {len(text)}")
            
            # Preprocess text to improve chunking
            text = self._preprocess_text(text)
            
            # Strategy 1: Always use sentence boundary splitting for better quality
            logger.info("Using sentence boundary splitting for improved chunk quality")
            return self._split_with_sentence_boundaries(text)
            
            # Strategy 2: Try line-based splitting for documents with many lines
            if "\n" in text and text.count("\n") > 10:
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                if len(lines) > 1:
                    logger.info(f"Found {len(lines)} lines, using line-based splitting")
                    chunks = []
                    current_chunk = ""
                    
                    for line in lines:
                        if len(current_chunk) + len(line) + 1 <= self.chunk_size:
                            current_chunk += (line + "\n") if current_chunk else line
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = line
                    
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                    logger.info(f"Line-based splitting created {len(chunks)} chunks")
                    return chunks
            
            # Strategy 3: Fallback to enhanced SentenceSplitter
            logger.info("Using enhanced SentenceSplitter as fallback")
            return self._split_with_enhanced_splitter(text)
            
        except Exception as e:
            logger.error(f"Error in hybrid chunking: {str(e)}")
            # Ultimate fallback
            return self._fallback_split(text)
    
    def process_document(self, file_path: Path) -> ProcessedDocument:
        """Process a single document end-to-end."""
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Validate file
            if not self.validate_file(file_path):
                raise ValueError(f"File validation failed: {file_path}")
            
            # Extract text
            content = self.extract_text(file_path)
            if not content:
                raise ValueError(f"No content extracted from: {file_path}")
            
            # Extract metadata
            metadata = self.extract_metadata(file_path, content)
            
            # Use Excel-specific processing for Excel files
            if file_path.suffix.lower() == '.xlsx':
                logger.info(f"Using Excel-specific processing for: {file_path}")
                excel_chunks = self.excel_processor.process_excel_file(file_path, content)
                
                # Convert ExcelChunk objects to text chunks
                chunks = []
                for excel_chunk in excel_chunks:
                    # Create enhanced chunk text with metadata
                    if excel_chunk.sheet_name:
                        chunk_text = f"[SHEET: {excel_chunk.sheet_name}] [{excel_chunk.chunk_type.upper()}] {excel_chunk.text}"
                    else:
                        chunk_text = f"[{excel_chunk.chunk_type.upper()}] {excel_chunk.text}"
                    
                    chunks.append(chunk_text)
                    
                    # Add Excel-specific metadata
                    if 'excel_metadata' not in metadata:
                        metadata['excel_metadata'] = []
                    metadata['excel_metadata'].append({
                        'chunk_type': excel_chunk.chunk_type,
                        'sheet_name': excel_chunk.sheet_name,
                        'metadata': excel_chunk.metadata
                    })
                
                metadata["chunk_count"] = len(chunks)
                metadata["excel_processing"] = True
                logger.info(f"Excel processing created {len(chunks)} enhanced chunks")
            else:
                # Use standard chunking for non-Excel files
                chunks = self.chunk_text(content)
                metadata["chunk_count"] = len(chunks)
            
            # Create processed document
            processed_doc = ProcessedDocument(
                file_path=file_path,
                content=content,
                chunks=chunks,
                metadata=metadata
            )
            
            logger.info(f"Successfully processed document: {file_path} into {len(chunks)} chunks")
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    def process_directory(self, directory_path: Path) -> List[ProcessedDocument]:
        """Process all supported documents in a directory."""
        processed_docs = []
        
        logger.info(f"Processing directory: {directory_path}")
        
        # Check for already processed documents
        processed_dir = Path(settings.DATA_DIR) / "processed"
        processed_files = set()
        if processed_dir.exists():
            for proc_file in processed_dir.iterdir():
                if proc_file.suffix.lower() == ".json":
                    # Extract original filename from processed filename
                    # e.g., "document_processed.json" -> "document.pdf"
                    original_name = proc_file.stem.replace("_processed", "")
                    for ext in self.supported_formats:
                        if original_name.endswith(ext):
                            processed_files.add(original_name)
                            break
        
        for file_path in directory_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                # Skip if already processed
                if file_path.name in processed_files:
                    logger.info(f"Skipping already processed document: {file_path.name}")
                    continue
                
                try:
                    processed_doc = self.process_document(file_path)
                    processed_docs.append(processed_doc)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    continue
        
        logger.info(f"Processed {len(processed_docs)} new documents from directory")
        return processed_docs
    
    def save_processed_document(self, processed_doc: ProcessedDocument, output_dir: Path) -> Path:
        """Save processed document to disk."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create output filename
            output_filename = f"{processed_doc.file_path.stem}_processed.json"
            output_path = output_dir / output_filename
            
            # Convert to JSON-serializable format
            data = {
                "file_path": str(processed_doc.file_path),
                "content": processed_doc.content,
                "chunks": processed_doc.chunks,
                "content_length": processed_doc.metadata.get("content_length", 0),
                "metadata": processed_doc.metadata
            }
            
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved processed document to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving processed document: {str(e)}")
            raise
