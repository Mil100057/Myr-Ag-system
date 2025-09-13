#!/usr/bin/env python3
"""
LlamaIndex-enhanced Excel processor for better semantic understanding.
Combines the best of both approaches: structured data processing + semantic search.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from llama_index.core import Settings, Document
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import VectorStoreIndex

logger = logging.getLogger(__name__)


@dataclass
class LlamaIndexExcelChunk:
    """Enhanced Excel chunk with LlamaIndex integration."""
    text: str
    chunk_type: str  # 'row', 'column', 'summary', 'table', 'semantic'
    metadata: Dict[str, Any]
    sheet_name: Optional[str] = None
    row_range: Optional[tuple] = None
    column_range: Optional[tuple] = None
    semantic_score: Optional[float] = None


class LlamaIndexExcelProcessor:
    """LlamaIndex-enhanced Excel processor for better semantic understanding."""
    
    def __init__(self, max_rows_per_chunk: int = 50, enable_semantic_search: bool = True):
        """
        Initialize the LlamaIndex Excel processor.
        
        Args:
            max_rows_per_chunk: Maximum rows to include in a single chunk
            enable_semantic_search: Whether to enable LlamaIndex semantic search
        """
        self.logger = logging.getLogger(__name__)
        self.max_rows_per_chunk = max_rows_per_chunk
        self.enable_semantic_search = enable_semantic_search
        
        # Initialize LlamaIndex if enabled
        if self.enable_semantic_search:
            self._setup_llamaindex()
        else:
            self.index = None
            self.query_engine = None
    
    def _setup_llamaindex(self):
        """Set up LlamaIndex with optimized settings."""
        try:
            # Optimized Ollama settings for speed and reliability
            Settings.llm = Ollama(
                model='llama3.2:3b',
                base_url='http://localhost:11434',
                request_timeout=60.0,
                temperature=0.1,
                max_tokens=100,
                top_p=0.8
            )
            Settings.embed_model = HuggingFaceEmbedding(
                model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                trust_remote_code=False
            )
            
            self.logger.info("✅ LlamaIndex configured with Ollama")
            
        except Exception as e:
            self.logger.error(f"Failed to setup LlamaIndex: {e}")
            self.enable_semantic_search = False
    
    def process_excel_file(self, file_path: Path, docling_content: str = None) -> List[LlamaIndexExcelChunk]:
        """
        Process Excel file with LlamaIndex enhancement.
        
        Args:
            file_path: Path to the Excel file
            docling_content: Content extracted by Docling (optional)
            
        Returns:
            List of LlamaIndexExcelChunk objects
        """
        try:
            self.logger.info(f"Processing Excel file with LlamaIndex: {file_path}")
            
            # Read Excel file
            excel_data = pd.read_excel(file_path, sheet_name=None)
            
            if not excel_data:
                self.logger.warning(f"No sheets found in Excel file: {file_path}")
                return [LlamaIndexExcelChunk(
                    text=f"Excel file {file_path.name} contains no readable sheets",
                    chunk_type="error",
                    metadata={"source_file": str(file_path), "error": "no_sheets"},
                    sheet_name="unknown"
                )]
            
            chunks = []
            total_rows = 0
            
            # Process each sheet
            for sheet_name, df in excel_data.items():
                self.logger.info(f"Processing sheet: {sheet_name} ({len(df)} rows)")
                total_rows += len(df)
                
                if df.empty:
                    continue
                
                try:
                    # Create enhanced chunks for this sheet
                    sheet_chunks = self._create_enhanced_sheet_chunks(df, sheet_name, file_path)
                    chunks.extend(sheet_chunks)
                    
                except Exception as sheet_error:
                    self.logger.error(f"Error processing sheet '{sheet_name}': {sheet_error}")
                    chunks.append(LlamaIndexExcelChunk(
                        text=f"Error processing sheet '{sheet_name}': {str(sheet_error)}",
                        chunk_type="error",
                        metadata={"source_file": str(file_path), "sheet_name": sheet_name, "error": str(sheet_error)},
                        sheet_name=sheet_name
                    ))
            
            # Create semantic index if enabled
            if self.enable_semantic_search and chunks:
                try:
                    self._create_semantic_index(chunks)
                except Exception as e:
                    self.logger.error(f"Failed to create semantic index: {e}")
            
            self.logger.info(f"Successfully created {len(chunks)} enhanced chunks from {file_path}")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error processing Excel file {file_path}: {e}")
            return [LlamaIndexExcelChunk(
                text=f"Error processing Excel file: {str(e)}",
                chunk_type="error",
                metadata={"source_file": str(file_path), "error": str(e)},
                sheet_name="unknown"
            )]
    
    def _create_enhanced_sheet_chunks(self, df: pd.DataFrame, sheet_name: str, file_path: Path) -> List[LlamaIndexExcelChunk]:
        """Create enhanced chunks with both structured and semantic information."""
        chunks = []
        
        # 1. Create semantic document for the entire sheet
        semantic_doc = self._create_semantic_document(df, sheet_name, file_path)
        if semantic_doc:
            chunks.append(semantic_doc)
        
        # 2. Create detailed row chunks
        row_chunks = self._create_enhanced_row_chunks(df, sheet_name, file_path)
        chunks.extend(row_chunks)
        
        # 3. Create column analysis chunks
        column_chunks = self._create_enhanced_column_chunks(df, sheet_name, file_path)
        chunks.extend(column_chunks)
        
        # 4. Create table overview chunk
        table_chunk = self._create_enhanced_table_chunk(df, sheet_name, file_path)
        if table_chunk:
            chunks.append(table_chunk)
        
        return chunks
    
    def _create_semantic_document(self, df: pd.DataFrame, sheet_name: str, file_path: Path) -> Optional[LlamaIndexExcelChunk]:
        """Create a semantic document for the entire sheet."""
        try:
            # Create a comprehensive document that captures the essence of the data
            doc_text = f"Financial Data Analysis - {sheet_name}\n\n"
            doc_text += f"Data Structure: {df.shape[0]} rows × {df.shape[1]} columns\n"
            doc_text += f"Columns: {list(df.columns)}\n\n"
            
            # Add all data in a structured format
            doc_text += "Financial Records:\n"
            for idx, row in df.iterrows():
                if not row.isna().all():
                    row_data = []
                    for col, val in row.items():
                        if pd.notna(val) and str(val).strip():
                            row_data.append(f"{col}={val}")
                    
                    if row_data:
                        doc_text += f"Row{idx+1}: {' '.join(row_data)}\n"
            
            return LlamaIndexExcelChunk(
                text=doc_text,
                chunk_type="semantic",
                metadata={
                    "source_file": str(file_path),
                    "sheet_name": sheet_name,
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "semantic_document": True
                },
                sheet_name=sheet_name
            )
            
        except Exception as e:
            self.logger.error(f"Error creating semantic document: {e}")
            return None
    
    def _create_enhanced_row_chunks(self, df: pd.DataFrame, sheet_name: str, file_path: Path) -> List[LlamaIndexExcelChunk]:
        """Create enhanced row chunks with better semantic understanding."""
        chunks = []
        
        for idx, row in df.iterrows():
            if row.isna().all():
                continue
            
            # Create detailed row description
            row_text = self._create_intelligent_row_description(row, idx, df.columns)
            
            if row_text.strip():
                chunk = LlamaIndexExcelChunk(
                    text=row_text,
                    chunk_type="row",
                    metadata={
                        "source_file": str(file_path),
                        "sheet_name": sheet_name,
                        "row_number": idx + 1,
                        "column_count": len(row),
                        "has_data": not row.isna().all()
                    },
                    sheet_name=sheet_name,
                    row_range=(idx, idx)
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_intelligent_row_description(self, row: pd.Series, row_idx: int, columns: pd.Index) -> str:
        """Create an intelligent row description that works with any data structure."""
        row_data = []
        
        for i, (col_name, value) in enumerate(row.items()):
            if pd.isna(value) or str(value).strip() == '':
                continue
            
            # Use meaningful column names
            col_position = f"Column_{i + 1}"
            col_str = str(col_name) if not pd.isna(col_name) and str(col_name) not in ['Unnamed: 0', 'Column A', 'Column B'] else col_position
            
            # Format the value appropriately
            if pd.api.types.is_numeric_dtype(type(value)):
                val_str = f"{value:.2f}" if isinstance(value, float) else str(value)
            else:
                val_str = str(value)
            
            row_data.append(f"{col_str}: {val_str}")
        
        if row_data:
            return f"Row {row_idx + 1} data: {' | '.join(row_data)}"
        else:
            return f"Row {row_idx + 1}: (empty row)"
    
    def _create_enhanced_column_chunks(self, df: pd.DataFrame, sheet_name: str, file_path: Path) -> List[LlamaIndexExcelChunk]:
        """Create enhanced column chunks with semantic analysis."""
        chunks = []
        
        for col_name in df.columns:
            if pd.isna(col_name):
                continue
            
            col_data = df[col_name].dropna()
            if len(col_data) == 0:
                continue
            
            # Create intelligent column description
            col_text = self._create_intelligent_column_description(col_name, col_data, df)
            
            chunk = LlamaIndexExcelChunk(
                text=col_text,
                chunk_type="column",
                metadata={
                    "source_file": str(file_path),
                    "sheet_name": sheet_name,
                    "column_name": str(col_name),
                    "value_count": len(col_data),
                    "data_type": str(col_data.dtype)
                },
                sheet_name=sheet_name,
                column_range=(col_name, col_name)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_intelligent_column_description(self, col_name: str, col_data: pd.Series, df: pd.DataFrame) -> str:
        """Create an intelligent column description based on data analysis."""
        col_position = f"Column_{list(df.columns).index(col_name) + 1}"
        col_str = str(col_name) if not pd.isna(col_name) and str(col_name) not in ['Unnamed: 0', 'Column A', 'Column B'] else col_position
        
        # Analyze the column data
        if pd.api.types.is_numeric_dtype(col_data):
            numeric_vals = pd.to_numeric(col_data, errors='coerce').dropna()
            if len(numeric_vals) > 0:
                desc = f"Column '{col_str}' contains {len(col_data)} numeric values. "
                desc += f"Range: {numeric_vals.min():.2f} to {numeric_vals.max():.2f}. "
                desc += f"Average: {numeric_vals.mean():.2f}. "
                
                # Add specific values for small datasets
                if len(numeric_vals) <= 10:
                    values = [str(v) for v in numeric_vals.tolist()]
                    desc += f"Values: {', '.join(values)}."
                else:
                    desc += f"Sample values: {', '.join([str(v) for v in numeric_vals.head(3).tolist()])}..."
            else:
                desc = f"Column '{col_str}' contains {len(col_data)} values: {', '.join([str(v) for v in col_data.unique()[:5]])}."
        else:
            unique_vals = col_data.unique()
            if len(unique_vals) <= 10:
                desc = f"Column '{col_str}' contains {len(col_data)} values. "
                desc += f"Values: {', '.join([str(v) for v in unique_vals])}."
            else:
                desc = f"Column '{col_str}' contains {len(col_data)} values. "
                desc += f"Sample: {', '.join([str(v) for v in col_data.head(3).tolist()])}..."
        
        return desc
    
    def _create_enhanced_table_chunk(self, df: pd.DataFrame, sheet_name: str, file_path: Path) -> Optional[LlamaIndexExcelChunk]:
        """Create an enhanced table overview chunk."""
        if df.empty:
            return None
        
        table_text = f"Table '{sheet_name}' contains {len(df)} rows and {len(df.columns)} columns. "
        table_text += f"Columns: {', '.join([str(col) for col in df.columns if not pd.isna(col)])}. "
        
        # Add data insights
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            table_text += f"Numeric columns: {', '.join(numeric_cols)}. "
            
            # Add value ranges for numeric columns
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    table_text += f"{col} range: {col_data.min():.2f} to {col_data.max():.2f}. "
        
        return LlamaIndexExcelChunk(
            text=table_text,
            chunk_type="table",
            metadata={
                "source_file": str(file_path),
                "sheet_name": sheet_name,
                "row_count": len(df),
                "column_count": len(df.columns),
                "numeric_columns": list(numeric_cols)
            },
            sheet_name=sheet_name
        )
    
    def _create_semantic_index(self, chunks: List[LlamaIndexExcelChunk]):
        """Create a semantic index for the chunks."""
        try:
            # Extract text from chunks
            documents = []
            for chunk in chunks:
                if chunk.chunk_type in ['semantic', 'row', 'column']:
                    doc = Document(text=chunk.text, metadata=chunk.metadata)
                    documents.append(doc)
            
            if not documents:
                self.logger.warning("No documents to index")
                return
            
            # Parse into nodes
            node_parser = SimpleNodeParser.from_defaults(
                chunk_size=1000,
                chunk_overlap=100
            )
            nodes = node_parser.get_nodes_from_documents(documents)
            
            # Create vector index
            self.index = VectorStoreIndex(nodes)
            self.query_engine = self.index.as_query_engine(
                response_mode="compact",
                similarity_top_k=3,
                verbose=False
            )
            
            self.logger.info(f"Created semantic index with {len(nodes)} nodes")
            
        except Exception as e:
            self.logger.error(f"Error creating semantic index: {e}")
            self.enable_semantic_search = False
    
    def query_semantic(self, question: str) -> str:
        """Query the semantic index."""
        if not self.query_engine:
            return "Semantic search not available"
        
        try:
            response = self.query_engine.query(question)
            return str(response)
        except Exception as e:
            self.logger.error(f"Semantic query error: {e}")
            return f"Query error: {e}"
    
    def get_direct_answer(self, question: str, chunks: List[LlamaIndexExcelChunk]) -> str:
        """Get direct answer from chunks without LLM."""
        question_lower = question.lower()
        
        # Look for specific patterns in the question
        if any(word in question_lower for word in ['salary', 'earn', 'income', 'gain']):
            # Find the largest positive number in row chunks
            max_value = None
            for chunk in chunks:
                if chunk.chunk_type == 'row' and 'data' in chunk.text:
                    # Extract numeric values from the chunk text
                    import re
                    numbers = re.findall(r'[-+]?\d*\.?\d+', chunk.text)
                    for num_str in numbers:
                        try:
                            num = float(num_str)
                            if num > 0 and (max_value is None or num > max_value):
                                max_value = num
                        except ValueError:
                            continue
            
            return str(max_value) if max_value is not None else "No salary data found"
        
        elif any(word in question_lower for word in ['largest', 'highest', 'maximum', 'biggest']):
            # Find the largest value
            max_value = None
            for chunk in chunks:
                if chunk.chunk_type in ['row', 'semantic']:
                    import re
                    numbers = re.findall(r'[-+]?\d*\.?\d+', chunk.text)
                    for num_str in numbers:
                        try:
                            num = float(num_str)
                            if max_value is None or num > max_value:
                                max_value = num
                        except ValueError:
                            continue
            
            return str(max_value) if max_value is not None else "No data found"
        
        return "No direct answer available"
