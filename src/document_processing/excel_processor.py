"""
Excel-specific processor for enhanced document processing.
Works with Docling output to create better chunks for LLM queries.
"""
import pandas as pd
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExcelChunk:
    """Represents a chunk of Excel data with enhanced metadata."""
    text: str
    chunk_type: str  # 'row', 'column', 'summary', 'table'
    metadata: Dict[str, Any]
    sheet_name: Optional[str] = None
    row_range: Optional[tuple] = None
    column_range: Optional[tuple] = None


class ExcelProcessor:
    """Enhanced Excel processor for better LLM querying."""
    
    def __init__(self, max_rows_per_chunk: int = 50, enable_grouping: bool = True):
        """
        Initialize the Excel processor.
        
        Args:
            max_rows_per_chunk: Maximum rows to include in a single chunk (for large files)
            enable_grouping: Whether to enable row grouping for better performance
        """
        self.logger = logging.getLogger(__name__)
        self.max_rows_per_chunk = max_rows_per_chunk
        self.enable_grouping = enable_grouping
    
    def process_excel_file(self, file_path: Path, docling_content: str) -> List[ExcelChunk]:
        """
        Process Excel file with enhanced chunking for better LLM queries.
        
        Args:
            file_path: Path to the Excel file
            docling_content: Content extracted by Docling
            
        Returns:
            List of ExcelChunk objects
        """
        try:
            # Read Excel file with pandas for structured processing
            self.logger.info(f"Reading Excel file: {file_path}")
            excel_data = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
            
            if not excel_data:
                self.logger.warning(f"No sheets found in Excel file: {file_path}")
                return [ExcelChunk(
                    text=f"Excel file {file_path.name} contains no readable sheets",
                    chunk_type="error",
                    metadata={"source_file": str(file_path), "error": "no_sheets"},
                    sheet_name="unknown"
                )]
            
            chunks = []
            total_rows = 0
            
            for sheet_name, df in excel_data.items():
                self.logger.info(f"Processing Excel sheet: {sheet_name} ({len(df)} rows)")
                total_rows += len(df)
                
                # Skip empty sheets
                if df.empty:
                    self.logger.warning(f"Sheet '{sheet_name}' is empty, skipping")
                    continue
                
                try:
                    # Create different types of chunks for this sheet
                    sheet_chunks = self._create_sheet_chunks(df, sheet_name, file_path)
                    chunks.extend(sheet_chunks)
                    self.logger.info(f"Created {len(sheet_chunks)} chunks from sheet '{sheet_name}'")
                    
                except Exception as sheet_error:
                    self.logger.error(f"Error processing sheet '{sheet_name}': {sheet_error}")
                    # Add error chunk for this sheet
                    chunks.append(ExcelChunk(
                        text=f"Error processing sheet '{sheet_name}': {str(sheet_error)}",
                        chunk_type="error",
                        metadata={"source_file": str(file_path), "sheet_name": sheet_name, "error": str(sheet_error)},
                        sheet_name=sheet_name
                    ))
            
            # Add summary chunks
            try:
                summary_chunks = self._create_summary_chunks(excel_data, file_path)
                chunks.extend(summary_chunks)
            except Exception as summary_error:
                self.logger.error(f"Error creating summary chunks: {summary_error}")
            
            self.logger.info(f"Successfully created {len(chunks)} Excel chunks from {file_path} ({total_rows} total rows)")
            return chunks
            
        except FileNotFoundError:
            self.logger.error(f"Excel file not found: {file_path}")
            return [ExcelChunk(
                text=f"Excel file not found: {file_path}",
                chunk_type="error",
                metadata={"source_file": str(file_path), "error": "file_not_found"},
                sheet_name="unknown"
            )]
        except pd.errors.EmptyDataError:
            self.logger.error(f"Excel file is empty: {file_path}")
            return [ExcelChunk(
                text=f"Excel file is empty: {file_path}",
                chunk_type="error",
                metadata={"source_file": str(file_path), "error": "empty_file"},
                sheet_name="unknown"
            )]
        except Exception as e:
            self.logger.error(f"Unexpected error processing Excel file {file_path}: {e}")
            # Fallback to original Docling content
            return [ExcelChunk(
                text=docling_content or f"Error processing Excel file: {str(e)}",
                chunk_type="fallback",
                metadata={"source": "docling_fallback", "file_path": str(file_path), "error": str(e)},
                sheet_name="unknown"
            )]
    
    def _create_sheet_chunks(self, df: pd.DataFrame, sheet_name: str, file_path: Path) -> List[ExcelChunk]:
        """Create various types of chunks from a single sheet."""
        chunks = []
        
        # 1. Row-based chunks (each row as a natural language description)
        row_chunks = self._create_row_chunks(df, sheet_name, file_path)
        chunks.extend(row_chunks)
        
        # 2. Column-based chunks (each column as a summary)
        column_chunks = self._create_column_chunks(df, sheet_name, file_path)
        chunks.extend(column_chunks)
        
        # 3. Table structure chunk (overview of the table)
        table_chunk = self._create_table_chunk(df, sheet_name, file_path)
        if table_chunk:
            chunks.append(table_chunk)
        
        return chunks
    
    def _create_row_chunks(self, df: pd.DataFrame, sheet_name: str, file_path: Path) -> List[ExcelChunk]:
        """Create row-based chunks with natural language descriptions."""
        chunks = []
        
        # For large files, group rows to avoid too many chunks
        if self.enable_grouping and len(df) > self.max_rows_per_chunk:
            chunks.extend(self._create_grouped_row_chunks(df, sheet_name, file_path))
        else:
            # Individual row chunks for smaller files
            for idx, row in df.iterrows():
                # Skip empty rows
                if row.isna().all():
                    continue
                
                # Create detailed row description with actual values
                row_text = self._create_detailed_row_description(row, idx, df.columns)
                
                if row_text.strip():
                    chunk = ExcelChunk(
                        text=row_text,
                        chunk_type="row",
                        metadata={
                            "source_file": str(file_path),
                            "sheet_name": sheet_name,
                            "row_number": idx + 1,  # 1-indexed
                            "column_count": len(row),
                            "has_data": not row.isna().all()
                        },
                        sheet_name=sheet_name,
                        row_range=(idx, idx)
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def _create_grouped_row_chunks(self, df: pd.DataFrame, sheet_name: str, file_path: Path) -> List[ExcelChunk]:
        """Create grouped row chunks for large files to improve performance."""
        chunks = []
        
        # Group rows into chunks of max_rows_per_chunk
        for start_idx in range(0, len(df), self.max_rows_per_chunk):
            end_idx = min(start_idx + self.max_rows_per_chunk, len(df))
            group_df = df.iloc[start_idx:end_idx]
            
            # Create detailed data chunks instead of summaries
            for idx, row in group_df.iterrows():
                if not row.isna().all():
                    # Create a more detailed row description with actual values
                    row_text = self._create_detailed_row_description(row, idx, df.columns)
                    
                    if row_text.strip():
                        chunk = ExcelChunk(
                            text=row_text,
                            chunk_type="row",
                            metadata={
                                "source_file": str(file_path),
                                "sheet_name": sheet_name,
                                "row_number": idx + 1,
                                "column_count": len(row),
                                "has_data": not row.isna().all(),
                                "grouped": True
                            },
                            sheet_name=sheet_name,
                            row_range=(idx, idx)
                        )
                        chunks.append(chunk)
        
        return chunks
    
    def _create_detailed_row_description(self, row: pd.Series, row_idx: int, columns: pd.Index) -> str:
        """Create a detailed row description with actual values for better searchability."""
        # Create a structured description with actual values
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
    
    def _create_column_chunks(self, df: pd.DataFrame, sheet_name: str, file_path: Path) -> List[ExcelChunk]:
        """Create column-based chunks with summaries."""
        chunks = []
        
        for col_name in df.columns:
            if pd.isna(col_name):
                continue
            
            # Get non-null values in this column
            col_data = df[col_name].dropna()
            
            if len(col_data) == 0:
                continue
            
            # Create column summary
            col_text = self._describe_column(col_name, col_data, df)
            
            chunk = ExcelChunk(
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
    
    def _create_table_chunk(self, df: pd.DataFrame, sheet_name: str, file_path: Path) -> Optional[ExcelChunk]:
        """Create a table overview chunk."""
        if df.empty:
            return None
        
        # Create table description
        table_text = f"Table '{sheet_name}' contains {len(df)} rows and {len(df.columns)} columns. "
        table_text += f"Columns: {', '.join([str(col) for col in df.columns if not pd.isna(col)])}. "
        
        # Add data summary
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            table_text += f"Numeric columns: {', '.join(numeric_cols)}. "
        
        # Add sample data
        if len(df) > 0:
            sample_row = df.iloc[0]
            sample_data = []
            for col, val in sample_row.items():
                if not pd.isna(val) and str(val).strip():
                    sample_data.append(f"{col}={val}")
            if sample_data:
                table_text += f"Sample data: {', '.join(sample_data[:5])}."
        
        return ExcelChunk(
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
    
    def _create_summary_chunks(self, excel_data: Dict[str, pd.DataFrame], file_path: Path) -> List[ExcelChunk]:
        """Create summary chunks for the entire Excel file."""
        chunks = []
        
        # Overall file summary
        total_sheets = len(excel_data)
        total_rows = sum(len(df) for df in excel_data.values())
        
        summary_text = f"Excel file contains {total_sheets} sheet(s) with {total_rows} total rows. "
        summary_text += f"Sheets: {', '.join(excel_data.keys())}. "
        
        # Add insights about the data
        all_numeric_data = []
        for sheet_name, df in excel_data.items():
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    all_numeric_data.extend(col_data.tolist())
        
        if all_numeric_data:
            summary_text += f"Contains {len(all_numeric_data)} numeric values. "
            summary_text += f"Value range: {min(all_numeric_data):.2f} to {max(all_numeric_data):.2f}."
        
        chunks.append(ExcelChunk(
            text=summary_text,
            chunk_type="summary",
            metadata={
                "source_file": str(file_path),
                "sheet_count": total_sheets,
                "total_rows": total_rows,
                "numeric_values": len(all_numeric_data)
            }
        ))
        
        return chunks
    
    def _infer_column_type(self, col_name: str, col_data: pd.Series) -> str:
        """
        Dynamically infer the type of a column based on DATA CONTENT, not just column names.
        This approach works better with real-world Excel files that have generic column names.
        
        Args:
            col_name: Name of the column
            col_data: Series containing the column data
            
        Returns:
            Inferred column type: 'date', 'amount', 'category', 'text', 'id', 'position'
        """
        # First, analyze the actual data content
        data_analysis = self._analyze_column_data(col_data)
        
        # If we have strong data-based evidence, use that
        if data_analysis['confidence'] > 0.7:
            return data_analysis['type']
        
        # Fallback to column name analysis (but with lower priority)
        col_name_lower = str(col_name).lower()
        
        # Check for date columns
        if pd.api.types.is_datetime64_any_dtype(col_data):
            return "date"
        elif any(keyword in col_name_lower for keyword in ['date', 'time', 'created', 'updated', 'jour', 'heure', 'timestamp']):
            return "date"
        
        # Check for amount/value columns
        elif pd.api.types.is_numeric_dtype(col_data):
            if any(keyword in col_name_lower for keyword in ['amount', 'value', 'price', 'cost', 'salary', 'montant', 'prix', 'total', 'sum']):
                return "amount"
            else:
                return "amount"  # Default numeric to amount
        
        # Check for ID columns
        elif any(keyword in col_name_lower for keyword in ['id', 'key', 'index', 'ref', 'reference', 'code']):
            return "id"
        
        # Check for category/name columns
        elif any(keyword in col_name_lower for keyword in ['name', 'type', 'category', 'description', 'reason', 'nom', 'type', 'categorie', 'raison', 'status', 'state']):
            return "category"
        
        # Use data analysis result as fallback
        else:
            return data_analysis['type']
    
    def _analyze_column_data(self, col_data: pd.Series) -> Dict[str, Any]:
        """
        Analyze column data content to determine type and confidence.
        This is the core improvement - data-driven type inference.
        """
        # Remove null values for analysis
        clean_data = col_data.dropna()
        
        if len(clean_data) == 0:
            return {'type': 'text', 'confidence': 0.0, 'reason': 'empty_column'}
        
        # Convert to string for pattern analysis
        str_data = clean_data.astype(str)
        
        # 1. Check for numeric patterns (amounts, quantities, etc.)
        numeric_score = self._analyze_numeric_patterns(clean_data, str_data)
        
        # 2. Check for date patterns
        date_score = self._analyze_date_patterns(clean_data, str_data)
        
        # 3. Check for ID patterns (sequential, unique, etc.)
        id_score = self._analyze_id_patterns(clean_data, str_data)
        
        # 4. Check for category patterns (limited unique values, categorical)
        category_score = self._analyze_category_patterns(clean_data, str_data)
        
        # 5. Check for positional patterns (row numbers, indices)
        position_score = self._analyze_position_patterns(clean_data, str_data)
        
        # Find the highest scoring type
        scores = {
            'amount': numeric_score,
            'date': date_score,
            'id': id_score,
            'category': category_score,
            'position': position_score
        }
        
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        return {
            'type': best_type,
            'confidence': confidence,
            'scores': scores,
            'reason': f"data_analysis_{best_type}"
        }
    
    def _analyze_numeric_patterns(self, clean_data: pd.Series, str_data: pd.Series) -> float:
        """Analyze if column contains numeric/amount data."""
        if not pd.api.types.is_numeric_dtype(clean_data):
            return 0.0
        
        # Check for monetary patterns
        monetary_patterns = [r'^\d+\.\d{2}$', r'^\d+,\d{2}$', r'^\d+$']
        monetary_matches = sum(1 for val in str_data if any(re.match(pattern, val) for pattern in monetary_patterns))
        
        # Check for percentage patterns
        percentage_matches = sum(1 for val in str_data if '%' in val or 'percent' in val.lower())
        
        # Check value ranges (typical for amounts)
        if pd.api.types.is_numeric_dtype(clean_data):
            numeric_vals = pd.to_numeric(clean_data, errors='coerce').dropna()
            if len(numeric_vals) > 0:
                # Typical amount ranges
                if 0.01 <= numeric_vals.min() <= 1000000 and 0.01 <= numeric_vals.max() <= 1000000:
                    return 0.8 + (monetary_matches / len(clean_data)) * 0.2
        
        return 0.5 if monetary_matches > len(clean_data) * 0.3 else 0.0
    
    def _analyze_date_patterns(self, clean_data: pd.Series, str_data: pd.Series) -> float:
        """Analyze if column contains date data."""
        if pd.api.types.is_datetime64_any_dtype(clean_data):
            return 0.9
        
        # Common date patterns
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
            r'^\d{2}-\d{2}-\d{4}$',  # MM-DD-YYYY
            r'^\d{1,2}/\d{1,2}/\d{2,4}$',  # M/D/YY or M/D/YYYY
        ]
        
        date_matches = sum(1 for val in str_data if any(re.match(pattern, val) for pattern in date_patterns))
        
        # Check for month names
        month_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        month_matches = sum(1 for val in str_data if any(month in val.lower() for month in month_names))
        
        total_matches = date_matches + month_matches
        return min(0.9, total_matches / len(clean_data) * 1.5)
    
    def _analyze_id_patterns(self, clean_data: pd.Series, str_data: pd.Series) -> float:
        """Analyze if column contains ID data."""
        # Check for sequential patterns
        if pd.api.types.is_numeric_dtype(clean_data):
            numeric_vals = pd.to_numeric(clean_data, errors='coerce').dropna()
            if len(numeric_vals) > 1:
                # Check if values are sequential or have ID-like patterns
                if numeric_vals.min() >= 1 and numeric_vals.max() <= len(numeric_vals) * 2:
                    return 0.7
        
        # Check for alphanumeric ID patterns
        id_patterns = [r'^[A-Z]{2,}\d+$', r'^\d{4,}$', r'^[A-Z]\d+$']
        id_matches = sum(1 for val in str_data if any(re.match(pattern, val) for pattern in id_patterns))
        
        # Check uniqueness (IDs should be mostly unique)
        uniqueness = len(clean_data.unique()) / len(clean_data)
        
        return min(0.8, (id_matches / len(clean_data)) * 0.5 + uniqueness * 0.3)
    
    def _analyze_category_patterns(self, clean_data: pd.Series, str_data: pd.Series) -> float:
        """Analyze if column contains categorical data."""
        # Check for limited unique values (typical of categories)
        unique_ratio = len(clean_data.unique()) / len(clean_data)
        
        if unique_ratio < 0.1:  # Less than 10% unique values
            return 0.8
        elif unique_ratio < 0.3:  # Less than 30% unique values
            return 0.6
        elif unique_ratio < 0.5:  # Less than 50% unique values
            return 0.4
        
        # Check for common category keywords
        category_keywords = ['yes', 'no', 'true', 'false', 'active', 'inactive', 'pending', 'completed', 'open', 'closed']
        keyword_matches = sum(1 for val in str_data if val.lower() in category_keywords)
        
        # Check for financial/transaction categories
        financial_keywords = ['salaire', 'salary', 'expense', 'depense', 'cost', 'income', 'airbus', 'company', 'freelance', 'transport', 'food']
        financial_matches = sum(1 for val in str_data if any(keyword in val.lower() for keyword in financial_keywords))
        
        # Check for month names (categorical)
        month_keywords = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        month_matches = sum(1 for val in str_data if any(month in val.lower() for month in month_keywords))
        
        total_matches = keyword_matches + financial_matches + month_matches
        return min(0.8, total_matches / len(clean_data) * 1.5)
    
    def _analyze_position_patterns(self, clean_data: pd.Series, str_data: pd.Series) -> float:
        """Analyze if column contains positional/row data."""
        if pd.api.types.is_numeric_dtype(clean_data):
            numeric_vals = pd.to_numeric(clean_data, errors='coerce').dropna()
            if len(numeric_vals) > 0:
                # Check if values are sequential starting from 1 or 0
                if (numeric_vals.min() in [0, 1] and 
                    numeric_vals.max() <= len(numeric_vals) + 5 and
                    len(numeric_vals.unique()) / len(numeric_vals) > 0.8):
                    return 0.9
        
        return 0.0
    
    def _describe_row(self, row: pd.Series, row_idx: int) -> str:
        """Create a natural language description of a row using DATA-DRIVEN pattern detection."""
        # Analyze the actual data content, not just column names
        row_data = {}
        column_types = {}
        
        for col_name, value in row.items():
            if pd.isna(value) or str(value).strip() == '':
                continue
            
            # Use position-based naming for generic columns
            col_position = f"Column_{list(row.index).index(col_name) + 1}"
            col_str = str(col_name) if not pd.isna(col_name) and str(col_name) not in ['Unnamed: 0', 'Column A', 'Column B'] else col_position
            val_str = str(value)
            row_data[col_str] = val_str
            
            # Infer column type based on data content
            col_data = pd.Series([value])
            column_types[col_str] = self._infer_column_type(col_name, col_data)
        
        # Group data by inferred types
        amount_data = []
        category_data = []
        date_data = []
        id_data = []
        text_data = []
        position_data = []
        
        for col_name, value in row_data.items():
            col_type = column_types.get(col_name, 'text')
            val_str = str(value)
            
            if col_type == 'amount':
                amount_data.append((col_name, val_str))
            elif col_type == 'category':
                category_data.append((col_name, val_str))
            elif col_type == 'date':
                date_data.append((col_name, val_str))
            elif col_type == 'id':
                id_data.append((col_name, val_str))
            elif col_type == 'position':
                position_data.append((col_name, val_str))
            else:
                text_data.append((col_name, val_str))
        
        # Create context-aware descriptions based on data patterns
        description_parts = []
        
        # 1. Handle financial/transaction patterns
        if amount_data and category_data:
            # Look for common financial patterns
            amounts = [val for _, val in amount_data]
            categories = [val for _, val in category_data]
            
            # Check if this looks like a salary/income entry
            if any('salaire' in cat.lower() or 'salary' in cat.lower() or 'airbus' in cat.lower() for cat in categories):
                # Find the largest amount (likely the salary)
                amount_vals = [float(val) for val in amounts if val.replace('.', '').replace(',', '').isdigit()]
                if amount_vals:
                    amount_val = max(amount_vals)
                    return f"Salary Entry {row_idx + 1}: {categories[0]} - Amount {amount_val}"
                else:
                    return f"Salary Entry {row_idx + 1}: {categories[0]} - Amount {amounts[0]}"
            
            # Check if this looks like an expense
            elif any('expense' in cat.lower() or 'depense' in cat.lower() or 'cost' in cat.lower() for cat in categories):
                # Find the expense amount
                amount_vals = [float(val) for val in amounts if val.replace('.', '').replace(',', '').isdigit()]
                if amount_vals:
                    amount_val = min(amount_vals)  # Expenses are usually smaller
                    return f"Expense Entry {row_idx + 1}: {categories[0]} - Amount {amount_val}"
                else:
                    return f"Expense Entry {row_idx + 1}: {categories[0]} - Amount {amounts[0]}"
            
            # Generic financial pattern
            else:
                category_part = ', '.join(categories[:2])
                amount_part = ', '.join(amounts[:2])
                description_parts.append(f"{category_part} | {amount_part}")
        
        # 1.5. Handle financial patterns with text data (when category_data is empty but we have text)
        elif amount_data and text_data:
            amounts = [val for _, val in amount_data]
            texts = [val for _, val in text_data]
            
            # Check if text contains financial keywords
            if any('salaire' in text.lower() or 'salary' in text.lower() or 'airbus' in text.lower() for text in texts):
                # Find the largest amount (likely the salary)
                amount_vals = [float(val) for val in amounts if val.replace('.', '').replace(',', '').isdigit()]
                if amount_vals:
                    amount_val = max(amount_vals)
                    return f"Salary Entry {row_idx + 1}: {texts[0]} - Amount {amount_val}"
                else:
                    return f"Salary Entry {row_idx + 1}: {texts[0]} - Amount {amounts[0]}"
            
            elif any('expense' in text.lower() or 'depense' in text.lower() or 'cost' in text.lower() for text in texts):
                # Find the expense amount
                amount_vals = [float(val) for val in amounts if val.replace('.', '').replace(',', '').isdigit()]
                if amount_vals:
                    amount_val = min(amount_vals)  # Expenses are usually smaller
                    return f"Expense Entry {row_idx + 1}: {texts[0]} - Amount {amount_val}"
                else:
                    return f"Expense Entry {row_idx + 1}: {texts[0]} - Amount {amounts[0]}"
            
            # Generic amount + text pattern
            else:
                text_part = ', '.join(texts[:2])
                amount_part = ', '.join(amounts[:2])
                description_parts.append(f"{text_part} | {amount_part}")
        
        # 2. Handle date-based patterns
        elif date_data and (amount_data or category_data):
            date_val = date_data[0][1] if date_data else "unknown"
            other_data = amount_data + category_data
            other_part = ', '.join([val for _, val in other_data[:2]])
            description_parts.append(f"Date: {date_val} | {other_part}")
        
        # 3. Handle ID-based patterns
        elif id_data and (amount_data or category_data):
            id_val = id_data[0][1] if id_data else "unknown"
            other_data = amount_data + category_data
            other_part = ', '.join([val for _, val in other_data[:2]])
            description_parts.append(f"ID: {id_val} | {other_part}")
        
        # 4. Handle pure numeric data
        elif amount_data and not category_data:
            amounts = [val for _, val in amount_data]
            if len(amounts) == 1:
                description_parts.append(f"Value: {amounts[0]}")
            else:
                description_parts.append(f"Values: {', '.join(amounts[:3])}")
        
        # 5. Handle categorical data
        elif category_data and not amount_data:
            categories = [val for _, val in category_data]
            description_parts.append(f"Categories: {', '.join(categories[:3])}")
        
        # 6. Handle positional data (row numbers, indices)
        elif position_data:
            positions = [val for _, val in position_data]
            other_data = amount_data + category_data + text_data
            if other_data:
                other_part = ', '.join([val for _, val in other_data[:2]])
                description_parts.append(f"Position {positions[0]} | {other_part}")
            else:
                description_parts.append(f"Position: {positions[0]}")
        
        # 7. Fallback: mix of all data types
        else:
            all_data = amount_data + category_data + date_data + id_data + text_data
            if all_data:
                data_part = ', '.join([val for _, val in all_data[:4]])
                description_parts.append(data_part)
        
        # Build final description
        if description_parts:
            return f"Row {row_idx + 1}: {' | '.join(description_parts)}"
        else:
            return f"Row {row_idx + 1}: (empty row)"
    
    def _describe_column(self, col_name: str, col_data: pd.Series, df: pd.DataFrame) -> str:
        """Create a natural language description of a column using DATA-DRIVEN analysis."""
        # Use position-based naming for generic columns
        col_position = f"Column_{list(df.columns).index(col_name) + 1}"
        col_str = str(col_name) if not pd.isna(col_name) and str(col_name) not in ['Unnamed: 0', 'Column A', 'Column B'] else col_position
        
        # Infer column type based on data content
        col_type = self._infer_column_type(col_name, col_data)
        
        # Basic column info
        desc = f"Column '{col_str}' ({col_type}) contains {len(col_data)} values. "
        
        # Type-specific descriptions based on actual data patterns
        if col_type == 'amount':
            if pd.api.types.is_numeric_dtype(col_data):
                numeric_vals = pd.to_numeric(col_data, errors='coerce').dropna()
                if len(numeric_vals) > 0:
                    desc += f"Values range from {numeric_vals.min():.2f} to {numeric_vals.max():.2f}. "
                    desc += f"Average: {numeric_vals.mean():.2f}. "
                    
                    # Check for common financial patterns
                    if 1000 <= numeric_vals.max() <= 10000:
                        desc += "Appears to contain salary or income amounts. "
                    elif numeric_vals.max() < 1000:
                        desc += "Contains smaller monetary values. "
                    
                    # Add specific values for small datasets
                    if len(numeric_vals) <= 10:
                        values = [str(v) for v in numeric_vals.tolist()]
                        desc += f"Values: {', '.join(values)}."
            else:
                desc += f"Contains monetary or numeric values: {', '.join([str(v) for v in col_data.unique()[:5]])}."
        
        elif col_type == 'date':
            if pd.api.types.is_datetime64_any_dtype(col_data):
                desc += f"Date range: {col_data.min().strftime('%Y-%m-%d')} to {col_data.max().strftime('%Y-%m-%d')}. "
            else:
                desc += f"Contains date information: {', '.join([str(v) for v in col_data.unique()[:5]])}."
        
        elif col_type == 'category':
            unique_vals = col_data.unique()
            if len(unique_vals) <= 10:
                desc += f"Categories: {', '.join([str(v) for v in unique_vals])}."
            else:
                desc += f"Contains {len(unique_vals)} unique categories. "
                desc += f"Most common: {', '.join([str(v) for v in col_data.value_counts().head(3).index])}."
        
        elif col_type == 'id':
            desc += f"Contains {len(col_data)} unique identifiers. "
            if len(col_data) <= 10:
                desc += f"IDs: {', '.join([str(v) for v in col_data.tolist()])}."
            else:
                desc += f"Sample IDs: {', '.join([str(v) for v in col_data.head(3).tolist()])}..."
        
        elif col_type == 'position':
            desc += f"Contains positional data (row numbers, indices). "
            if len(col_data) <= 10:
                desc += f"Values: {', '.join([str(v) for v in col_data.tolist()])}."
            else:
                desc += f"Range: {col_data.min()} to {col_data.max()}."
        
        else:  # text
            unique_vals = col_data.unique()
            if len(unique_vals) <= 10:
                desc += f"Values: {', '.join([str(v) for v in unique_vals])}."
            else:
                desc += f"Contains {len(unique_vals)} unique text values. "
                desc += f"Sample: {', '.join([str(v) for v in col_data.head(3).tolist()])}..."
        
        return desc
