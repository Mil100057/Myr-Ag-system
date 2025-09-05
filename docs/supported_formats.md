# Supported Document Formats - Myr-Ag

## Overview

Myr-Ag supports a wide range of document formats through the powerful Docling library, providing comprehensive document processing capabilities for various use cases.

## Complete Format Support

### Office Documents
- **DOCX** - Microsoft Word documents
- **XLSX** - Microsoft Excel spreadsheets with enhanced processing (⚠️ EXPERIMENTAL)
  - **Row-based chunking**: Each row becomes a searchable chunk
  - **Column-aware processing**: Automatic detection of amounts, names, dates
  - **Natural language descriptions**: "Employee: John Doe | Salary: 50000"
  - **Multi-sheet support**: Processes all sheets in the workbook
  - **Pattern recognition**: Works with any Excel structure (named/unnamed columns)
  - **Query examples**: "What is John's salary?", "Show me all sales employees"
  - **⚠️ Note**: Excel processing uses LlamaIndex (experimental) and may have limitations
- **PPTX** - Microsoft PowerPoint presentations

### PDF Documents
- **PDF** - Standard PDF documents
- **Scanned PDFs** - OCR-enabled processing for scanned documents
- **Web-printed PDFs** - Fallback processing with pypdf

### 🌐 Web Content
- **HTML** - Web pages and HTML documents
- **XHTML** - Extended HTML documents

### Text Formats
- **TXT** - Plain text files
- **MD** - Markdown documents
- **AsciiDoc** - AsciiDoc documentation format

### Data Formats
- **CSV** - Comma-separated values files

### 🖼️ Image Formats (with OCR)
- **PNG** - Portable Network Graphics
- **JPEG** - Joint Photographic Experts Group
- **TIFF** - Tagged Image File Format
- **BMP** - Bitmap images
- **WEBP** - WebP images

### Specialized Formats
- **XML** - Extensible Markup Language
- **USPTO XML** - USPTO patent documents
- **JATS XML** - Journal Article Tag Suite (scientific articles)

## Processing Capabilities

### OCR (Optical Character Recognition)
- **Automatic text extraction** from scanned documents
- **Image processing** for text within images
- **Multi-language support** for various scripts
- **High accuracy** text recognition

### Excel Processing
- **Table structure preservation** in XLSX files
- **Cell content extraction** with formatting awareness
- **Multiple sheet support** for complex workbooks
- **Data type recognition** (dates, numbers, text)

### PDF Processing
- **Native PDF text extraction** for digital PDFs
- **OCR fallback** for scanned PDFs
- **Layout preservation** for complex documents
- **Metadata extraction** (title, author, creation date)

## Processing Statistics

Based on real-world testing with the Myr-Ag system:

| Format | Processing Speed | Accuracy | Special Features |
|--------|------------------|----------|------------------|
| PDF | ~20-30 seconds | High | OCR + Layout preservation |
| DOCX | ~5-10 seconds | Very High | Structure preservation |
| XLSX | ~1-2 seconds | Very High | Table structure + Data types |
| PPTX | ~10-15 seconds | High | Slide content extraction |
| Images | ~15-25 seconds | High | OCR text extraction |
| HTML | ~2-5 seconds | Very High | Clean text extraction |

## Enhanced Excel Processing (EXPERIMENTAL)

**⚠️ Important Note**: Excel processing uses LlamaIndex and is currently experimental. This feature may have limitations or unexpected behavior. Use with caution in production environments.

### **Smart Chunking Strategy**
Excel files receive special processing that transforms tabular data into searchable, queryable content:

#### **Row-Based Processing**
- **Individual chunks**: Each row becomes a separate searchable chunk
- **Context preservation**: Maintains relationships between columns
- **Granular search**: Find specific rows, not entire tables

#### **Column Detection**
- **Amount columns**: Automatically identifies numeric values (salary, price, quantity)
- **Name columns**: Detects text fields (employee, product, category)
- **Date columns**: Recognizes temporal data (hire date, transaction date)
- **Pattern recognition**: Works with any column naming convention

#### **Natural Language Descriptions**
**Before (Raw Excel):**
```
| Employee | Department | Salary | Hire_Date |
|----------|------------|--------|-----------|
| John Doe | Sales      | 50000  | 2020-01-15|
```

**After (Searchable Chunks):**
```
Row 1: Employee: John Doe, Department: Sales | Salary: 50000
```

### **Supported Excel Patterns**
- ✅ **Named columns**: `Employee`, `Salary`, `Date`, `Category`
- ✅ **Unnamed columns**: `Unnamed: 1`, `Unnamed: 2`, `Unnamed: 3`
- ✅ **Mixed data types**: Text, numbers, dates, formulas
- ✅ **Multiple sheets**: Processes all sheets in the workbook
- ✅ **Any language**: English and French column names supported

### **Query Examples**
- **Employee data**: "What is John Doe's salary?" → Finds specific employee information
- **Department filtering**: "Show me all employees in Sales" → Filters by department
- **Date queries**: "Find employees hired after 2020" → Date-based filtering
- **Financial data**: "What is the total revenue for Q1?" → Aggregation queries
- **Threshold filtering**: "Show me all expenses over $1000" → Value comparisons

## Usage Examples

### Uploading Documents
```bash
# Upload multiple formats via API
curl -X POST -F "files=@document.pdf" -F "files=@spreadsheet.xlsx" http://localhost:8199/documents/upload
```

### Processing Status
```bash
# Check processing status
curl -X GET http://localhost:8199/system/processing-status
```

### Querying Documents
```bash
# Query across all document types
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What is the budget for January 2021?", "n_chunks": 5}' \
  http://localhost:8199/query
```

## ⚠️ Limitations

### File Size
- **Maximum file size**: 100MB per file
- **Large documents**: May take longer to process
- **Memory usage**: ~2-3GB during processing

### Processing Time
- **Small files** (< 1MB): 1-5 seconds
- **Medium files** (1-10MB): 5-30 seconds
- **Large files** (10-100MB): 30-300 seconds

### Quality Considerations
- **Scanned documents**: OCR accuracy depends on image quality
- **Complex layouts**: May not preserve exact formatting
- **Handwritten text**: Limited OCR support

## Fallback Mechanisms

### PDF Processing
1. **Primary**: Docling with OCR
2. **Fallback**: pypdf for problematic PDFs
3. **Result**: High success rate for all PDF types

### Text Extraction
1. **Primary**: Docling for complex formats
2. **Fallback**: Direct text reading for simple formats
3. **Result**: Comprehensive format coverage

## Performance Optimization

### Chunking Strategy
- **Chunk size**: 400 characters (optimized for sentence boundaries)
- **Overlap**: 100 characters (ensures context preservation)
- **Strategy**: Hybrid approach combining paragraph and sentence splitting

### Vector Storage
- **Database**: LEANN with 97% space savings
- **Embeddings**: 384-dimensional vectors
- **Model**: paraphrase-multilingual-MiniLM-L12-v2

## Best Practices

### Document Preparation
1. **Use high-quality scans** for OCR processing
2. **Ensure proper file extensions** for format detection
3. **Keep file sizes reasonable** (< 50MB recommended)
4. **Use standard formats** when possible

### Query Optimization
1. **Be specific** in your questions
2. **Use relevant keywords** from your documents
3. **Adjust chunk count** based on document complexity
4. **Test with different models** for better results

## Technical Details

### Docling Integration
- **Version**: Latest stable release
- **Backend**: PyTorch with MPS acceleration
- **OCR Engines**: Tesseract, EasyOCR, RapidOCR
- **Language Detection**: Automatic

### System Requirements
- **Python**: 3.11+
- **Memory**: 4GB+ RAM recommended
- **Storage**: 1GB+ for models and indices
- **GPU**: Optional but recommended for OCR

---

*Last updated: September 2024*
*Based on testing with Myr-Ag v1.0*
