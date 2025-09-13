# System Monitoring & Status Documentation

## Overview

The Myr-Ag RAG system provides comprehensive monitoring and status information across all domain collections. This document explains the monitoring capabilities, statistics, and management tools available in the system.

## System Status Monitoring

### Real-Time Statistics

The system provides accurate, real-time statistics across all domain collections:

- **Total Documents**: Aggregated count of all documents across all domains
- **Total Chunks**: Aggregated count of all chunks across all domains
- **Domain Distribution**: Document and chunk counts per domain
- **Collection Status**: Individual collection health and status

### System Status Endpoint

**Endpoint**: `GET /system/info`

**Response Structure**:
```json
{
  "ollama_health": boolean,
  "available_models": array,
  "default_model": string,
  "index_info": {
    "index_name": "main_collection",
    "document_count": number,
    "chunk_count": number,
    "embedding_model": string,
    "backend": string,
    "index_exists": boolean,
    "searcher_initialized": boolean
  },
  "total_documents": number,
  "total_chunks": number
}
```

## LEANN Vector Store Management

### Comprehensive Collection Monitoring

The system monitors all 7 collections (main + 6 domains) with detailed information:

**Endpoint**: `GET /system/vector-store`

**Response Structure**:
```json
{
  "main_collection": {
    "index_name": "main_collection",
    "document_count": number,
    "chunk_count": number,
    "is_initialized": boolean,
    "store_type": "LEANN",
    "embedding_model": string,
    "backend": string
  },
  "domain_collections": {
    "financial": { /* collection details */ },
    "legal": { /* collection details */ },
    "medical": { /* collection details */ },
    "academic": { /* collection details */ },
    "excel": { /* collection details */ },
    "general": { /* collection details */ }
  },
  "summary": {
    "total_collections": number,
    "total_documents": number,
    "total_chunks": number,
    "embedding_model": string,
    "backend": string
  }
}
```

### Collection Details

Each domain collection provides:

- **Index Name**: Unique identifier for the collection
- **Document Count**: Number of documents in the collection
- **Chunk Count**: Number of chunks in the collection
- **Initialization Status**: Whether the collection is properly initialized
- **Store Type**: Always "LEANN" for vector collections
- **Configuration**: Embedding model and backend information

## Domain Statistics

### Real-Time Domain Monitoring

**Endpoint**: `GET /domains/statistics`

**Response Structure**:
```json
{
  "financial": {
    "document_count": number,
    "chunk_count": number,
    "is_initialized": boolean
  },
  "legal": {
    "document_count": number,
    "chunk_count": number,
    "is_initialized": boolean
  },
  "medical": {
    "document_count": number,
    "chunk_count": number,
    "is_initialized": boolean
  },
  "academic": {
    "document_count": number,
    "chunk_count": number,
    "is_initialized": boolean
  },
  "excel": {
    "document_count": number,
    "chunk_count": number,
    "is_initialized": boolean
  },
  "general": {
    "document_count": number,
    "chunk_count": number,
    "is_initialized": boolean
  }
}
```

## UI Monitoring Interface

### System Status Section

The UI displays real-time system status with:

- **Total Documents**: Accurate count across all domains
- **Total Chunks**: Accurate count across all domains
- **System Health**: Overall system status
- **Model Information**: Available and active models

### LEANN Vector Store Management Section

The UI provides comprehensive vector store monitoring:

- **Collection Overview**: All 7 collections with individual metrics
- **Summary Statistics**: Aggregated totals and configuration
- **Real-Time Updates**: Automatic status refresh
- **Collection Details**: Individual collection status and metrics

### Domain Statistics Section

The UI shows domain-specific information:

- **Domain Distribution**: Documents and chunks per domain
- **Collection Status**: Individual domain collection health
- **Real-Time Updates**: Automatic statistics refresh
- **Domain-Specific Metrics**: Detailed information per domain

## Monitoring Best Practices

### Regular Monitoring

1. **Check System Status**: Monitor total documents and chunks
2. **Review Collection Health**: Ensure all collections are initialized
3. **Monitor Domain Distribution**: Verify documents are properly distributed
4. **Check Processing Status**: Monitor document processing progress

### Troubleshooting

1. **Statistics Mismatch**: If counts don't match, try rebuilding indexes
2. **Collection Issues**: Check individual collection initialization status
3. **Processing Problems**: Monitor processing status and logs
4. **Performance Issues**: Check collection sizes and system resources

### Maintenance

1. **Regular Rebuilds**: Use rebuild operations to refresh indexes
2. **Collection Cleanup**: Clear unused domain data when needed
3. **System Monitoring**: Monitor system resources and performance
4. **Log Analysis**: Review logs for errors and performance issues

## API Endpoints Summary

| Endpoint | Purpose | Response |
|----------|---------|----------|
| `/system/info` | System status and totals | SystemInfo |
| `/system/vector-store` | All collection details | Comprehensive collection info |
| `/domains/statistics` | Domain-specific statistics | Domain statistics |
| `/system/llamaindex-status` | Excel index status | LlamaIndex status |
| `/system/processing-status` | Processing queue status | Processing status |

## Recent Improvements

### System Status & Statistics Fixed

- **Accurate Totals**: Fixed total_documents and total_chunks calculation
- **Domain Aggregation**: Proper summation across all domain collections
- **Real-Time Updates**: Automatic statistics refresh in UI
- **Collection Monitoring**: Individual collection status and metrics

### LEANN Vector Store Management Enhanced

- **Comprehensive View**: Shows all 7 collections instead of just main collection
- **Collection Details**: Individual collection status, document count, and chunk count
- **Summary Statistics**: Aggregated totals and configuration information
- **Real-Time Monitoring**: Automatic status updates with accurate counts

### UI Improvements

- **Automatic Loading**: Domain statistics load automatically on startup
- **Real-Time Updates**: Status information updates in real-time
- **Comprehensive Monitoring**: Complete system overview in single interface
- **Better Error Handling**: Improved error messages and status indicators

## Conclusion

The Myr-Ag system now provides comprehensive monitoring and status information across all domain collections. The system accurately tracks documents and chunks, provides real-time statistics, and offers detailed collection management capabilities. This monitoring infrastructure ensures system health, performance optimization, and effective troubleshooting.
