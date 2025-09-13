"""
Gradio frontend for Myr-Ag RAG System.
"""
import gradio as gr
import requests
import json
from pathlib import Path
from typing import List, Dict, Any
import time
import logging
from src.ui.user_guides import get_english_guide, get_french_guide, get_spanish_guide, get_german_guide

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
    handlers=[
        logging.FileHandler('logs/ui.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# API configuration
API_BASE_URL = "http://localhost:8199"


class GradioFrontend:
    """Gradio frontend for Myr-Ag RAG System."""
    
    def __init__(self):
        """Initialize the Gradio frontend."""
        self.api_url = API_BASE_URL
        logger.info(f"Gradio frontend initialized with API URL: {self.api_url}")
        
    def check_api_health(self):
        """Check if the API is running."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=15)
            is_healthy = response.status_code == 200
            logger.info(f"API health check: {'Healthy' if is_healthy else 'Unhealthy'}")
            return is_healthy
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return False
    
    def get_system_info(self):
        """Get system information from API."""
        try:
            response = requests.get(f"{self.api_url}/system/info", timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            return None
    
    def get_processing_status(self):
        """Get real-time processing status from API."""
        try:
            response = requests.get(f"{self.api_url}/system/processing-status", timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "unknown", "message": "Could not retrieve status"}
        except Exception as e:
            return {"status": "error", "message": f"Error: {str(e)}"}
    
    def get_available_models(self):
        """Get list of available Ollama models from API."""
        try:
            response = requests.get(f"{self.api_url}/system/models", timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                return {"models": [], "default_model": "llama3.2:3b", "current_model": "llama3.2:3b"}
        except Exception as e:
            return {"models": [], "default_model": "llama3.2:3b", "current_model": "llama3.2:3b"}
    
    def change_model(self, model_name: str):
        """Change the current Ollama model."""
        try:
            response = requests.post(f"{self.api_url}/system/change-model", json={"model_name": model_name}, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return f"{result.get('message', 'Model changed successfully')}"
            else:
                return f"Failed to change model: {response.text}"
        except Exception as e:
            return f"Error changing model: {str(e)}"
    
    def refresh_models_list(self):
        """Refresh the list of available models from API."""
        try:
            models_info = self.get_available_models()
            if models_info.get("models"):
                # Extract model names from the models list
                model_names = [model.get("name", "") for model in models_info["models"] if model.get("name")]
                if model_names:
                    return gr.Dropdown(choices=model_names, value=models_info.get("current_model", "llama3.2:3b"))
                else:
                    return gr.Dropdown(choices=["llama3.2:3b"], value="llama3.2:3b")
            else:
                return gr.Dropdown(choices=["llama3.2:3b"], value="llama3.2:3b")
        except Exception as e:
            return gr.Dropdown(choices=["llama3.2:3b"], value="llama3.2:3b")
    
    def get_vector_store_info(self):
        """Get vector store information from API."""
        try:
            response = requests.get(f"{self.api_url}/system/vector-store", timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API returned status {response.status_code}"}
        except Exception as e:
            logger.error(f"Error getting vector store info: {e}")
            return {"error": str(e)}
    

    
    def wait_for_processing_completion(self, max_wait_time=900):
        """Wait for document processing to complete with progress updates."""
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            status = self.get_processing_status()
            if status.get("status") == "completed":
                return f"Processing completed successfully!\n{status.get('message', '')}"
            elif status.get("status") == "failed":
                return f"Processing failed: {status.get('message', 'Unknown error')}"
            elif status.get("status") == "processing":
                # Continue waiting
                time.sleep(5)
            elif status.get("status") == "idle":
                # Check if processing is actually happening by comparing document counts
                if status.get("processed_documents", 0) > 0:
                    return f"Processing completed! {status.get('processed_documents')} documents are now available."
                else:
                    time.sleep(2)
            else:
                time.sleep(2)
        
        return "⏰ Processing timeout - check system status for details"
    
    def upload_and_process_documents(self, files):
        """Upload and process documents with real-time status monitoring."""
        if not files:
            return "No files selected for upload."
        
        try:
            # Prepare files for upload
            file_list = []
            for file in files:
                if hasattr(file, 'name') and file.name:
                    # Gradio file objects have a .name attribute and the file content
                    # We need to open and read the file content
                    try:
                        with open(file.name, 'rb') as f:
                            file_content = f.read()
                        file_list.append(('files', (Path(file.name).name, file_content, 'application/octet-stream')))
                    except Exception as file_error:
                        return f"Error reading file {file.name}: {str(file_error)}"
            
            # Upload files with increased timeout for hybrid chunking
            response = requests.post(
                f"{self.api_url}/documents/upload",
                files=file_list,
                timeout=900  # 15 minutes for upload + processing
            )
            
            if response.status_code == 200:
                result = response.json()
                upload_message = f"{result['message']}\n\nUploaded files:\n" + "\n".join([f"- {Path(f).name}" for f in result['uploaded_files']])
                
                # Check if any documents were actually processed
                if result.get('processed_count', 0) == 0:
                    return f"{upload_message}\n\nℹ️ No new documents to process - all documents are already up to date!"
                
                # Wait for processing completion
                processing_message = self.wait_for_processing_completion()
                
                return f"{upload_message}\n\n{processing_message}"
            else:
                return f"Upload failed: {response.text}"
                
        except Exception as e:
            return f"Error during upload: {str(e)}"
    
    def upload_only_documents(self, files):
        """Upload documents without processing them."""
        if not files:
            return "No files selected for upload."
        
        try:
            # Prepare files for upload
            file_list = []
            for file in files:
                if hasattr(file, 'name') and file.name:
                    try:
                        with open(file.name, 'rb') as f:
                            file_content = f.read()
                        file_list.append(('files', (Path(file.name).name, file_content, 'application/octet-stream')))
                    except Exception as file_error:
                        return f"Error reading file {file.name}: {str(file_error)}"
            
            # Upload files without processing
            response = requests.post(
                f"{self.api_url}/documents/upload-only",
                files=file_list,
                timeout=300  # 5 minutes for upload only
            )
            
            if response.status_code == 200:
                result = response.json()
                return f"{result['message']}\n\nUploaded files:\n" + "\n".join([f"- {Path(f).name}" for f in result['uploaded_files']])
            else:
                return f"Upload failed: {response.text}"
                
        except Exception as e:
            return f"Error during upload: {str(e)}"
    
    def process_existing_documents(self):
        """Process documents already in uploads directory."""
        try:
            response = requests.post(f"{self.api_url}/documents/process", timeout=900)  # 15 minutes for processing
            
            if response.status_code == 200:
                result = response.json()
                return f"{result['message']}"
            else:
                return f"Processing failed: {response.text}"
                
        except Exception as e:
            return f"Error during processing: {str(e)}"
    
    def process_uploaded_only_documents(self):
        """Process only documents that are uploaded but not yet processed."""
        try:
            response = requests.post(f"{self.api_url}/documents/process-uploaded-only", timeout=900)  # 15 minutes for processing
            
            if response.status_code == 200:
                result = response.json()
                message = f"{result['message']}"
                
                # Add details about which files were processed
                if result.get('unprocessed_files'):
                    message += f"\n\nFiles processed:\n" + "\n".join([f"- {f}" for f in result['unprocessed_files']])
                
                return message
            else:
                return f"Processing failed: {response.text}"
                
        except Exception as e:
            return f"Error during processing: {str(e)}"
    
    def get_domain_statistics(self):
        """Get domain statistics from API."""
        try:
            response = requests.get(f"{self.api_url}/domains/statistics", timeout=30)
            
            if response.status_code == 200:
                stats = response.json()
                
                # Format statistics
                result = "Domain Statistics:\n\n"
                for domain, info in stats.items():
                    if "error" not in info:
                        result += f"{domain.upper()}:\n"
                        result += f"   Documents: {info.get('document_count', 0)}\n"
                        result += f"   Chunks: {info.get('chunk_count', 0)}\n"
                        result += f"   Status: {'Active' if info.get('is_initialized', False) else 'Inactive'}\n\n"
                    else:
                        result += f"{domain.upper()}: {info['error']}\n\n"
                
                return result
            else:
                return f"Failed to get domain statistics: {response.text}"
                
        except Exception as e:
            return f"Error getting domain statistics: {str(e)}"
    
    
    def reset_all_domain_indexes(self):
        """Reset all domain indexes."""
        try:
            response = requests.post(f"{self.api_url}/domains/reset-all", timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                results = result.get("results", {})
                
                message = f"{result['message']}\n\n"
                message += "Reset Results:\n"
                for domain, success in results.items():
                    status = "Success" if success else "Failed"
                    message += f"   {domain}: {status}\n"
                
                return message
            else:
                return f"Failed to reset all domains: {response.text}"
                
        except Exception as e:
            return f"Error resetting all domains: {str(e)}"
    
    def rebuild_all_domain_indexes(self):
        """Rebuild all domain indexes."""
        try:
            response = requests.post(f"{self.api_url}/domains/rebuild-all", timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                results = result.get("results", {})
                
                message = f"{result['message']}\n\n"
                message += "Rebuild Results:\n"
                for domain, success in results.items():
                    status = "Success" if success else "Failed"
                    message += f"  {domain}: {status}\n"
                
                return message
            else:
                return f"Failed to rebuild all domains: {response.text}"
                
        except Exception as e:
            return f"Error rebuilding all domains: {str(e)}"
    
    def update_index_description(self, selected_index):
        """Update description based on selected index."""
        descriptions = {
            "general": "*Removes only the GENERAL LEANN vector index (main_collection), preserves domain-specific indexes and all processed documents*",
            "financial": "*Manages the Financial domain index - processes financial documents with specialized chunking and retrieval*",
            "legal": "*Manages the Legal domain index - processes legal documents with specialized chunking and retrieval*",
            "medical": "*Manages the Medical domain index - processes medical documents with specialized chunking and retrieval*",
            "academic": "*Manages the Academic domain index - processes academic documents with specialized chunking and retrieval*",
            "all_indexes": "*Rebuilds ALL indexes: General LEANN + All specialized domains (financial, legal, medical, academic) + LlamaIndex Excel*",
            "llamaindex": "*Removes only the LlamaIndex Excel index, preserves all processed Excel files and uploads*"
        }
        return descriptions.get(selected_index, "*Select an index above to see its description*")
    
    def update_clear_description(self, selected_clear):
        """Update description based on selected clear operation."""
        descriptions = {
            "general": "*Removes General LEANN index + all processed documents (preserves uploads and domain-specific indexes)*",
            "financial": "*Removes Financial domain index + financial processed documents only (preserves other domains and uploads)*",
            "legal": "*Removes Legal domain index + legal processed documents only (preserves other domains and uploads)*",
            "medical": "*Removes Medical domain index + medical processed documents only (preserves other domains and uploads)*",
            "academic": "*Removes Academic domain index + academic processed documents only (preserves other domains and uploads)*",
            "excel": "*Removes LlamaIndex Excel index + processed Excel files only (preserves uploads and other documents)*",
            "all": "*Removes EVERYTHING: All indexes + all processed documents + all uploads (⚠️ DANGER: This will delete all your data!)*"
        }
        return descriptions.get(selected_clear, "*Select domain to clear above to see its description*")
    
    def reset_selected_index(self, selected_index, confirm):
        """Reset the selected index."""
        if not confirm:
            return "Please confirm the reset operation by checking the confirmation box."
        
        try:
            if selected_index == "general":
                response = requests.post(f"{self.api_url}/system/reset-index", timeout=120)
            elif selected_index in ["financial", "legal", "medical", "academic"]:
                response = requests.post(f"{self.api_url}/domains/{selected_index}/reset", timeout=120)
            elif selected_index == "all_indexes":
                # Reset all indexes: general + domains + excel
                general_response = requests.post(f"{self.api_url}/system/reset-index", timeout=120)
                domains_response = requests.post(f"{self.api_url}/domains/reset-all", timeout=120)
                excel_response = requests.post(f"{self.api_url}/system/reset-llamaindex", timeout=120)
                
                # Combine results
                if general_response.status_code == 200 and domains_response.status_code == 200 and excel_response.status_code == 200:
                    return "All indexes reset successfully: General + Domains + Excel"
                else:
                    return f"Partial reset completed. General: {general_response.status_code}, Domains: {domains_response.status_code}, Excel: {excel_response.status_code}"
            elif selected_index == "llamaindex":
                response = requests.post(f"{self.api_url}/system/reset-llamaindex", timeout=120)
            else:
                return "Invalid index selection."
            
            if response.status_code == 200:
                result = response.json()
                return f"{result['message']}"
            else:
                return f"Failed to reset {selected_index}: {response.text}"
                
        except Exception as e:
            return f"Error resetting {selected_index}: {str(e)}"
    
    def rebuild_selected_index(self, selected_index):
        """Rebuild the selected index."""
        try:
            if selected_index == "general":
                response = requests.post(f"{self.api_url}/system/rebuild-index", timeout=300)
            elif selected_index in ["financial", "legal", "medical", "academic"]:
                response = requests.post(f"{self.api_url}/domains/{selected_index}/rebuild", timeout=300)
            elif selected_index == "all_indexes":
                response = requests.post(f"{self.api_url}/system/rebuild-all", timeout=300)
            elif selected_index == "llamaindex":
                response = requests.post(f"{self.api_url}/system/rebuild-llamaindex", timeout=300)
            else:
                return "Invalid index selection."
            
            if response.status_code == 200:
                result = response.json()
                if selected_index == "all_indexes":
                    results = result.get("results", {})
                    message = f"{result['message']}\n\n"
                    message += "Rebuild Results:\n"
                    for index_name, success in results.items():
                        status = "Success" if success else "Failed"
                        message += f"  {index_name}: {status}\n"
                    return message
                else:
                    return f"{result['message']}"
            else:
                return f"Failed to rebuild {selected_index}: {response.text}"
                
        except Exception as e:
            return f"Error rebuilding {selected_index}: {str(e)}"
    
    
    def get_document_domain(self, file_path):
        """Get the current domain of a document."""
        try:
            if not file_path:
                return "No document selected"
            
            # The dropdown now contains just the file name
            file_name = file_path
            
            response = requests.get(f"{self.api_url}/documents/{file_name}/domain", timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                domain = result.get("current_domain", "general")
                return f"Current domain: {domain}"
            else:
                return f"Error: {response.text}"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    
    def upload_and_process_with_domain(self, files, domain):
        """Upload and process documents with specified domain."""
        try:
            if not files:
                return "Please select files first.", " No files selected"
            
            # Prepare files for upload - handle Gradio file objects
            file_data = []
            for file in files:
                if hasattr(file, 'name') and file.name:
                    # For Gradio file objects, read the file content
                    try:
                        with open(file.name, 'rb') as f:
                            file_content = f.read()
                        file_data.append(('files', (file.name.split('/')[-1], file_content, 'application/octet-stream')))
                    except Exception as e:
                        return f" Error reading file {file.name}: {str(e)}", f" File read error"
            
            # Upload directly with domain specification
            response = requests.post(
                f"{self.api_url}/documents/upload-with-domain",
                files=file_data,
                data={'domain': domain},
                timeout=900
            )
            
            if response.status_code == 200:
                result = response.json()
                message = result.get("message", "Upload successful")
                processed_count = result.get("processed_count", 0)
                uploaded_files = result.get("uploaded_files", [])
                
                # Format success message
                success_msg = f"{message}\n"
                success_msg += f" Domain: {domain.upper()}\n"
                success_msg += f"Processed: {processed_count}/{len(uploaded_files)} documents\n"
                
                if uploaded_files:
                    success_msg += f"Files: {', '.join([Path(f).name for f in uploaded_files])}\n"
                
                return success_msg, f"Successfully uploaded and processed in {domain} domain"
            else:
                error_msg = f"Upload failed: {response.text}"
                return error_msg, f"Upload error: {response.status_code}"
                
        except Exception as e:
            error_msg = f"Error uploading documents: {str(e)}"
            return error_msg, f"Upload error: {str(e)}"
    
    def query_documents(self, question, n_chunks, temperature, max_tokens, model_name=None, use_enhanced=False, use_specialized=False, pipeline_domain=None):
        """Query documents using RAG pipeline."""
        if not question.strip():
            return "Please enter a question.", "Please enter a question."
        
        try:
            # Step 1: Prepare query request
            query_data = {
                "question": question,
                "n_chunks": int(n_chunks),
                "temperature": float(temperature),
                "max_tokens": int(max_tokens)
            }
            
            # Add model parameter if specified
            if model_name:
                query_data["model"] = model_name
            
            # Step 2: Choose endpoint based on flags
            if use_specialized:
                if pipeline_domain and pipeline_domain != "auto":
                    endpoint = f"/query-specialized/{pipeline_domain}"
                else:
                    endpoint = "/query-specialized"
            elif use_enhanced:
                endpoint = "/query-enhanced"
            else:
                endpoint = "/query"
            
            # Step 3: Send query to API
            response = requests.post(
                f"{self.api_url}{endpoint}",
                json=query_data,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Format response
                answer = f"**Question:** {result['question']}\n\n"
                answer += f"**Answer:** {result['answer']}\n\n"
                answer += f"**Confidence:** {result['confidence_score']:.2f}\n"
                answer += f"**Processing Time:** {result['processing_time']:.2f}s\n"
                
                # Add processing method info
                if use_specialized:
                    pipeline_used = result.get('pipeline_used', 'unknown')
                    domain = result.get('domain', 'unknown')
                    answer += f"**Processing Method:** Specialized Pipeline ({pipeline_used})\n"
                    answer += f"**Domain:** {domain}\n"
                    
                    # Add query enhancement info
                    enhanced_query = result.get('enhanced_query', {})
                    if enhanced_query and enhanced_query.get('enhanced'):
                        answer += f"**Query Enhancement:**\n"
                        answer += f"- Original: {enhanced_query.get('original', 'N/A')}\n"
                        answer += f"- Enhanced: {enhanced_query.get('enhanced', 'N/A')}\n"
                        answer += f"- Reason: {enhanced_query.get('reason', 'N/A')}\n"
                elif use_enhanced:
                    answer += f"**Processing Method:** LlamaIndex + LEANN (Enhanced)\n"
                else:
                    answer += f"**Processing Method:** LEANN (Standard)\n"
                answer += "\n"
                
                # Add sources
                if result['sources']:
                    answer += "**Sources:**\n"
                    for i, source in enumerate(result['sources'][:3], 1):
                        answer += f"{i}. {source['file_name']} (relevance: {source['relevance_score']:.2f})\n"
                        answer += f"   Preview: {source['text_preview']}\n\n"
                
                # Add Excel sources if available (enhanced mode)
                if use_enhanced and 'excel_sources' in result and result['excel_sources']:
                    answer += "**Excel Sources:**\n"
                    for i, source in enumerate(result['excel_sources'][:5], 1):
                        answer += f"{i}. {source['file_name']} - {source['sheet_name']} ({source['chunk_type']})\n"
                        answer += f"   Preview: {source['text_preview']}\n\n"
                
                # Add direct answer if available (enhanced mode)
                if use_enhanced and 'direct_answer' in result and result['direct_answer']:
                    answer += f"**Direct Answer:** {result['direct_answer']}\n\n"
                
                return answer, "Query completed successfully!"
            else:
                return f"Query failed: {response.text}", "Query failed"
                
        except Exception as e:
            return f"Error during query: {str(e)}", "Query error"
    
    def query_documents_with_status(self, question, n_chunks, temperature, max_tokens, model_name=None, use_enhanced=False):
        """Query documents with status updates."""
        if not question.strip():
            return "Please enter a question.", "Please enter a question."
        
        # Show initial status
        return self.query_documents(question, n_chunks, temperature, max_tokens, model_name, use_enhanced)
    
    def list_documents(self, force_refresh=False):
        """List all documents (both uploaded and processed) with optional force refresh."""
        try:
            # Add refresh parameter as boolean if force refresh is requested
            params = {}
            if force_refresh:
                params['refresh'] = 'true'
            
            response = requests.get(f"{self.api_url}/documents", params=params, timeout=60)
            
            if response.status_code == 200:
                documents = response.json()
                
                if not documents:
                    return "No documents found."
                
                # Separate processed and uploaded documents
                processed_docs = [doc for doc in documents if doc['chunk_count'] > 0]
                uploaded_docs = [doc for doc in documents if doc['chunk_count'] == 0]
                
                
                # Helper function to get domain for a document
                def get_document_domain(file_name):
                    try:
                        domain_response = requests.get(f"{self.api_url}/documents/{file_name}/domain", timeout=5)
                        if domain_response.status_code == 200:
                            return domain_response.json().get("current_domain", "general")
                    except:
                        pass
                    return "general"
                
                # Format document list
                doc_list = ""
                
                if processed_docs:
                    doc_list += "** Processed Documents (Ready for Query):**\n\n"
                    for i, doc in enumerate(processed_docs, 1):
                        domain = get_document_domain(doc['file_name'])
                        doc_list += f"{i}. **{doc['file_name']}** ({domain.upper()})\n"
                        doc_list += f"   - Size: {doc['file_size']} bytes\n"
                        doc_list += f"   - Content: {doc['content_length']} characters\n"
                        doc_list += f"   - Chunks: {doc['chunk_count']}\n"
                        doc_list += f"   - Processed: {doc['processing_timestamp']}\n\n"
                
                if uploaded_docs:
                    if processed_docs:
                        doc_list += "---\n\n"
                    doc_list += "** Uploaded Documents (Not Processed Yet):**\n\n"
                    for i, doc in enumerate(uploaded_docs, 1):
                        domain = get_document_domain(doc['file_name'])
                        doc_list += f"{i}. **{doc['file_name']}** ({domain.upper()})\n"
                        doc_list += f"   - Size: {doc['file_size']} bytes\n"
                        doc_list += f"   - Status: ⏳ Ready to process\n\n"
                
                return doc_list
            else:
                return f" Failed to retrieve documents: {response.text}"
                
        except Exception as e:
            return f" Error retrieving documents: {str(e)}"
    
    def refresh_documents_with_dropdown(self):
        """Refresh documents and update both list and dropdown."""
        try:
            # Get document list
            doc_list = self.list_documents(force_refresh=True)
            
            # Get documents for dropdown with status and domain
            response = requests.get(f"{self.api_url}/documents", params={'refresh': 'true'}, timeout=60)
            dropdown_choices = []
            
            if response.status_code == 200:
                documents = response.json()
                for doc in documents:
                    
                    dropdown_choices.append(doc['file_name'])
            
            return doc_list, gr.Dropdown(choices=dropdown_choices, value=None)
            
        except Exception as e:
            error_msg = f" Error refreshing documents: {str(e)}"
            return error_msg, gr.Dropdown(choices=[], value=None)
    
    def delete_document(self, file_name_with_status):
        """Delete a specific document."""
        if not file_name_with_status:
            return "Please select a document to delete.", gr.Dropdown(choices=[], value=None), gr.Dropdown(choices=[], value=None)
        
        try:
            # The dropdown now contains just the file name
            file_name = file_name_with_status
            
            # URL encode the file name
            import urllib.parse
            encoded_file_name = urllib.parse.quote(file_name, safe='')
            
            response = requests.delete(f"{self.api_url}/documents/{encoded_file_name}", timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                success_msg = f"{result['message']}\n\nDeleted files:\n" + "\n".join(f"- {file}" for file in result.get('deleted_files', []))
                
                # Use the same refresh function as other operations to ensure consistency
                doc_list, doc_dropdown = self.refresh_documents_with_dropdown()
                
                return doc_list, doc_dropdown, success_msg
            else:
                error_msg = f"Failed to delete document: {response.text}"
                doc_list, doc_dropdown = self.refresh_documents_with_dropdown()
                return doc_list, doc_dropdown, error_msg
                
        except Exception as e:
            error_msg = f"Error deleting document: {str(e)}"
            doc_list, doc_dropdown = self.refresh_documents_with_dropdown()
            return doc_list, doc_dropdown, error_msg
    
    def delete_document_with_dropdown(self, file_name_with_status):
        """Delete a specific document and refresh dropdowns."""
        if not file_name_with_status:
            return "Please select a document to delete.", gr.Dropdown(choices=[], value=None)
        
        try:
            # The dropdown now contains just the file name
            file_name = file_name_with_status
            
            # URL encode the file name
            import urllib.parse
            encoded_file_name = urllib.parse.quote(file_name, safe='')
            
            response = requests.delete(f"{self.api_url}/documents/{encoded_file_name}", timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                success_msg = f"{result['message']}\n\nDeleted files:\n" + "\n".join(f"- {file}" for file in result.get('deleted_files', []))
                
                # Use the same refresh function as other operations to ensure consistency
                doc_list, doc_dropdown = self.refresh_documents_with_dropdown()
                
                return doc_list, doc_dropdown
            else:
                error_msg = f"Failed to delete document: {response.text}"
                doc_list, doc_dropdown = self.refresh_documents_with_dropdown()
                return doc_list, doc_dropdown
                
        except Exception as e:
            error_msg = f"Error deleting document: {str(e)}"
            doc_list, doc_dropdown = self.refresh_documents_with_dropdown()
            return doc_list, doc_dropdown
    
    def upload_and_process_documents_with_dropdown(self, files):
        """Upload and process documents, then refresh dropdown."""
        upload_result = self.upload_and_process_documents(files)
        doc_list = self.list_documents(force_refresh=True)
        
        # Get updated dropdown choices with status
        try:
            response = requests.get(f"{self.api_url}/documents", params={'refresh': 'true'}, timeout=60)
            dropdown_choices = []
            if response.status_code == 200:
                documents = response.json()
                for doc in documents:
                    if doc['chunk_count'] > 0:
                        status = " Processed"
                    else:
                        status = " Uploaded Only"
                    dropdown_choices.append(f"{doc['file_name']} ({status})")
        except Exception:
            dropdown_choices = []
        
        return upload_result, doc_list, gr.Dropdown(choices=dropdown_choices, value=None)
    
    def upload_only_documents_with_dropdown(self, files):
        """Upload only documents, then refresh dropdown."""
        upload_result = self.upload_only_documents(files)
        doc_list = self.list_documents(force_refresh=True)
        
        # Get updated dropdown choices with status and domain
        try:
            response = requests.get(f"{self.api_url}/documents", params={'refresh': 'true'}, timeout=60)
            dropdown_choices = []
            if response.status_code == 200:
                documents = response.json()
                for doc in documents:
                    
                    dropdown_choices.append(doc['file_name'])
        except Exception:
            dropdown_choices = []
        
        return upload_result, doc_list, gr.Dropdown(choices=dropdown_choices, value=None), gr.Dropdown(choices=dropdown_choices, value=None)
    
    def upload_only_documents_with_domain(self, files, domain):
        """Upload only documents with domain assignment, then refresh dropdown."""
        if not files:
            return "No files selected for upload.", "", gr.Dropdown(choices=[], value=None)
        
        try:
            # Prepare files for upload
            file_list = []
            for file in files:
                if hasattr(file, 'name') and file.name:
                    try:
                        with open(file.name, 'rb') as f:
                            file_content = f.read()
                        file_list.append(('files', (Path(file.name).name, file_content, 'application/octet-stream')))
                    except Exception as file_error:
                        return f"Error reading file {file.name}: {str(file_error)}", "", gr.Dropdown(choices=[], value=None)
            
            # Upload files with domain metadata and process them
            response = requests.post(
                f"{self.api_url}/documents/upload-with-domain",
                files=file_list,
                data={'domain': domain},
                timeout=300  # 5 minutes for upload and processing
            )
            
            if response.status_code == 200:
                result = response.json()
                upload_result = f" {result['message']}\n\nUploaded files:\n" + "\n".join([f"- {Path(f).name}" for f in result['uploaded_files']])
            else:
                upload_result = f" Upload failed: {response.text}"
                return upload_result, "", gr.Dropdown(choices=[], value=None)
            
            # Refresh document list and dropdowns
            doc_list = self.list_documents(force_refresh=True)
            
            # Get updated dropdown choices with status and domain
            try:
                response = requests.get(f"{self.api_url}/documents", params={'refresh': 'true'}, timeout=60)
                dropdown_choices = []
                if response.status_code == 200:
                    documents = response.json()
                    for doc in documents:
                        # Get domain information
                        domain = "general"  # Default
                        try:
                            domain_response = requests.get(f"{self.api_url}/documents/{doc['file_name']}/domain", timeout=5)
                            if domain_response.status_code == 200:
                                domain = domain_response.json().get("current_domain", "general")
                        except:
                            pass  # Use default if domain lookup fails
                        
                        
                        
                        if doc['chunk_count'] > 0:
                            status = " Processed"
                        else:
                            status = " Uploaded Only"
                        
                        dropdown_choices.append(doc['file_name'])
            except Exception:
                dropdown_choices = []
            
            return upload_result, doc_list, gr.Dropdown(choices=dropdown_choices, value=None)
            
        except Exception as e:
            error_msg = f" Error during upload: {str(e)}"
            return error_msg, "", gr.Dropdown(choices=[], value=None)
    
    def reset_index(self):
        """Reset the vector index."""
        try:
            response = requests.delete(f"{self.api_url}/system/reset-index", timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return f"{result['message']}"
            else:
                return f"Reset failed: {response.text}"
                
        except Exception as e:
            return f" Error during reset: {str(e)}"
    
    def clear_documents(self):
        """Clear all documents and processed data."""
        try:
            response = requests.delete(f"{self.api_url}/system/clear-documents", timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return f"{result['message']}"
            else:
                return f"Clear failed: {response.text}"
                
        except Exception as e:
            return f" Error during clear: {str(e)}"
    
    def clear_all(self):
        """Clear everything including uploads."""
        try:
            response = requests.delete(f"{self.api_url}/system/clear-all", timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return f"{result['message']}"
            else:
                return f"Clear all failed: {response.text}"
                
        except Exception as e:
            return f"Error during clear all: {str(e)}"
    
    # Enhancement 3: Methods with confirmation
    def reset_index_with_confirmation(self, confirm: bool):
        """Reset index with confirmation."""
        if not confirm:
            return " Operation cancelled - confirmation required"
        return self.reset_index()
    
    def clear_documents_with_confirmation(self, confirm: bool):
        """Clear documents with confirmation."""
        if not confirm:
            return " Operation cancelled - confirmation required"
        return self.clear_documents()
    
    def clear_all_with_confirmation(self, confirm: bool):
        """Clear everything with confirmation."""
        if not confirm:
            return " Operation cancelled - confirmation required"
        return self.clear_all()
    
    def clear_selected_data(self, selected_clear, confirm):
        """Clear the selected data."""
        if not confirm:
            return "Please confirm the clear operation by checking the confirmation box."
        
        try:
            if selected_clear == "general":
                response = requests.delete(f"{self.api_url}/system/clear-general", timeout=120)
            elif selected_clear in ["financial", "legal", "medical", "academic"]:
                response = requests.delete(f"{self.api_url}/system/clear-domain/{selected_clear}", timeout=120)
            elif selected_clear == "excel":
                response = requests.delete(f"{self.api_url}/system/clear-excel", timeout=120)
            elif selected_clear == "all":
                response = requests.delete(f"{self.api_url}/system/clear-all", timeout=120)
            else:
                return f" Unknown clear operation: {selected_clear}"
            
            if response.status_code == 200:
                result = response.json()
                return f"{result['message']}"
            else:
                return f"Clear failed: {response.text}"
                
        except Exception as e:
            return f" Error during clear operation: {str(e)}"
    
    
    def reset_llamaindex(self):
        """Reset LlamaIndex Excel index only (preserves processed files)."""
        try:
            response = requests.delete(f"{self.api_url}/system/reset-llamaindex", timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return f"{result['message']}"
            else:
                return f"Error resetting LlamaIndex: {response.text}"
                
        except Exception as e:
            return f"Error resetting LlamaIndex: {str(e)}"
    
    def clear_llamaindex(self):
        """Clear LlamaIndex Excel index and related data."""
        try:
            response = requests.delete(f"{self.api_url}/system/clear-llamaindex", timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                success_msg = f" {result['message']}\n\n"
                
                if result.get('deleted_processed_files'):
                    success_msg += f"Deleted processed files:\n" + "\n".join(f"- {file}" for file in result['deleted_processed_files']) + "\n\n"
                
                if result.get('deleted_upload_files'):
                    success_msg += f"Deleted upload files:\n" + "\n".join(f"- {file}" for file in result['deleted_upload_files'])
                
                return success_msg
            else:
                return f" Error clearing LlamaIndex: {response.text}"
                
        except Exception as e:
            return f" Error clearing LlamaIndex: {str(e)}"
    
    def rebuild_leann(self):
        """Rebuild LEANN index from existing processed documents."""
        try:
            response = requests.post(f"{self.api_url}/system/rebuild-leann", timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                return f"{result['message']}"
            else:
                return f"Error rebuilding LEANN: {response.text}"
                
        except Exception as e:
            return f"Error rebuilding LEANN: {str(e)}"
    
    def reset_llamaindex_with_confirmation(self, confirm: bool):
        """Reset LlamaIndex with confirmation."""
        if not confirm:
            return " Operation cancelled - confirmation required"
        return self.reset_llamaindex()
    
    def reset_domain_indexes_with_confirmation(self, confirm: bool):
        """Reset all domain indexes with confirmation."""
        if not confirm:
            return " Operation cancelled - confirmation required"
        return self.reset_domain_indexes()
    
    def reset_domain_indexes(self):
        """Reset all domain-specific indexes."""
        try:
            response = requests.post(f"{self.api_url}/domains/reset-all", timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                message = f"{result['message']}\n\n"
                
                if 'results' in result:
                    message += "Domain reset results:\n"
                    for domain, success in result['results'].items():
                        status = "" if success else ""
                        message += f"  {status} {domain}: {'Success' if success else 'Failed'}\n"
                
                return message
            else:
                return f" Error resetting domain indexes: {response.text}"
                
        except Exception as e:
            return f" Error resetting domain indexes: {str(e)}"
    
    def rebuild_llamaindex(self):
        """Rebuild LlamaIndex Excel index from existing processed Excel files."""
        try:
            response = requests.post(f"{self.api_url}/system/rebuild-llamaindex", timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                return f"{result['message']}"
            else:
                return f"Error rebuilding LlamaIndex: {response.text}"
                
        except Exception as e:
            return f"Error rebuilding LlamaIndex: {str(e)}"
    
    def clear_llamaindex_with_confirmation(self, confirm: bool):
        """Clear LlamaIndex with confirmation."""
        if not confirm:
            return " Operation cancelled - confirmation required"
        return self.clear_llamaindex()
    
    def get_llamaindex_status(self):
        """Get LlamaIndex status and statistics."""
        try:
            response = requests.get(f"{self.api_url}/system/llamaindex-status", timeout=60)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Failed to get LlamaIndex status: {response.text}"}
                
        except Exception as e:
            return {"error": f"Error getting LlamaIndex status: {str(e)}"}
    
    def create_interface(self):
        """Create the Gradio interface."""
        with gr.Blocks(
            title="Myr-Ag", 
            theme=gr.themes.Default(
                font=gr.themes.GoogleFont("Roboto")
            )
        ) as interface:
            
            # Header
            gr.Markdown("# Myr-Ag System")
            gr.Markdown("<div style='height: 40px;'></div>")
            
            # Main content with tabs
            with gr.Tabs() as tabs:
                
                # Tab 1: RAG Query Interface
                with gr.Tab("✒️ RAG Query", id=0):
                    gr.Markdown("<div style='height: 20px;'></div>")
                    question_input = gr.Textbox(
                        label="Ask Questions About Your Documents",
                        placeholder="e.g., What is the main topic discussed in the documents?",
                        lines=3
                    )
                    
                    
                    query_btn = gr.Button(
                        "🔍 Query Documents", 
                        variant="primary", 
                        size="lg",
                        elem_id="query_btn"
                    )
                    
                    # Query Mode Selection - Enhanced with Specialized Pipelines
                    gr.Markdown("### Query Settings")


                    query_mode = gr.Radio(
                        choices=[
                            ("Standard", "standard"),
                            ("Excel Specific (Exp.)", "excel"),
                            ("Academic", "academic"),
                            ("Financial", "financial"),
                            ("Legal", "legal"),
                            ("Medical", "medical")
                        ],
                        value="standard",
                        label="Query Method",
                        info="Choose the appropriate method for your document type and domain"
                    )
                    
                                        
                    # Model Selection and Search Parameters - Side by Side
                    with gr.Row():
                        # Model Selection Accordion
                        with gr.Column():
                            with gr.Accordion("LLM Model Selection", open=False):
                                model_selector = gr.Dropdown(
                                    choices=["llama3.2:3b", "llama3.2:7b"],
                                    value="llama3.2:3b",
                                    label="Select the model for your queries",
                                    allow_custom_value=True
                                )
                                
                                with gr.Row():
                                    change_model_btn = gr.Button("🔄 Change Model", variant="secondary", size="lg")
                                    refresh_models_btn = gr.Button("🔄 Refresh Models", variant="secondary", size="lg")
                                
                                model_change_output = gr.Textbox(label="Model Change Status", lines=2, visible=False)
                        
                        # Search Parameters Accordion
                        with gr.Column():
                            with gr.Accordion("Search Parameters", open=False):
                                with gr.Row():
                                    n_chunks_input = gr.Slider(
                                        minimum=1, maximum=40, value=20, step=2,
                                        label="Chunks"
                                    )
                                    temperature_input = gr.Slider(
                                        minimum=0.1, maximum=1.0, value=0.3, step=0.1,
                                        label="Temperature"
                                    )
                                    max_tokens_input = gr.Slider(
                                        minimum=100, maximum=4000, value=2048, step=100,
                                        label="Max tokens"
                                    )
                    
                    # Query status and progress
                    query_status = gr.Textbox(
                        label="Query Status", 
                        value="Ready to query...", 
                        interactive=False,
                        visible=True
                    )
                    
                    query_output = gr.Markdown(label="Query Result")
                
                # Tab 2: Document Management
                with gr.Tab(" Document Management", id=1):
                    gr.Markdown("                                    ")
                    
                    # Document Upload Section
                    file_input = gr.File(
                        label="Select Documents to upload",
                        file_count="multiple",
                        file_types=[".pdf", ".docx", ".txt", ".md", ".html", ".xlsx", ".pptx", ".csv", ".png", ".jpeg", ".jpg", ".tiff", ".bmp", ".webp", ".adoc", ".xml"]
                    )
                    
                    # Domain Selection for Upload   

                    with gr.Row():
                        upload_domain_selector = gr.Dropdown(
                            choices=[
                                ("Financial", "financial"),
                                ("Legal", "legal"), 
                                ("Medical", "medical"),
                                ("Academic", "academic"),
                                ("Excel", "excel"),
                                ("General", "general")
                            ],
                            value="general",
                            label="Target Domain",
                            info="Select the domain that best fits your documents"
                        )
                    
                    # Upload buttons in single column
                    upload_btn = gr.Button("Upload & Process", variant="primary", size="lg")
                    upload_only_btn = gr.Button("Upload Only", variant="secondary", size="lg")
                    process_existing_btn = gr.Button("Process All Existing", variant="secondary", size="lg")
                    process_uploaded_btn = gr.Button("Process Uploaded Only", variant="secondary", size="lg")
                    
                    upload_output = gr.Textbox(label="Upload Result", lines=3)
                    
                    # Visual separator
                    gr.Markdown("---")
                    
                    # Document List Section
                    gr.Markdown("#### Current Documents")
                    
                    # Document list display
                    doc_list_output = gr.Textbox(label="Document List", lines=8, interactive=False)
                    
                    # Document management controls
                    with gr.Row():
                        refresh_docs_btn = gr.Button("Refresh Documents", variant="secondary", size="lg")
                    
                    # Visual separator
                    gr.Markdown("---")
                    
                    
                    # Visual separator
                    gr.Markdown("---")
                    
                    # Document deletion section
                    gr.Markdown("#### Delete Documents")
                    gr.Markdown("⚠️ **Warning**: This will permanently delete the selected document from both uploads and processed data, and remove it from the search index.")
                    
                    doc_dropdown = gr.Dropdown(
                        label="Select Document to Delete",
                        choices=[],
                        value=None,
                        interactive=True,
                        info="Choose a document to delete (both uploaded file and processed data will be removed)"
                    )
                    delete_doc_btn = gr.Button("🗑️ Delete Document", variant="stop", size="lg")
                
                    # Visual separator
                    gr.Markdown("---")
                    
                    # Domain Management Section
                    gr.Markdown("#### 🎯 Domain-Specific Management")
                    gr.Markdown("Manage documents by domain (Financial, Legal, Medical, Academic) for better organization and search performance.")
                    
                    # Domain statistics
                    domain_stats_output = gr.Textbox(label="Domain Statistics", lines=6, interactive=False, value=self.get_domain_statistics())
                    
                    with gr.Row():
                        refresh_domain_stats_btn = gr.Button("🔄 Refresh Domain Stats", variant="secondary", size="lg")
                        reset_all_domains_btn = gr.Button("🔄 Reset All Domains", variant="stop", size="lg")
                    
                
                # Tab 3: System Management
                with gr.Tab("⚙️ System Management", id=2):
                    gr.Markdown("                                    ")
                    
                    # System Maintenance Section
                    gr.Markdown("#### ⚠️ System Maintenance / Warning: These actions cannot be undone!")
                    gr.Markdown("---")
                    
                    # Index Management with Radio Selection
                    gr.Markdown("#### Index Management")
                    
                    index_selector = gr.Radio(
                         choices=[
                             ("General Index", "general"),
                             ("Academic Domain", "academic"),
                             ("Financial Domain", "financial"),
                             ("Legal Domain", "legal"),
                             ("Medical Domain", "medical"),
                             ("Excel", "llamaindex"),
                             ("All Indexes", "all_indexes")
                         ],
                         value="general",
                         label="Select Index to Manage",
                     )

                    with gr.Row():
                        reset_index_btn = gr.Button("Reset Selected Index", variant="secondary", size="lg")
                        rebuild_index_btn = gr.Button("Rebuild Selected Index", variant="primary", size="lg")
                    
                    reset_confirm = gr.Checkbox(label="Confirm Reset Operation", value=False)
                    
                    index_description = gr.Markdown("*Select an index above to see its description*")
                    gr.Markdown("---")

                    gr.Markdown("<div style='height: 20px;'></div>")
                    gr.Markdown("#### Clearing Operations")
                    
                    clear_selector = gr.Radio(
                        choices=[
                            ("General Index", "general"),
                            ("Financial Domain", "financial"),
                            ("Legal Domain", "legal"),
                            ("Medical Domain", "medical"),
                            ("Academic Domain", "academic"),
                            ("Excel Index", "excel"),
                            ("All Data", "all")
                        ],
                        value="general",
                        label="Select Domain to Clear",
                    )
                    
                    with gr.Row():
                        clear_selected_btn = gr.Button("Clear Selected Domain", variant="secondary", size="lg")
                        clear_confirm = gr.Checkbox(label="Confirm Clear Operation", value=False)
                    
                    clear_description = gr.Markdown("*Select domain to clear above to see its description*")
                    gr.Markdown("---")
                    
                    management_output = gr.Textbox(label="Operation Result", lines=3)
                    
                    # Visual separator
                    gr.Markdown("---")
                    
                    # System Status Section
                    gr.Markdown("#### System Status")
                    status_output = gr.JSON(label="Current Status")
                    refresh_status_btn = gr.Button("Refresh Status", variant="secondary", size="lg")
                    
                    # Visual separator
                    gr.Markdown("---")
                    
                    # Vector Store Management Section
                    gr.Markdown("#### LEANN Vector Store Management")
                    gr.Markdown("Ultra-efficient vector storage with 97% space savings and fast retrieval.")
                    
                    vector_store_info = gr.JSON(label="LEANN Vector Store Info")
                    refresh_vector_store_btn = gr.Button("Refresh Vector Store Info", variant="secondary", size="lg")
                    
                    # Visual separator
                    gr.Markdown("---")
                    
                    # LlamaIndex Management Section
                    gr.Markdown("#### LlamaIndex Excel Management")
                    gr.Markdown("Advanced Excel processing with LlamaIndex for enhanced spreadsheet analysis.")
                    
                    llamaindex_info = gr.JSON(label="LlamaIndex Excel Info")
                    refresh_llamaindex_btn = gr.Button("Refresh LlamaIndex Info", variant="secondary", size="lg")
                
                # Tab 4: User Guide
                with gr.Tab("📖 User Guide", id=3):
                    gr.Markdown("### Myr-Ag RAG System - Complete User Guide")
                    
                    # Language Selection
                    with gr.Accordion("🌍 Language / Langue / Idioma / Sprache", open=False):
                        language_selector = gr.Radio(
                            choices=["English", "Français", "Español", "Deutsch"],
                            value="English",
                            label="Select Language / Choisir la langue / Seleccionar idioma / Sprache wählen"
                        )
                    
                    # English Guide
                    with gr.Group(visible=True) as english_guide:
                        with gr.Accordion("🎯 System Overview", open=False):
                            gr.Markdown("""
                            **Myr-Ag** is a powerful RAG (Retrieval-Augmented Generation) system that allows you to:
                            
                            * **Upload and process documents** in multiple formats (PDF, DOCX, XLSX, PPTX, TXT, MD, HTML, XHTML, CSV, PNG, JPEG, TIFF, BMP, WEBP, AsciiDoc, XML)
                            * **Ask questions** about your documents using natural language
                            * **Get intelligent answers** powered by local LLM models
                            * **Search across multiple documents** with semantic understanding
                            
                            **Dual Processing Architecture:**
                            * **Standard Method**: Uses LEANN vector store for general documents (PDFs, Word docs, text files)
                            * **Excel Specific Method**: Uses LlamaIndex for Excel files with advanced spreadsheet processing (⚠️ EXPERIMENTAL)
                            
                            The system uses advanced document processing, vector embeddings, and local LLM inference to provide accurate, contextual answers optimized for different document types.
                            """)
                        
                        with gr.Accordion(" Document Management", open=False):
                            gr.Markdown("""
                            ### Uploading Documents
                            
                            **Supported Formats:**
                            
                            **Office Documents:**
                            * **PDF**: Standard PDFs, scanned PDFs with OCR, and web-printed PDFs
                            * **DOCX**: Microsoft Word documents
                            * **XLSX**: Microsoft Excel spreadsheets with enhanced processing
                              - Row-based chunking for granular search
                              - Automatic column detection (amounts, names, dates)
                              - Multi-sheet support
                              - Natural language descriptions
                            * **PPTX**: PowerPoint presentations
                            
                            **Text & Web:**
                            * **TXT**: Plain text files
                            * **MD**: Markdown files
                            * **HTML/XHTML**: Web pages and structured documents
                            * **CSV**: Comma-separated value files
                            
                            **Images (with OCR):**
                            * **PNG, JPEG, TIFF, BMP, WEBP**: Scanned documents and images with automatic text extraction
                            
                            **Specialized Formats:**
                            * **AsciiDoc**: Technical documentation
                            * **XML**: Structured data and documents
                            
                            **Upload Options:**
                            * **Upload & Process**: Upload and immediately process documents for querying
                            * **Upload Only**: Upload documents without processing (useful for batch operations)
                            * **Process Existing**: Process documents already in the uploads directory
                            * **Process Uploaded Only**: Process only documents that haven't been processed yet
                            """)
                        
                        with gr.Accordion("🔍 Query Methods", open=False):
                            gr.Markdown("""
                            ### Understanding Query Methods
                            
                            The system offers two distinct query methods optimized for different document types:
                            
                            #### **Standard Method (LEANN only)**
                            
                            **Best for:** General documents (PDFs, Word docs, text files, images)
                            
                            **Features:**
                            * Uses LEANN vector store for all document types
                            * Sentence-based chunking for text documents
                            * OCR processing for images and scanned documents
                            * Consistent processing across all supported formats
                            
                            **When to use:**
                            * General document search and analysis
                            * Mixed document collections
                            * When you want consistent processing across all documents
                            
                            #### **Excel Specific Method (LlamaIndex only) - EXPERIMENTAL**
                            
                            **Best for:** Excel files and spreadsheet data
                            
                            **Features:**
                            * Uses LlamaIndex for Excel files only (EXPERIMENTAL)
                            * Advanced Excel chunking (row-based, column-aware)
                            * Better accuracy for spreadsheet queries
                            * Preserves Excel structure and relationships
                            
                            **When to use:**
                            * Working primarily with Excel files
                            * Need precise spreadsheet data extraction
                            * Employee records, inventory, budgets
                            * When you need precise Excel data extraction
                            
                            **⚠️ Important Note:** This feature is experimental and may have limitations or unexpected behavior. Use with caution for production environments.
                            
                            #### **Parameter Impact by Method**
                            
                            | Parameter | Standard Method | Excel Specific Method |
                            |-----------|----------------|---------------------|
                            | Chunks to Retrieve |  Used (1-40) |  Ignored (processes all data) |
                            | Temperature |  Used |  Used |
                            | Max Tokens |  Used |  Used |
                            | Model Selection |  Used |  Used |
                            """)
                        
                        with gr.Accordion("⚙️ LLM Parameters Explained", open=False):
                            gr.Markdown("""
                            ### Understanding Query Parameters
                            
                            #### **LLM Model Selection**
                            
                            **Model Types (examples):**
                            * **Small models (3B parameters)**: Fast, good for general queries, lower resource usage
                            * **Medium models (7B parameters)**: Better quality, more detailed responses, moderate resource usage
                            * **Large models (13B+ parameters)**: High quality, very detailed responses, higher resource usage
                            
                            **Note**: Available models depend on what you have installed in Ollama. Use the "Refresh" button to see your current models.
                            
                            #### **Search Parameters**
                            
                            **Chunks to Retrieve (1-40, Default: 20)**
                            * **Lower values (5-10)**: Faster responses, more focused answers
                            * **Higher values (20-40)**: More comprehensive answers, slower responses
                            * **Impact**: More chunks = more context = longer, more detailed answers
                            
                            **Temperature (0.1-1.0, Default: 0.7)**
                            * **0.1-0.3**: Very focused, deterministic responses
                            * **0.4-0.6**: Balanced creativity and accuracy
                            * **0.7-0.9**: More creative, varied responses
                            * **1.0**: Maximum creativity, may be less accurate
                            * **Impact**: Higher temperature = more creative but potentially less accurate answers
                            
                            **Max Tokens (100-4000, Default: 2048)**
                            * **100-500**: Short, concise answers
                            * **1000-2000**: Standard length answers
                            * **3000-4000**: Very detailed, comprehensive answers
                            * **Impact**: Higher values = longer responses, lower values = shorter responses
                            
                            #### **Parameter Combinations**
                            
                            **For Quick Facts:**
                            * Chunks: 5-10, Temperature: 0.3, Max Tokens: 500
                            
                            **For Detailed Analysis:**
                            * Chunks: 20-30, Temperature: 0.7, Max Tokens: 3000
                            
                            **For Creative Exploration:**
                            * Chunks: 15-25, Temperature: 0.9, Max Tokens: 2500
                            
                            **For Academic Research:**
                            * Chunks: 30-40, Temperature: 0.5, Max Tokens: 4000
                            """)
                        
                        with gr.Accordion("Excel Querying Examples", open=False):
                            gr.Markdown("""
                            ### Excel-Specific Querying (EXPERIMENTAL)
                            
                            **Important:** Use the "Excel Specific" query method for best results with Excel files.
                            
                            **⚠️ Note:** Excel processing with LlamaIndex is experimental and may have limitations.
                            
                            Excel files are processed with advanced LlamaIndex chunking that makes each row and column searchable:
                            
                            #### **Employee Data Queries**
                            * "What is John Doe's salary?" → Finds specific employee information
                            * "Show me all employees in the Sales department" → Filters by department
                            * "Find employees hired after 2020" → Date-based filtering
                            * "Who has the highest salary?" → Comparative queries
                            * "List all employees with salary between $50,000 and $75,000" → Range queries
                            
                            #### **Financial Data Queries**
                            * "What is the total revenue for Q1?" → Aggregation queries
                            * "Show me all expenses over $1000" → Threshold filtering
                            * "What was the budget for Marketing in 2023?" → Category and time filtering
                            * "Find transactions with 'Airbus' in the description" → Text search
                            * "Calculate the average monthly expenses" → Statistical queries
                            
                            #### **Inventory/Product Queries**
                            * "How many units of Product X are in stock?" → Quantity queries
                            * "What products are out of stock?" → Status filtering
                            * "Show me all products with price over $50" → Price filtering
                            * "Which supplier has the most products?" → Supplier analysis
                            * "Find products with low stock levels" → Business intelligence queries
                            
                            #### **Advanced Excel Queries**
                            * "Compare sales performance between Q1 and Q2" → Comparative analysis
                            * "Show me the top 10 customers by revenue" → Ranking queries
                            * "What are the trends in monthly expenses?" → Trend analysis
                            * "Find all duplicate entries in the customer list" → Data quality queries
                            
                            #### **Tips for Excel Queries**
                            * **Use Excel Specific method**: Select "Excel Specific" for best results
                            * **Be specific**: Use exact column names or values when possible
                            * **Use natural language**: "What is the salary of..." works better than "SELECT salary WHERE..."
                            * **Include context**: "Show me sales data for 2023" is clearer than "sales 2023"
                            * **Try variations**: If one query doesn't work, try rephrasing
                            * **Note**: The "Chunks to Retrieve" parameter is ignored in Excel Specific mode (processes all data)
                            """)
                        
                        with gr.Accordion("💡 Best Practices", open=False):
                            gr.Markdown("""
                            ### Getting the Best Results
                            
                            #### **Query Optimization**
                            * **Start with simple questions**: Build understanding gradually
                            * **Use specific terms**: Include relevant keywords from your documents
                            * **Ask one question at a time**: Avoid complex multi-part questions
                            * **Refine based on results**: Use initial answers to ask more specific follow-ups
                            
                            #### **Parameter Combinations**
                            
                            **For Quick Facts:**
                            * Chunks: 5-10, Temperature: 0.3, Max Tokens: 500
                            
                            **For Detailed Analysis:**
                            * Chunks: 20-30, Temperature: 0.7, Max Tokens: 3000
                            
                            **For Creative Exploration:**
                            * Chunks: 15-25, Temperature: 0.9, Max Tokens: 2500
                            
                            **For Academic Research:**
                            * Chunks: 30-40, Temperature: 0.5, Max Tokens: 4000
                            """)
                        
                        with gr.Accordion("🔍 Query Interface", open=False):
                            gr.Markdown("""
                            ### Asking Questions
                            
                            **How to Query:**
                            1. **Enter your question** in natural language (French or English)
                            2. **Choose your query method** (Standard or Excel Specific)
                            3. **Select your preferred LLM model** from the dropdown
                            4. **Adjust search parameters** if needed
                            5. **Click "Query Documents"** and wait for the response
                            
                            **Query Tips:**
                            * **Be specific**: "What is the main argument in the document about climate change?" vs "What is this about?"
                            * **Use context**: "According to the author, what are the benefits of renewable energy?"
                            * **Ask follow-up questions**: "Can you elaborate on the methodology mentioned in the previous answer?"
                            * **Cross-document queries**: "Compare the approaches discussed in Document A and Document B"
                            
                            **Language Support:**
                            * **French queries** work best with French documents
                            * **English queries** work best with English documents
                            * **Cross-language queries** are supported but may have reduced accuracy
                            """)
                        
                        with gr.Accordion("⚙️ Query Methods", open=False):
                            gr.Markdown("""
                            ### Understanding Query Methods
                            
                            The system offers two distinct query methods optimized for different document types:
                            
                            #### **Standard Method (LEANN only)**
                            
                            **Best for:** General documents (PDFs, Word docs, text files, etc.)
                            
                            **Features:**
                            * Uses LEANN vector store for all document types
                            * Fast and efficient processing
                            * Good for general text-based queries
                            * Processes all document types uniformly
                            
                            **When to use:**
                            * Querying PDFs, Word documents, text files
                            * General research and analysis
                            * When you want consistent processing across all documents
                            
                            #### **Excel Specific Method (LlamaIndex only)**
                            
                            **Best for:** Excel files and spreadsheet data
                            
                            **Features:**
                            * Uses LlamaIndex for Excel files only
                            * Advanced Excel chunking (row-based, column-aware)
                            * Better accuracy for spreadsheet queries
                            * Processes ALL Excel files and ALL their data
                            * Ignores the "Chunks to Retrieve" parameter (processes everything)
                            
                            **When to use:**
                            * Querying Excel files and spreadsheets
                            * Financial data analysis
                            * Employee records, inventory, budgets
                            * When you need precise Excel data extraction
                            
                            #### **Parameter Impact by Method**
                            
                            **Standard Method:**
                            *  **Chunks to Retrieve**: Controls how many document chunks to search
                            *  **Temperature**: Controls response creativity
                            *  **Max Tokens**: Controls response length
                            *  **Model**: Controls which LLM to use
                            
                            **Excel Specific Method:**
                            *  **Chunks to Retrieve**: Ignored (processes ALL Excel data)
                            *  **Temperature**: Controls response creativity
                            *  **Max Tokens**: Controls response length
                            *  **Model**: Controls which LLM to use
                            
                            #### **Choosing the Right Method**
                            
                            **Use Standard when:**
                            * You have mixed document types
                            * You want fast, general queries
                            * You're working with PDFs, Word docs, or text files
                            
                            **Use Excel Specific when:**
                            * You're primarily working with Excel files
                            * You need precise spreadsheet data extraction
                            * You want the most accurate Excel querying
                            * You're doing financial or data analysis
                            """)
                        
                        with gr.Accordion("🚀 Advanced Features", open=False):
                            gr.Markdown("""
                            ### Advanced Usage
                            
                            #### **Multi-Document Queries**
                            * **Cross-document comparison**: "Compare the methodologies in Document A and Document B"
                            * **Synthesis queries**: "Summarize the main themes across all uploaded documents"
                            * **Contradiction detection**: "Are there any contradictions between these documents?"
                            
                            #### **Research Workflows**
                            1. **Upload research papers** in your field
                            2. **Ask for summaries** of key findings
                            3. **Request comparisons** between different approaches
                            4. **Generate research questions** based on gaps in the literature
                            
                            #### **Content Analysis**
                            * **Sentiment analysis**: "What is the overall tone of this document?"
                            * **Key concept extraction**: "What are the main concepts discussed?"
                            * **Timeline creation**: "Create a timeline of events mentioned in these documents"
                            """)
                        
                        with gr.Accordion("ℹ️ System Information", open=False):
                            gr.Markdown("""
                            ### Technical Details
                            
                            **Document Processing:**
                            * **Chunking Strategy**: Sentence-based with paragraph awareness
                            * **Chunk Size**: 400 characters with 100-character overlap
                            * **Text Extraction**: Docling (primary) with pypdf fallback for PDFs
                            * **Supported Formats**: PDF, DOCX, XLSX, PPTX, TXT, MD, HTML, XHTML, CSV, PNG, JPEG, TIFF, BMP, WEBP, AsciiDoc, XML
                            * **Processing Dependencies**: Docling for complex formats, pypdf for web-printed PDFs, direct text reading for simple files
                            * **Embedding Model**: nomic-embed-text-v2-moe (768 dimensions)
                            * **Vector Database**: LEANN with ultra-efficient storage (97% space savings)
                            
                            **LLM Integration:**
                            * **Local Processing**: All queries processed locally via Ollama
                            * **Model Support**: Any Ollama-compatible model
                            * **GPU Acceleration**: MPS support for Apple Silicon
                            * **Response Time**: 15-30 seconds depending on model and parameters
                            
                            **Test Environment:**
                            * **Hardware**: Mac mini (Model Identifier: Mac16,10)
                            * **Processor**: Apple M4 chip
                            * **Memory**: 16GB unified memory
                            * **Languages Tested**: English, French, Spanish, German
                            * **Performance**: Excellent across all supported languages
                            * **Architecture**: Local processing for optimal performance and complete data privacy
                            """)
                        
                        with gr.Accordion("🔧 Model Installation", open=False):
                            gr.Markdown("""
                            ### Installing LLM Models
                            
                            **To use the system, you need to install models in Ollama:**
                            
                            #### **Popular Model Options:**
                            
                            **For General Use:**
                            ```bash
                            ollama pull llama3.2:3b    # Fast, good quality
                            ollama pull llama3.2:7b    # Better quality
                            ```
                            
                            **For High Quality:**
                            ```bash
                            ollama pull llama3.2:13b   # High quality
                            ollama pull llama3.2:70b   # Best quality (requires more resources)
                            ```
                            
                            **For French Content:**
                            ```bash
                            ollama pull mistral:7b     # Good French support
                            ollama pull mixtral:8x7b   # Excellent multilingual
                            ```
                            
                            #### **Installation Steps:**
                            1. **Install Ollama** from https://ollama.ai
                            2. **Pull a model** using the commands above
                            3. **Refresh the model list** in the interface
                            4. **Select your model** from the dropdown
                            
                            **Note**: Model download can take several minutes depending on size and internet speed.
                            """)
                        
                        with gr.Accordion("⚠️ Limitations", open=False):
                            gr.Markdown("""
                            ### Current and Potential Limitations
                            
                            #### **Document Processing Limitations**
                            * **Scanned PDFs**: Fully supported with automatic OCR processing
                            * **Complex layouts**: Tables, multi-column layouts may not be perfectly preserved
                            * **Large files**: Very large documents (>100MB) may cause processing delays
                            * **Image content**: Text within images is automatically extracted with OCR
                            
                            #### **Language and Model Limitations**
                            * **Language sensitivity**: Cross-language queries may have reduced accuracy
                            * **Model dependency**: Performance varies significantly between different LLM models
                            * **Context window**: Very long documents may exceed model context limits
                            * **Specialized domains**: Technical or domain-specific content may require specialized models
                            
                            #### **System Limitations**
                            * **Local processing**: All processing happens locally, requiring adequate hardware resources
                            * **Memory usage**: Large document collections require significant RAM
                            * **Processing time**: Complex queries with large context can take 30+ seconds
                            * **Concurrent users**: System designed for single-user or small team usage
                            
                            """)
                        
                        with gr.Accordion("🔧 System Management", open=False):
                            gr.Markdown(get_english_guide())
                    
                    # French Guide
                    with gr.Group(visible=False) as french_guide:
                        with gr.Accordion("🎯 Vue d'ensemble du système", open=False):
                            gr.Markdown("""
                            **Myr-Ag** est un système RAG (Retrieval-Augmented Generation) puissant qui vous permet de :
                            
                            * **Télécharger et traiter des documents** dans plusieurs formats (PDF, DOCX, XLSX, PPTX, TXT, MD, HTML, XHTML, CSV, PNG, JPEG, TIFF, BMP, WEBP, AsciiDoc, XML)
                            * **Poser des questions** sur vos documents en langage naturel
                            * **Obtenir des réponses intelligentes** alimentées par des modèles LLM locaux
                            * **Rechercher dans plusieurs documents** avec une compréhension sémantique
                            
                            **Architecture de traitement dual :**
                            * **Méthode Standard** : Utilise le magasin vectoriel LEANN pour les documents généraux (PDFs, documents Word, fichiers texte)
                            * **Méthode Spécifique Excel** : Utilise LlamaIndex pour les fichiers Excel avec un traitement avancé des feuilles de calcul (⚠️ EXPÉRIMENTAL)
                            
                            Le système utilise un traitement de documents avancé, des embeddings vectoriels et une inférence LLM locale pour fournir des réponses précises et contextuelles optimisées pour différents types de documents.
                            """)
                        
                        with gr.Accordion(" Gestion des documents", open=False):
                            gr.Markdown("""
                            ### Téléchargement de documents
                            
                            **Formats supportés :**
                            
                            **Documents Office :**
                            * **PDF** : PDFs standards, PDFs scannés avec OCR, et PDFs imprimés depuis le web
                            * **DOCX** : Documents Microsoft Word
                            * **XLSX** : Feuilles de calcul Microsoft Excel
                            * **PPTX** : Présentations PowerPoint
                            
                            **Texte et Web :**
                            * **TXT** : Fichiers texte simples
                            * **MD** : Fichiers Markdown
                            * **HTML/XHTML** : Pages web et documents structurés
                            * **CSV** : Fichiers de valeurs séparées par des virgules
                            
                            **Images (avec OCR) :**
                            * **PNG, JPEG, TIFF, BMP, WEBP** : Documents scannés et images avec extraction automatique de texte
                            
                            **Formats spécialisés :**
                            * **AsciiDoc** : Documentation technique
                            * **XML** : Données et documents structurés
                            
                            **Options de téléchargement :**
                            * **Télécharger et traiter** : Télécharger et traiter immédiatement les documents pour les requêtes
                            * **Télécharger seulement** : Télécharger les documents sans les traiter (utile pour les opérations par lot)
                            * **Traiter l'existant** : Traiter les documents déjà dans le répertoire uploads
                            * **Traiter les téléchargés seulement** : Traiter uniquement les documents qui n'ont pas encore été traités
                            """)
                        
                        with gr.Accordion("🔍 Méthodes de requête", open=False):
                            gr.Markdown("""
                            ### Comprendre les méthodes de requête
                            
                            Le système offre deux méthodes de requête distinctes optimisées pour différents types de documents :
                            
                            #### **Méthode Standard (LEANN uniquement)**
                            
                            **Idéal pour :** Documents généraux (PDFs, documents Word, fichiers texte, images)
                            
                            **Fonctionnalités :**
                            * Utilise le magasin vectoriel LEANN pour tous les types de documents
                            * Découpage basé sur les phrases pour les documents texte
                            * Traitement OCR pour les images et documents scannés
                            * Traitement cohérent pour tous les formats supportés
                            
                            **Quand l'utiliser :**
                            * Recherche et analyse de documents généraux
                            * Collections de documents mixtes
                            * Quand vous voulez un traitement cohérent pour tous les documents
                            
                            #### **Méthode Spécifique Excel (LlamaIndex uniquement) - EXPÉRIMENTAL**
                            
                            **Idéal pour :** Fichiers Excel et données de feuilles de calcul
                            
                            **Fonctionnalités :**
                            * Utilise LlamaIndex pour les fichiers Excel uniquement (EXPÉRIMENTAL)
                            * Découpage Excel avancé (basé sur les lignes, conscient des colonnes)
                            * Meilleure précision pour les requêtes de feuilles de calcul
                            * Préserve la structure et les relations Excel
                            
                            **Quand l'utiliser :**
                            * Travailler principalement avec des fichiers Excel
                            * Besoin d'extraction précise de données de feuilles de calcul
                            * Dossiers d'employés, inventaires, budgets
                            * Quand vous avez besoin d'extraction précise de données Excel
                            
                            **⚠️ Note importante :** Cette fonctionnalité est expérimentale et peut avoir des limitations ou des comportements inattendus. Utilisez avec prudence dans les environnements de production.
                            
                            #### **Impact des paramètres par méthode**
                            
                            | Paramètre | Méthode Standard | Méthode Spécifique Excel |
                            |-----------|------------------|-------------------------|
                            | Chunks à récupérer |  Utilisé (1-40) |  Ignoré (traite toutes les données) |
                            | Température |  Utilisé |  Utilisé |
                            | Max Tokens |  Utilisé |  Utilisé |
                            | Sélection du modèle |  Utilisé |  Utilisé |
                            """)
                        
                        with gr.Accordion("⚙️ Paramètres LLM expliqués", open=False):
                            gr.Markdown("""
                            ### Comprendre les paramètres de requête
                            
                            #### **Sélection du modèle LLM**
                            
                            **Types de modèles (exemples) :**
                            * **Petits modèles (3B paramètres)** : Rapides, bons pour les requêtes générales, faible utilisation des ressources
                            * **Modèles moyens (7B paramètres)** : Meilleure qualité, réponses plus détaillées, utilisation modérée des ressources
                            * **Grands modèles (13B+ paramètres)** : Haute qualité, réponses très détaillées, utilisation élevée des ressources
                            
                            **Note** : Les modèles disponibles dépendent de ce que vous avez installé dans Ollama. Utilisez le bouton "Actualiser" pour voir vos modèles actuels.
                            
                            #### **Paramètres de recherche**
                            
                            **Chunks à récupérer (1-40, Par défaut : 20)**
                            * **Valeurs faibles (5-10)** : Réponses plus rapides, réponses plus ciblées
                            * **Valeurs élevées (20-40)** : Réponses plus complètes, réponses plus lentes
                            * **Impact** : Plus de chunks = plus de contexte = réponses plus longues et détaillées
                            
                            **Température (0.1-1.0, Par défaut : 0.7)**
                            * **0.1-0.3** : Réponses très ciblées, déterministes
                            * **0.4-0.6** : Créativité et précision équilibrées
                            * **0.7-0.9** : Plus créatif, réponses variées
                            * **1.0** : Créativité maximale, peut être moins précis
                            * **Impact** : Température plus élevée = plus créatif mais potentiellement moins précis
                            
                            **Max Tokens (100-4000, Par défaut : 2048)**
                            * **100-500** : Réponses courtes et concises
                            * **1000-2000** : Réponses de longueur standard
                            * **3000-4000** : Réponses très détaillées et complètes
                            * **Impact** : Valeurs plus élevées = réponses plus longues, valeurs plus faibles = réponses plus courtes
                            
                            #### **Combinaisons de paramètres**
                            
                            **Pour des faits rapides :**
                            * Chunks : 5-10, Température : 0.3, Max Tokens : 500
                            
                            **Pour une analyse détaillée :**
                            * Chunks : 20-30, Température : 0.7, Max Tokens : 3000
                            
                            **Pour une exploration créative :**
                            * Chunks : 15-25, Température : 0.9, Max Tokens : 2500
                            
                            **Pour la recherche académique :**
                            * Chunks : 30-40, Température : 0.5, Max Tokens : 4000
                            """)
                        
                        with gr.Accordion("💡 Meilleures pratiques", open=False):
                            gr.Markdown("""
                            ### Obtenir les meilleurs résultats
                            
                            #### **Optimisation des requêtes**
                            * **Commencez par des questions simples** : Construisez la compréhension progressivement
                            * **Utilisez des termes spécifiques** : Incluez des mots-clés pertinents de vos documents
                            * **Posez une question à la fois** : Évitez les questions complexes en plusieurs parties
                            * **Affinez selon les résultats** : Utilisez les réponses initiales pour poser des questions de suivi plus spécifiques
                            
                            #### **Combinaisons de paramètres**
                            
                            **Pour des faits rapides :**
                            * Chunks : 5-10, Température : 0.3, Max Tokens : 500
                            
                            **Pour une analyse détaillée :**
                            * Chunks : 20-30, Température : 0.7, Max Tokens : 3000
                            
                            **Pour une exploration créative :**
                            * Chunks : 15-25, Température : 0.9, Max Tokens : 2500
                            
                            **Pour la recherche académique :**
                            * Chunks : 30-40, Température : 0.5, Max Tokens : 4000
                            """)
                        
                        with gr.Accordion("🔍 Interface de requête", open=False):
                            gr.Markdown("""
                            ### Poser des questions
                            
                            **Comment faire une requête :**
                            1. **Entrez votre question** en langage naturel (français ou anglais)
                            2. **Choisissez votre méthode de requête** (Standard ou Spécifique Excel)
                            3. **Sélectionnez votre modèle LLM préféré** dans le menu déroulant
                            4. **Ajustez les paramètres de recherche** si nécessaire
                            5. **Cliquez sur "Query Documents"** et attendez la réponse
                            
                            **Conseils pour les requêtes :**
                            * **Soyez spécifique** : "Quel est l'argument principal du document sur le changement climatique ?" vs "De quoi s'agit-il ?"
                            * **Utilisez le contexte** : "Selon l'auteur, quels sont les avantages des énergies renouvelables ?"
                            * **Posez des questions de suivi** : "Pouvez-vous élaborer sur la méthodologie mentionnée dans la réponse précédente ?"
                            * **Requêtes multi-documents** : "Comparez les approches discutées dans le Document A et le Document B"
                            
                            **Support linguistique :**
                            * **Requêtes en français** fonctionnent mieux avec des documents français
                            * **Requêtes en anglais** fonctionnent mieux avec des documents anglais
                            * **Requêtes interlangues** sont supportées mais peuvent avoir une précision réduite
                            """)
                        
                        with gr.Accordion("⚙️ Méthodes de requête", open=False):
                            gr.Markdown("""
                            ### Comprendre les méthodes de requête
                            
                            Le système propose deux méthodes de requête distinctes optimisées pour différents types de documents :
                            
                            #### **Méthode Standard (LEANN uniquement)**
                            
                            **Idéal pour :** Documents généraux (PDFs, documents Word, fichiers texte, etc.)
                            
                            **Fonctionnalités :**
                            * Utilise le magasin vectoriel LEANN pour tous les types de documents
                            * Traitement rapide et efficace
                            * Bon pour les requêtes basées sur le texte
                            * Traite tous les types de documents de manière uniforme
                            
                            **Quand l'utiliser :**
                            * Interroger des PDFs, documents Word, fichiers texte
                            * Recherche et analyse générale
                            * Quand vous voulez un traitement cohérent de tous les documents
                            
                            #### **Méthode Spécifique Excel (LlamaIndex uniquement)**
                            
                            **Idéal pour :** Fichiers Excel et données de feuilles de calcul
                            
                            **Fonctionnalités :**
                            * Utilise LlamaIndex pour les fichiers Excel uniquement
                            * Découpage Excel avancé (basé sur les lignes, conscient des colonnes)
                            * Meilleure précision pour les requêtes de feuilles de calcul
                            * Traite TOUS les fichiers Excel et TOUTES leurs données
                            * Ignore le paramètre "Chunks à récupérer" (traite tout)
                            
                            **Quand l'utiliser :**
                            * Interroger des fichiers Excel et des feuilles de calcul
                            * Analyse de données financières
                            * Dossiers d'employés, inventaire, budgets
                            * Quand vous avez besoin d'une extraction précise des données Excel
                            
                            #### **Impact des paramètres par méthode**
                            
                            **Méthode Standard :**
                            *  **Chunks à récupérer** : Contrôle le nombre de fragments de documents à rechercher
                            *  **Température** : Contrôle la créativité de la réponse
                            *  **Max Tokens** : Contrôle la longueur de la réponse
                            *  **Modèle** : Contrôle quel LLM utiliser
                            
                            **Méthode Spécifique Excel :**
                            *  **Chunks à récupérer** : Ignoré (traite TOUTES les données Excel)
                            *  **Température** : Contrôle la créativité de la réponse
                            *  **Max Tokens** : Contrôle la longueur de la réponse
                            *  **Modèle** : Contrôle quel LLM utiliser
                            
                            #### **Choisir la bonne méthode**
                            
                            **Utilisez Standard quand :**
                            * Vous avez des types de documents mixtes
                            * Vous voulez des requêtes rapides et générales
                            * Vous travaillez avec des PDFs, documents Word ou fichiers texte
                            
                            **Utilisez Spécifique Excel quand :**
                            * Vous travaillez principalement avec des fichiers Excel
                            * Vous avez besoin d'une extraction précise des données de feuilles de calcul
                            * Vous voulez la requête Excel la plus précise
                            * Vous faites de l'analyse financière ou de données
                            """)
                        
                        with gr.Accordion("🚀 Fonctionnalités avancées", open=False):
                            gr.Markdown("""
                            ### Utilisation avancée
                            
                            #### **Requêtes multi-documents**
                            * **Comparaison inter-documents** : "Comparez les méthodologies du Document A et du Document B"
                            * **Requêtes de synthèse** : "Résumez les thèmes principaux à travers tous les documents téléchargés"
                            * **Détection de contradictions** : "Y a-t-il des contradictions entre ces documents ?"
                            
                            #### **Workflows de recherche**
                            1. **Téléchargez des articles de recherche** dans votre domaine
                            2. **Demandez des résumés** des principales découvertes
                            3. **Demandez des comparaisons** entre différentes approches
                            4. **Générez des questions de recherche** basées sur les lacunes dans la littérature
                            
                            #### **Analyse de contenu**
                            * **Analyse de sentiment** : "Quel est le ton général de ce document ?"
                            * **Extraction de concepts clés** : "Quels sont les concepts principaux discutés ?"
                            * **Création de chronologie** : "Créez une chronologie des événements mentionnés dans ces documents"
                            """)
                        
                        with gr.Accordion("ℹ️ Informations système", open=False):
                            gr.Markdown("""
                            ### Détails techniques
                            
                            **Traitement de documents :**
                            * **Stratégie de chunking** : Basée sur les phrases avec conscience des paragraphes
                            * **Taille des chunks** : 400 caractères avec un chevauchement de 100 caractères
                            * **Extraction de texte** : Docling (principal) avec fallback pypdf pour les PDFs
                            * **Formats supportés** : PDF, DOCX, XLSX, PPTX, TXT, MD, HTML, XHTML, CSV, PNG, JPEG, TIFF, BMP, WEBP, AsciiDoc, XML
                            * **Dépendances de traitement** : Docling pour les formats complexes, pypdf pour les PDFs imprimés depuis le web, lecture directe pour les fichiers simples
                            * **Modèle d'embedding** : nomic-embed-text-v2-moe (768 dimensions)
                            * **Base de données vectorielle** : LEANN avec stockage ultra-efficace (97% d'économie d'espace)
                            
                            **Intégration LLM :**
                            * **Traitement local** : Toutes les requêtes traitées localement via Ollama
                            * **Support de modèles** : Tout modèle compatible Ollama
                            * **Accélération GPU** : Support MPS pour Apple Silicon
                            * **Temps de réponse** : 15-30 secondes selon le modèle et les paramètres
                            
                            **Environnement de test :**
                            * **Matériel** : Mac mini (Identifiant modèle : Mac16,10)
                            * **Processeur** : Puce Apple M4
                            * **Mémoire** : 16GB de mémoire unifiée
                            * **Langues testées** : Anglais, français, espagnol, allemand
                            * **Performance** : Excellente dans toutes les langues supportées
                            * **Architecture** : Traitement local pour des performances optimales et une confidentialité complète des données
                            """)
                        
                        with gr.Accordion("🔧 Installation des modèles", open=False):
                            gr.Markdown("""
                            ### Installation des modèles LLM
                            
                            **Pour utiliser le système, vous devez installer des modèles dans Ollama :**
                            
                            #### **Options de modèles populaires :**
                            
                            **Pour usage général :**
                            ```bash
                            ollama pull llama3.2:3b    # Rapide, bonne qualité
                            ollama pull llama3.2:7b    # Meilleure qualité
                            ```
                            
                            **Pour haute qualité :**
                            ```bash
                            ollama pull llama3.2:13b   # Haute qualité
                            ollama pull llama3.2:70b   # Meilleure qualité (nécessite plus de ressources)
                            ```
                            
                            **Pour contenu français :**
                            ```bash
                            ollama pull mistral:7b     # Bon support français
                            ollama pull mixtral:8x7b   # Excellent multilingue
                            ```
                            
                            #### **Étapes d'installation :**
                            1. **Installez Ollama** depuis https://ollama.ai
                            2. **Téléchargez un modèle** en utilisant les commandes ci-dessus
                            3. **Actualisez la liste des modèles** dans l'interface
                            4. **Sélectionnez votre modèle** dans le menu déroulant
                            
                            **Note** : Le téléchargement des modèles peut prendre plusieurs minutes selon la taille et la vitesse d'internet.
                            """)
                        
                        with gr.Accordion("⚠️ Limitations", open=False):
                            gr.Markdown("""
                            ### Limitations actuelles et probables
                            
                            #### **Limitations du traitement de documents**
                            * **PDFs scannés** : Entièrement supportés avec traitement OCR automatique
                            * **Mises en page complexes** : Tableaux, mises en page multi-colonnes peuvent ne pas être parfaitement préservés
                            * **Fichiers volumineux** : Documents très volumineux (>100MB) peuvent causer des délais de traitement
                            * **Contenu d'images** : Le texte dans les images est automatiquement extrait avec OCR
                            
                            #### **Limitations linguistiques et de modèles**
                            * **Sensibilité linguistique** : Les requêtes interlangues peuvent avoir une précision réduite
                            * **Dépendance au modèle** : Les performances varient considérablement entre différents modèles LLM
                            * **Fenêtre de contexte** : Documents très longs peuvent dépasser les limites de contexte du modèle
                            * **Domaines spécialisés** : Contenu technique ou spécialisé peut nécessiter des modèles spécialisés
                            
                            #### **Limitations système**
                            * **Traitement local** : Tout le traitement se fait localement, nécessitant des ressources matérielles adéquates
                            * **Utilisation mémoire** : De grandes collections de documents nécessitent une RAM importante
                            * **Temps de traitement** : Requêtes complexes avec un grand contexte peuvent prendre 30+ secondes
                            * **Utilisateurs simultanés** : Système conçu pour un utilisateur unique ou une petite équipe
                            
                            """)
                        
                        with gr.Accordion("🔧 Gestion du système", open=False):
                            gr.Markdown(get_french_guide())
                    
                    # Spanish Guide
                    with gr.Group(visible=False) as spanish_guide:
                        with gr.Accordion("🎯 Descripción general del sistema", open=False):
                            gr.Markdown("""
                            **Myr-Ag** es un sistema RAG (Retrieval-Augmented Generation) potente que te permite:
                            
                            * **Cargar y procesar documentos** en múltiples formatos (PDF, DOCX, XLSX, PPTX, TXT, MD, HTML, XHTML, CSV, PNG, JPEG, TIFF, BMP, WEBP, AsciiDoc, XML)
                            * **Hacer preguntas** sobre tus documentos en lenguaje natural
                            * **Obtener respuestas inteligentes** alimentadas por modelos LLM locales
                            * **Buscar en múltiples documentos** con comprensión semántica
                            
                            **Arquitectura de procesamiento dual:**
                            * **Método Estándar**: Utiliza el almacén vectorial LEANN para documentos generales (PDFs, documentos Word, archivos de texto)
                            * **Método Específico Excel**: Utiliza LlamaIndex para archivos Excel con procesamiento avanzado de hojas de cálculo (⚠️ EXPERIMENTAL)
                            
                            El sistema utiliza procesamiento avanzado de documentos, embeddings vectoriales e inferencia LLM local para proporcionar respuestas precisas y contextuales optimizadas para diferentes tipos de documentos.
                            """)
                        
                        with gr.Accordion(" Gestión de documentos", open=False):
                            gr.Markdown("""
                            ### Carga de documentos
                            
                            **Formatos soportados:**
                            
                            **Documentos de Office:**
                            * **PDF** : PDFs estándar, PDFs escaneados con OCR, y PDFs impresos desde web
                            * **DOCX** : Documentos de Microsoft Word
                            * **XLSX** : Hojas de cálculo de Microsoft Excel (⚠️ EXPERIMENTAL)
                            * **PPTX** : Presentaciones de Microsoft PowerPoint
                            
                            **Contenido Web:**
                            * **HTML/XHTML** : Páginas web y documentos HTML
                            
                            **Formatos de texto:**
                            * **TXT** : Archivos de texto plano
                            * **MD** : Documentos Markdown
                            * **AsciiDoc** : Formato de documentación AsciiDoc
                            
                            **Formatos de datos:**
                            * **CSV** : Archivos de valores separados por comas
                            
                            **Formatos de imagen (con OCR):**
                            * **PNG, JPEG, TIFF, BMP, WEBP** : Imágenes con extracción automática de texto
                            
                            **Formatos especializados:**
                            * **XML** : Lenguaje de marcado extensible
                            * **USPTO XML** : Documentos de patentes USPTO
                            * **JATS XML** : Journal Article Tag Suite (artículos científicos)
                            """)
                        
                        with gr.Accordion("🔧 Gestión del sistema", open=False):
                            gr.Markdown(get_spanish_guide())
                        
                        with gr.Accordion("💡 Mejores prácticas", open=False):
                            gr.Markdown("""
                            ### Obtener los mejores resultados
                            
                            #### **Optimización de consultas**
                            * **Comienza con preguntas simples** : Construye la comprensión progresivamente
                            * **Usa términos específicos** : Incluye palabras clave relevantes de tus documentos
                            * **Haz una pregunta a la vez** : Evita preguntas complejas en múltiples partes
                            * **Refina según los resultados** : Usa las respuestas iniciales para hacer preguntas de seguimiento más específicas
                            
                            #### **Combinaciones de parámetros**
                            
                            **Para hechos rápidos:**
                            * Chunks: 5-10, Temperatura: 0.3, Max Tokens: 500
                            
                            **Para análisis detallado:**
                            * Chunks: 20-30, Temperatura: 0.7, Max Tokens: 3000
                            
                            **Para exploración creativa:**
                            * Chunks: 15-25, Temperatura: 0.9, Max Tokens: 2500
                            
                            #### **Estrategias de búsqueda**
                            
                            **Búsqueda semántica:**
                            * **Consulta natural** : "¿Cuáles son los principales hallazgos del estudio?"
                            * **Términos específicos** : "ROI", "presupuesto", "cronograma"
                            * **Contexto amplio** : "Análisis financiero del Q3"
                            
                            **Búsqueda de Excel:**
                            * **Preguntas sobre datos** : "¿Cuál es el salario de Juan?"
                            * **Filtros de datos** : "Muestra todos los empleados de ventas"
                            * **Análisis de tendencias** : "¿Cómo ha cambiado el rendimiento?"
                            
                            #### **Optimización del rendimiento**
                            
                            **Configuración de chunks:**
                            * **Documentos técnicos** : 10-15 chunks para contexto completo
                            * **Documentos narrativos** : 5-10 chunks para respuestas concisas
                            * **Hojas de cálculo** : 20-30 chunks para análisis detallado
                            
                            **Configuración de temperatura:**
                            * **Hechos precisos** : 0.1-0.3 para respuestas consistentes
                            * **Análisis creativo** : 0.7-0.9 para respuestas innovadoras
                            * **Equilibrio** : 0.5-0.7 para respuestas balanceadas
                            
                            #### **Solución de problemas comunes**
                            
                            **Respuestas irrelevantes:**
                            * Ajusta el número de chunks
                            * Refina la consulta con términos más específicos
                            * Verifica que los documentos estén correctamente procesados
                            
                            **Respuestas incompletas:**
                            * Aumenta el número máximo de tokens
                            * Ajusta la temperatura para más creatividad
                            * Usa consultas de seguimiento más específicas
                            
                            **Problemas de rendimiento:**
                            * Reduce el número de chunks para consultas rápidas
                            * Usa modelos más pequeños para respuestas más rápidas
                            * Limpia datos no utilizados regularmente
                            """)
                            gr.Markdown("""
                            **Myr-Ag** es un potente sistema RAG (Retrieval-Augmented Generation) que te permite:
                            
                            * **Cargar y procesar documentos** en múltiples formatos (PDF, DOCX, XLSX, PPTX, TXT, MD, HTML, XHTML, CSV, PNG, JPEG, TIFF, BMP, WEBP, AsciiDoc, XML)
                            * **Hacer preguntas** sobre tus documentos en lenguaje natural
                            * **Obtener respuestas inteligentes** impulsadas por modelos LLM locales
                            * **Buscar en múltiples documentos** con comprensión semántica
                            
                            **Arquitectura de procesamiento dual:**
                            * **Método Estándar**: Utiliza el almacén vectorial LEANN para documentos generales (PDFs, documentos Word, archivos de texto)
                            * **Método Específico Excel**: Utiliza LlamaIndex para archivos Excel con procesamiento avanzado de hojas de cálculo (⚠️ EXPERIMENTAL)
                            
                            El sistema utiliza procesamiento avanzado de documentos, embeddings vectoriales e inferencia LLM local para proporcionar respuestas precisas y contextuales optimizadas para diferentes tipos de documentos.
                            """)
                        
                        with gr.Accordion(" Gestión de documentos", open=False):
                            gr.Markdown("""
                            ### Carga de documentos
                            
                            **Formatos soportados:**
                            
                            **Documentos de Office:**
                            * **PDF**: PDFs estándar, PDFs escaneados con OCR, y PDFs impresos desde web
                            * **DOCX**: Documentos de Microsoft Word
                            * **XLSX**: Hojas de cálculo de Microsoft Excel
                            * **PPTX**: Presentaciones de PowerPoint
                            
                            **Texto y Web:**
                            * **TXT**: Archivos de texto plano
                            * **MD**: Archivos Markdown
                            * **HTML/XHTML**: Páginas web y documentos estructurados
                            * **CSV**: Archivos de valores separados por comas
                            
                            **Imágenes (con OCR):**
                            * **PNG, JPEG, TIFF, BMP, WEBP**: Documentos escaneados e imágenes con extracción automática de texto
                            
                            **Formatos especializados:**
                            * **AsciiDoc**: Documentación técnica
                            * **XML**: Datos y documentos estructurados
                            
                            **Opciones de carga:**
                            * **Cargar y procesar**: Cargar y procesar inmediatamente los documentos para consultas
                            * **Solo cargar**: Cargar documentos sin procesarlos (útil para operaciones por lotes)
                            * **Procesar existentes**: Procesar documentos ya en el directorio uploads
                            * **Procesar solo cargados**: Procesar solo documentos que aún no han sido procesados
                            """)
                        
                        with gr.Accordion("🔍 Métodos de consulta", open=False):
                            gr.Markdown("""
                            ### Entender los métodos de consulta
                            
                            El sistema ofrece dos métodos de consulta distintos optimizados para diferentes tipos de documentos:
                            
                            #### **Método Estándar (solo LEANN)**
                            
                            **Ideal para:** Documentos generales (PDFs, documentos Word, archivos de texto, imágenes)
                            
                            **Características:**
                            * Utiliza el almacén vectorial LEANN para todos los tipos de documentos
                            * Chunking basado en oraciones para documentos de texto
                            * Procesamiento OCR para imágenes y documentos escaneados
                            * Procesamiento consistente para todos los formatos soportados
                            
                            **Cuándo usar:**
                            * Búsqueda y análisis de documentos generales
                            * Colecciones de documentos mixtas
                            * Cuando quieres procesamiento consistente en todos los documentos
                            
                            #### **Método Específico Excel (solo LlamaIndex) - EXPERIMENTAL**
                            
                            **Ideal para:** Archivos Excel y datos de hojas de cálculo
                            
                            **Características:**
                            * Utiliza LlamaIndex solo para archivos Excel (EXPERIMENTAL)
                            * Chunking avanzado de Excel (basado en filas, consciente de columnas)
                            * Mejor precisión para consultas de hojas de cálculo
                            * Preserva la estructura y relaciones de Excel
                            
                            **Cuándo usar:**
                            * Trabajar principalmente con archivos Excel
                            * Necesitar extracción precisa de datos de hojas de cálculo
                            * Registros de empleados, inventarios, presupuestos
                            * Cuando necesitas extracción precisa de datos Excel
                            
                            **⚠️ Nota importante:** Esta funcionalidad es experimental y puede tener limitaciones o comportamientos inesperados. Usa con precaución en entornos de producción.
                            
                            #### **Impacto de parámetros por método**
                            
                            | Parámetro | Método Estándar | Método Específico Excel |
                            |-----------|-----------------|-------------------------|
                            | Chunks a recuperar |  Usado (1-40) |  Ignorado (procesa todos los datos) |
                            | Temperatura |  Usado |  Usado |
                            | Max Tokens |  Usado |  Usado |
                            | Selección de modelo |  Usado |  Usado |
                            """)
                        
                        with gr.Accordion("⚙️ Parámetros LLM explicados", open=False):
                            gr.Markdown("""
                            ### Entender los parámetros de consulta
                            
                            #### **Selección del modelo LLM**
                            
                            **Tipos de modelos (ejemplos):**
                            * **Modelos pequeños (3B parámetros)**: Rápidos, buenos para consultas generales, menor uso de recursos
                            * **Modelos medianos (7B parámetros)**: Mejor calidad, respuestas más detalladas, uso moderado de recursos
                            * **Modelos grandes (13B+ parámetros)**: Alta calidad, respuestas muy detalladas, mayor uso de recursos
                            
                            **Nota**: Los modelos disponibles dependen de lo que tengas instalado en Ollama. Usa el botón "Actualizar" para ver tus modelos actuales.
                            
                            #### **Parámetros de búsqueda**
                            
                            **Chunks a recuperar (1-40, Por defecto: 20)**
                            * **Valores bajos (5-10)**: Respuestas más rápidas, respuestas más enfocadas
                            * **Valores altos (20-40)**: Respuestas más completas, respuestas más lentas
                            * **Impacto**: Más chunks = más contexto = respuestas más largas y detalladas
                            
                            **Temperatura (0.1-1.0, Por defecto: 0.7)**
                            * **0.1-0.3**: Respuestas muy enfocadas, deterministas
                            * **0.4-0.6**: Creatividad y precisión equilibradas
                            * **0.7-0.9**: Más creativo, respuestas variadas
                            * **1.0**: Creatividad máxima, puede ser menos preciso
                            * **Impacto**: Temperatura más alta = más creativo pero potencialmente menos preciso
                            
                            **Max Tokens (100-4000, Por defecto: 2048)**
                            * **100-500**: Respuestas cortas y concisas
                            * **1000-2000**: Respuestas de longitud estándar
                            * **3000-4000**: Respuestas muy detalladas y completas
                            * **Impacto**: Valores más altos = respuestas más largas, valores más bajos = respuestas más cortas
                            
                            #### **Combinaciones de parámetros**
                            
                            **Para hechos rápidos:**
                            * Chunks: 5-10, Temperatura: 0.3, Max Tokens: 500
                            
                            **Para análisis detallado:**
                            * Chunks: 20-30, Temperatura: 0.7, Max Tokens: 3000
                            
                            **Para exploración creativa:**
                            * Chunks: 15-25, Temperatura: 0.9, Max Tokens: 2500
                            
                            **Para investigación académica:**
                            * Chunks: 30-40, Temperatura: 0.5, Max Tokens: 4000
                            """)
                        
                        with gr.Accordion("💡 Mejores prácticas", open=False):
                            gr.Markdown("""
                            ### Obtener los mejores resultados
                            
                            #### **Optimización de consultas**
                            * **Comienza con preguntas simples**: Construye la comprensión gradualmente
                            * **Usa términos específicos**: Incluye palabras clave relevantes de tus documentos
                            * **Haz una pregunta a la vez**: Evita preguntas complejas de múltiples partes
                            * **Refina basándote en resultados**: Usa respuestas iniciales para hacer preguntas de seguimiento más específicas
                            
                            #### **Combinaciones de parámetros**
                            
                            **Para hechos rápidos:**
                            * Chunks: 5-10, Temperatura: 0.3, Max Tokens: 500
                            
                            **Para análisis detallado:**
                            * Chunks: 20-30, Temperatura: 0.7, Max Tokens: 3000
                            
                            **Para exploración creativa:**
                            * Chunks: 15-25, Temperatura: 0.9, Max Tokens: 2500
                            
                            **Para investigación académica:**
                            * Chunks: 30-40, Temperatura: 0.5, Max Tokens: 4000
                            """)
                        
                        with gr.Accordion("🔍 Interfaz de consulta", open=False):
                            gr.Markdown("""
                            ### Hacer preguntas
                            
                            **Cómo consultar:**
                            1. **Ingresa tu pregunta** en lenguaje natural (español o inglés)
                            2. **Elige tu método de consulta** (Estándar o Específico Excel)
                            3. **Selecciona tu modelo LLM preferido** del menú desplegable
                            4. **Ajusta los parámetros de búsqueda** si es necesario
                            5. **Haz clic en "Query Documents"** y espera la respuesta
                            
                            **Consejos para consultas:**
                            * **Sé específico**: "¿Cuál es el argumento principal del documento sobre el cambio climático?" vs "¿De qué se trata esto?"
                            * **Usa contexto**: "Según el autor, ¿cuáles son los beneficios de las energías renovables?"
                            * **Haz preguntas de seguimiento**: "¿Puedes elaborar sobre la metodología mencionada en la respuesta anterior?"
                            * **Consultas multi-documento**: "Compara los enfoques discutidos en el Documento A y el Documento B"
                            
                            **Soporte de idiomas:**
                            * **Consultas en español** funcionan mejor con documentos en español
                            * **Consultas en inglés** funcionan mejor con documentos en inglés
                            * **Consultas entre idiomas** son compatibles pero pueden tener precisión reducida
                            """)
                        
                        with gr.Accordion("⚙️ Métodos de consulta", open=False):
                            gr.Markdown("""
                            ### Entender los métodos de consulta
                            
                            El sistema ofrece dos métodos de consulta distintos optimizados para diferentes tipos de documentos:
                            
                            #### **Método Estándar (solo LEANN)**
                            
                            **Ideal para:** Documentos generales (PDFs, documentos Word, archivos de texto, etc.)
                            
                            **Características:**
                            * Utiliza el almacén vectorial LEANN para todos los tipos de documentos
                            * Procesamiento rápido y eficiente
                            * Bueno para consultas basadas en texto
                            * Procesa todos los tipos de documentos de manera uniforme
                            
                            **Cuándo usar:**
                            * Consultar PDFs, documentos Word, archivos de texto
                            * Investigación y análisis general
                            * Cuando quieres procesamiento consistente en todos los documentos
                            
                            #### **Método Específico Excel (solo LlamaIndex)**
                            
                            **Ideal para:** Archivos Excel y datos de hojas de cálculo
                            
                            **Características:**
                            * Utiliza LlamaIndex solo para archivos Excel
                            * Fragmentación Excel avanzada (basada en filas, consciente de columnas)
                            * Mejor precisión para consultas de hojas de cálculo
                            * Procesa TODOS los archivos Excel y TODOS sus datos
                            * Ignora el parámetro "Chunks a recuperar" (procesa todo)
                            
                            **Cuándo usar:**
                            * Consultar archivos Excel y hojas de cálculo
                            * Análisis de datos financieros
                            * Registros de empleados, inventario, presupuestos
                            * Cuando necesitas extracción precisa de datos Excel
                            
                            #### **Impacto de parámetros por método**
                            
                            **Método Estándar:**
                            *  **Chunks a recuperar**: Controla cuántos fragmentos de documentos buscar
                            *  **Temperatura**: Controla la creatividad de la respuesta
                            *  **Max Tokens**: Controla la longitud de la respuesta
                            *  **Modelo**: Controla qué LLM usar
                            
                            **Método Específico Excel:**
                            *  **Chunks a recuperar**: Ignorado (procesa TODOS los datos Excel)
                            *  **Temperatura**: Controla la creatividad de la respuesta
                            *  **Max Tokens**: Controla la longitud de la respuesta
                            *  **Modelo**: Controla qué LLM usar
                            
                            #### **Elegir el método correcto**
                            
                            **Usa Estándar cuando:**
                            * Tienes tipos de documentos mixtos
                            * Quieres consultas rápidas y generales
                            * Trabajas con PDFs, documentos Word o archivos de texto
                            
                            **Usa Específico Excel cuando:**
                            * Trabajas principalmente con archivos Excel
                            * Necesitas extracción precisa de datos de hojas de cálculo
                            * Quieres la consulta Excel más precisa
                            * Haces análisis financiero o de datos
                            """)
                        
                        with gr.Accordion("🚀 Características avanzadas", open=False):
                            gr.Markdown("""
                            ### Uso avanzado
                            
                            #### **Consultas multi-documento**
                            * **Comparación entre documentos**: "Compara las metodologías del Documento A y el Documento B"
                            * **Consultas de síntesis**: "Resume los temas principales a través de todos los documentos cargados"
                            * **Detección de contradicciones**: "¿Hay contradicciones entre estos documentos?"
                            
                            #### **Flujos de trabajo de investigación**
                            1. **Carga artículos de investigación** en tu campo
                            2. **Pide resúmenes** de los hallazgos clave
                            3. **Solicita comparaciones** entre diferentes enfoques
                            4. **Genera preguntas de investigación** basadas en brechas en la literatura
                            
                            #### **Análisis de contenido**
                            * **Análisis de sentimientos**: "¿Cuál es el tono general de este documento?"
                            * **Extracción de conceptos clave**: "¿Cuáles son los conceptos principales discutidos?"
                            * **Creación de cronología**: "Crea una cronología de eventos mencionados en estos documentos"
                            """)
                        
                        with gr.Accordion("ℹ️ Información del sistema", open=False):
                            gr.Markdown("""
                            ### Detalles técnicos
                            
                            **Procesamiento de documentos:**
                            * **Estrategia de fragmentación**: Basada en oraciones con conciencia de párrafos
                            * **Tamaño de fragmentos**: 400 caracteres con superposición de 100 caracteres
                            * **Extracción de texto**: Docling (principal) con respaldo pypdf para PDFs
                            * **Formatos soportados**: PDF, DOCX, XLSX, PPTX, TXT, MD, HTML, XHTML, CSV, PNG, JPEG, TIFF, BMP, WEBP, AsciiDoc, XML
                            * **Dependencias de procesamiento**: Docling para formatos complejos, pypdf para PDFs impresos desde web, lectura directa para archivos simples
                            * **Modelo de embedding**: nomic-embed-text-v2-moe (768 dimensiones)
                            * **Base de datos vectorial**: LEANN con almacenamiento ultra-eficiente (97% de ahorro de espacio)
                            
                            **Integración LLM:**
                            * **Procesamiento local**: Todas las consultas procesadas localmente vía Ollama
                            * **Soporte de modelos**: Cualquier modelo compatible con Ollama
                            * **Aceleración GPU**: Soporte MPS para Apple Silicon
                            * **Tiempo de respuesta**: 15-30 segundos dependiendo del modelo y parámetros
                            
                            **Entorno de pruebas:**
                            * **Hardware**: Mac mini (Identificador de modelo: Mac16,10)
                            * **Procesador**: Chip Apple M4
                            * **Memoria**: 16GB de memoria unificada
                            * **Idiomas probados**: Inglés, francés, español, alemán
                            * **Rendimiento**: Excelente en todos los idiomas soportados
                            * **Arquitectura**: Procesamiento local para rendimiento óptimo y privacidad completa de los datos
                            """)
                        
                        with gr.Accordion("🔧 Instalación de modelos", open=False):
                            gr.Markdown("""
                            ### Instalación de modelos LLM
                            
                            **Para usar el sistema, necesitas instalar modelos en Ollama:**
                            
                            #### **Opciones de modelos populares:**
                            
                            **Para uso general:**
                            ```bash
                            ollama pull llama3.2:3b    # Rápido, buena calidad
                            ollama pull llama3.2:7b    # Mejor calidad
                            ```
                            
                            **Para alta calidad:**
                            ```bash
                            ollama pull llama3.2:13b   # Alta calidad
                            ollama pull llama3.2:70b   # Mejor calidad (requiere más recursos)
                            ```
                            
                            **Para contenido en español:**
                            ```bash
                            ollama pull mistral:7b     # Buen soporte en español
                            ollama pull mixtral:8x7b   # Excelente multilingüe
                            ```
                            
                            #### **Pasos de instalación:**
                            1. **Instala Ollama** desde https://ollama.ai
                            2. **Descarga un modelo** usando los comandos de arriba
                            3. **Actualiza la lista de modelos** en la interfaz
                            4. **Selecciona tu modelo** del menú desplegable
                            
                            **Nota**: La descarga de modelos puede tomar varios minutos dependiendo del tamaño y velocidad de internet.
                            """)
                        
                        with gr.Accordion("⚠️ Limitaciones", open=False):
                            gr.Markdown("""
                            ### Limitaciones actuales y probables
                            
                            #### **Limitaciones del procesamiento de documentos**
                            * **PDFs escaneados**: Completamente soportados con procesamiento OCR automático
                            * **Diseños complejos**: Tablas, diseños multi-columna pueden no preservarse perfectamente
                            * **Archivos grandes**: Documentos muy grandes (>100MB) pueden causar retrasos en el procesamiento
                            * **Contenido de imágenes**: El texto dentro de imágenes se extrae automáticamente con OCR
                            
                            #### **Limitaciones de idioma y modelos**
                            * **Sensibilidad de idioma**: Consultas entre idiomas pueden tener precisión reducida
                            * **Dependencia del modelo**: El rendimiento varía significativamente entre diferentes modelos LLM
                            * **Ventana de contexto**: Documentos muy largos pueden exceder los límites de contexto del modelo
                            * **Dominios especializados**: Contenido técnico o especializado puede requerir modelos especializados
                            
                            #### **Limitaciones del sistema**
                            * **Procesamiento local**: Todo el procesamiento ocurre localmente, requiriendo recursos de hardware adecuados
                            * **Uso de memoria**: Grandes colecciones de documentos requieren RAM significativa
                            * **Tiempo de procesamiento**: Consultas complejas con gran contexto pueden tomar 30+ segundos
                            * **Usuarios concurrentes**: Sistema diseñado para usuario único o equipo pequeño
                            
                            """)
                        
                        with gr.Accordion("🔧 Gestión del sistema", open=False):
                            gr.Markdown("""
                            ### Gestionar tu sistema
                            
                            El sistema proporciona herramientas de gestión completas para mantener tus colecciones de documentos e índices.
                            
                            #### **Gestión de documentos**
                            
                            **Ver documentos:**
                            * **Lista de documentos**: Ver todos los documentos cargados y procesados
                            * **Información de estado**: Ver el estado de procesamiento, conteo de chunks y tamaños de archivos
                            * **Eliminar documentos**: Remover documentos individuales del sistema
                            
                            **Opciones de carga:**
                            * **Cargar y procesar**: Procesar inmediatamente los documentos para consultas
                            * **Solo cargar**: Almacenar documentos sin procesarlos
                            * **Procesar existentes**: Procesar documentos ya en uploads
                            * **Procesar solo cargados**: Procesar solo documentos no procesados
                            
                            #### **Mantenimiento del sistema**
                            
                            **Operaciones de reinicio (solo índice):**
                            * **Reiniciar índice LEANN**: Reconstruye el índice LEANN, preserva todos los datos
                            * **Reiniciar LlamaIndex Excel (EXPERIMENTAL)**: Reconstruye el índice LlamaIndex, preserva todos los datos
                            
                            **Operaciones de reconstrucción (rápido, sin reprocesamiento):**
                            * **Reconstruir índice LEANN**: Reconstruye desde documentos procesados existentes
                            * **Reconstruir LlamaIndex Excel (EXPERIMENTAL)**: Reconstruye desde archivos Excel procesados existentes
                            
                            **Operaciones de limpieza (índice + datos):**
                            * **Limpiar documentos LEANN**: Elimina índice LEANN + documentos no-Excel procesados
                            * **Limpiar LlamaIndex Excel (EXPERIMENTAL)**: Elimina índice LlamaIndex + archivos Excel procesados
                            * **Limpiar todo**: Elimina todos los índices y todos los datos
                            
                            #### **Gestión de almacenes vectoriales**
                            
                            **Almacén vectorial LEANN:**
                            * **Monitoreo de estado**: Ver el estado del índice, conteo de documentos y tamaño
                            * **Información del índice**: Ver configuración y métricas de rendimiento
                            * **Actualizar estado**: Actualizar información de estado en tiempo real
                            
                            **Almacén LlamaIndex Excel (EXPERIMENTAL):**
                            * **Monitoreo de estado**: Ver el estado del índice Excel y conteo de archivos
                            * **Seguimiento de archivos Excel**: Monitorear archivos Excel cargados y procesados
                            * **Actualizar estado**: Actualizar información del índice Excel
                            * **⚠️ Nota**: El procesamiento de Excel con LlamaIndex es experimental
                            
                            #### **Información del sistema**
                            
                            **Gestión de modelos:**
                            * **Modelos disponibles**: Ver todos los modelos Ollama instalados
                            * **Selección de modelo**: Elegir el mejor modelo para tus necesidades
                            * **Actualizar modelos**: Actualizar la lista de modelos desde Ollama
                            
                            **Estado de procesamiento:**
                            * **Monitoreo en tiempo real**: Ver el estado de procesamiento actual
                            * **Información de cola**: Ver tareas de procesamiento pendientes
                            * **Métricas de rendimiento**: Monitorear el rendimiento del sistema
                            
                            #### **Mejores prácticas para la gestión del sistema**
                            
                            **Mantenimiento regular:**
                            * Usa operaciones "Reconstruir" para refresco rápido del índice
                            * Usa operaciones "Reiniciar" cuando los índices se corrompan
                            * Usa operaciones "Limpiar" solo cuando quieras eliminar datos
                            
                            **Solución de problemas:**
                            * Si las consultas devuelven malos resultados, intenta reconstruir el índice relevante
                            * Si el procesamiento falla, revisa los logs e intenta reprocesar
                            * Si ocurren problemas de memoria, limpia datos no utilizados y reconstruye
                            
                            **Seguridad de datos:**
                            * Siempre respalda documentos importantes antes de operaciones mayores
                            * Usa "Reiniciar" en lugar de "Limpiar" cuando sea posible
                            * Prueba operaciones en pequeños conjuntos de datos primero
                            """)
                    
                    # German Guide
                    with gr.Group(visible=False) as german_guide:
                        with gr.Accordion("🎯 Systemübersicht", open=False):
                            gr.Markdown("""
                            **Myr-Ag** ist ein leistungsstarkes RAG (Retrieval-Augmented Generation) System, das Ihnen ermöglicht:
                            
                            * **Dokumente hochzuladen und zu verarbeiten** in mehreren Formaten (PDF, DOCX, XLSX, PPTX, TXT, MD, HTML, XHTML, CSV, PNG, JPEG, TIFF, BMP, WEBP, AsciiDoc, XML)
                            * **Fragen zu Ihren Dokumenten** in natürlicher Sprache zu stellen
                            * **Intelligente Antworten** zu erhalten, die von lokalen LLM-Modellen angetrieben werden
                            * **Über mehrere Dokumente zu suchen** mit semantischem Verständnis
                            
                            **Duale Verarbeitungsarchitektur:**
                            * **Standard-Methode**: Nutzt LEANN Vektorspeicher für allgemeine Dokumente (PDFs, Word-Dokumente, Textdateien)
                            * **Excel-spezifische Methode**: Nutzt LlamaIndex für Excel-Dateien mit erweiterter Tabellenkalkulations-Verarbeitung (⚠️ EXPERIMENTELL)
                            
                            Das System verwendet fortschrittliche Dokumentenverarbeitung, Vektor-Einbettungen und lokale LLM-Inferenz, um präzise und kontextuelle Antworten zu liefern, die für verschiedene Dokumententypen optimiert sind.
                            """)
                        
                        with gr.Accordion(" Dokumentenverwaltung", open=False):
                            gr.Markdown("""
                            ### Dokumenten-Upload
                            
                            **Unterstützte Formate:**
                            
                            **Office-Dokumente:**
                            * **PDF** : Standard-PDFs, gescannte PDFs mit OCR und Web-gedruckte PDFs
                            * **DOCX** : Microsoft Word-Dokumente
                            * **XLSX** : Microsoft Excel-Tabellenkalkulationen (⚠️ EXPERIMENTELL)
                            * **PPTX** : Microsoft PowerPoint-Präsentationen
                            
                            **Web-Inhalte:**
                            * **HTML/XHTML** : Webseiten und HTML-Dokumente
                            
                            **Textformate:**
                            * **TXT** : Einfache Textdateien
                            * **MD** : Markdown-Dokumente
                            * **AsciiDoc** : AsciiDoc-Dokumentationsformat
                            
                            **Datenformate:**
                            * **CSV** : Komma-getrennte Werte-Dateien
                            
                            **Bildformate (mit OCR):**
                            * **PNG, JPEG, TIFF, BMP, WEBP** : Bilder mit automatischer Textextraktion
                            
                            **Spezialisierte Formate:**
                            * **XML** : Erweiterbare Markup-Sprache
                            * **USPTO XML** : USPTO-Patentdokumente
                            * **JATS XML** : Journal Article Tag Suite (wissenschaftliche Artikel)
                            """)
                        
                        with gr.Accordion("🔧 Systemverwaltung", open=False):
                            gr.Markdown(get_german_guide())
                        
                        with gr.Accordion("🔍 Abfragemethoden", open=False):
                            gr.Markdown("""
                            ### Abfragemethoden verstehen
                            
                            Das System bietet zwei verschiedene Abfragemethoden, die für verschiedene Dokumententypen optimiert sind:
                            
                            #### **Standard-Methode (nur LEANN)**
                            
                            **Ideal für:** Allgemeine Dokumente (PDFs, Word-Dokumente, Textdateien, Bilder)
                            
                            **Funktionen:**
                            * Verwendet LEANN-Vektorspeicher für alle Dokumententypen
                            * Satz-basierte Chunking für Textdokumente
                            * OCR-Verarbeitung für Bilder und gescannte Dokumente
                            * Konsistente Verarbeitung für alle unterstützten Formate
                            
                            **Wann verwenden:**
                            * Allgemeine Dokumentsuche und -analyse
                            * Gemischte Dokumentensammlungen
                            * Wenn Sie konsistente Verarbeitung für alle Dokumente wünschen
                            
                            #### **Excel-spezifische Methode (nur LlamaIndex) - EXPERIMENTELL**
                            
                            **Ideal für:** Excel-Dateien und Tabellenkalkulationsdaten
                            
                            **Funktionen:**
                            * Verwendet LlamaIndex nur für Excel-Dateien (EXPERIMENTELL)
                            * Erweiterte Excel-Chunking (zeilenbasiert, spaltenbewusst)
                            * Bessere Genauigkeit für Tabellenkalkulationsabfragen
                            * Erhält Excel-Struktur und -Beziehungen
                            
                            **Wann verwenden:**
                            * Hauptsächlich mit Excel-Dateien arbeiten
                            * Präzise Tabellenkalkulationsdatenextraktion benötigen
                            * Mitarbeiterdaten, Inventar, Budgets
                            * Wenn Sie präzise Excel-Datenextraktion benötigen
                            
                            **⚠️ Wichtiger Hinweis:** Diese Funktion ist experimentell und kann Einschränkungen oder unerwartetes Verhalten haben. Verwenden Sie mit Vorsicht in Produktionsumgebungen.
                            
                            #### **Parameterauswirkung nach Methode**
                            
                            | Parameter | Standard-Methode | Excel-spezifische Methode |
                            |-----------|------------------|---------------------------|
                            | Zu abrufende Chunks |  Verwendet (1-40) |  Ignoriert (verarbeitet alle Daten) |
                            | Temperatur |  Verwendet |  Verwendet |
                            | Max Tokens |  Verwendet |  Verwendet |
                            | Modellauswahl |  Verwendet |  Verwendet |
                            """)
                        
                        with gr.Accordion("⚙️ LLM-Parameter erklärt", open=False):
                            gr.Markdown("""
                            ### Abfrageparameter verstehen
                            
                            #### **LLM-Modellauswahl**
                            
                            **Modelltypen (Beispiele):**
                            * **Kleine Modelle (3B Parameter)**: Schnell, gut für allgemeine Abfragen, geringerer Ressourcenverbrauch
                            * **Mittlere Modelle (7B Parameter)**: Bessere Qualität, detailliertere Antworten, moderater Ressourcenverbrauch
                            * **Große Modelle (13B+ Parameter)**: Hohe Qualität, sehr detaillierte Antworten, höherer Ressourcenverbrauch
                            
                            **Hinweis**: Verfügbare Modelle hängen davon ab, was Sie in Ollama installiert haben. Verwenden Sie die "Aktualisieren"-Schaltfläche, um Ihre aktuellen Modelle zu sehen.
                            
                            #### **Suchparameter**
                            
                            **Zu abrufende Chunks (1-40, Standard: 20)**
                            * **Niedrige Werte (5-10)**: Schnellere Antworten, fokussiertere Antworten
                            * **Hohe Werte (20-40)**: Umfassendere Antworten, langsamere Antworten
                            * **Auswirkung**: Mehr Chunks = mehr Kontext = längere, detailliertere Antworten
                            
                            **Temperatur (0.1-1.0, Standard: 0.7)**
                            * **0.1-0.3**: Sehr fokussierte, deterministische Antworten
                            * **0.4-0.6**: Ausgewogene Kreativität und Genauigkeit
                            * **0.7-0.9**: Kreativer, vielfältigere Antworten
                            * **1.0**: Maximale Kreativität, kann weniger genau sein
                            * **Auswirkung**: Höhere Temperatur = kreativer aber potenziell weniger genau
                            
                            **Max Tokens (100-4000, Standard: 2048)**
                            * **100-500**: Kurze, prägnante Antworten
                            * **1000-2000**: Standardlänge Antworten
                            * **3000-4000**: Sehr detaillierte, umfassende Antworten
                            * **Auswirkung**: Höhere Werte = längere Antworten, niedrigere Werte = kürzere Antworten
                            
                            #### **Parameterkombinationen**
                            
                            **Für schnelle Fakten:**
                            * Chunks: 5-10, Temperatur: 0.3, Max Tokens: 500
                            
                            **Für detaillierte Analyse:**
                            * Chunks: 20-30, Temperatur: 0.7, Max Tokens: 3000
                            
                            **Für kreative Erkundung:**
                            * Chunks: 15-25, Temperatur: 0.9, Max Tokens: 2500
                            
                            **Für akademische Forschung:**
                            * Chunks: 30-40, Temperatur: 0.5, Max Tokens: 4000
                            """)
                        
                        with gr.Accordion("💡 Beste Praktiken", open=False):
                            gr.Markdown("""
                            ### Die besten Ergebnisse erzielen
                            
                            #### **Abfrage-Optimierung**
                            * **Beginnen Sie mit einfachen Fragen**: Bauen Sie das Verständnis schrittweise auf
                            * **Verwenden Sie spezifische Begriffe**: Fügen Sie relevante Schlüsselwörter aus Ihren Dokumenten ein
                            * **Stellen Sie eine Frage zur Zeit**: Vermeiden Sie komplexe mehrteilige Fragen
                            * **Verfeinern Sie basierend auf Ergebnissen**: Verwenden Sie erste Antworten für spezifischere Nachfragen
                            
                            #### **Parameterkombinationen**
                            
                            **Für schnelle Fakten:**
                            * Chunks: 5-10, Temperatur: 0.3, Max Tokens: 500
                            
                            **Für detaillierte Analyse:**
                            * Chunks: 20-30, Temperatur: 0.7, Max Tokens: 3000
                            
                            **Für kreative Erkundung:**
                            * Chunks: 15-25, Temperatur: 0.9, Max Tokens: 2500
                            
                            **Für akademische Forschung:**
                            * Chunks: 30-40, Temperatur: 0.5, Max Tokens: 4000
                            """)
                        
                        with gr.Accordion("🔍 Abfrage-Interface", open=False):
                            gr.Markdown("""
                            ### Fragen stellen
                            
                            **Wie man abfragt:**
                            1. **Geben Sie Ihre Frage** in natürlicher Sprache ein (Deutsch oder Englisch)
                            2. **Wählen Sie Ihre Abfragemethode** (Standard oder Excel-spezifisch)
                            3. **Wählen Sie Ihr bevorzugtes LLM-Modell** aus dem Dropdown-Menü
                            4. **Passen Sie die Suchparameter** bei Bedarf an
                            5. **Klicken Sie auf "Query Documents"** und warten Sie auf die Antwort
                            
                            **Abfrage-Tipps:**
                            * **Seien Sie spezifisch**: "Was ist das Hauptargument des Dokuments über den Klimawandel?" vs "Worum geht es hier?"
                            * **Verwenden Sie Kontext**: "Laut dem Autor, was sind die Vorteile erneuerbarer Energien?"
                            * **Stellen Sie Nachfragen**: "Können Sie die in der vorherigen Antwort erwähnte Methodik näher erläutern?"
                            * **Multi-Dokument-Abfragen**: "Vergleichen Sie die in Dokument A und Dokument B diskutierten Ansätze"
                            
                            **Sprachunterstützung:**
                            * **Deutsche Abfragen** funktionieren am besten mit deutschen Dokumenten
                            * **Englische Abfragen** funktionieren am besten mit englischen Dokumenten
                            * **Sprachübergreifende Abfragen** werden unterstützt, können aber reduzierte Genauigkeit haben
                            """)
                        
                        with gr.Accordion("⚙️ Abfragemethoden", open=False):
                            gr.Markdown("""
                            ### Abfragemethoden verstehen
                            
                            Das System bietet zwei verschiedene Abfragemethoden, die für verschiedene Dokumententypen optimiert sind:
                            
                            #### **Standard-Methode (nur LEANN)**
                            
                            **Ideal für:** Allgemeine Dokumente (PDFs, Word-Dokumente, Textdateien, etc.)
                            
                            **Eigenschaften:**
                            * Nutzt LEANN Vektorspeicher für alle Dokumententypen
                            * Schnelle und effiziente Verarbeitung
                            * Gut für textbasierte Abfragen
                            * Verarbeitet alle Dokumententypen einheitlich
                            
                            **Wann verwenden:**
                            * Abfragen von PDFs, Word-Dokumenten, Textdateien
                            * Allgemeine Recherche und Analyse
                            * Wenn Sie einheitliche Verarbeitung aller Dokumente wünschen
                            
                            #### **Excel-spezifische Methode (nur LlamaIndex)**
                            
                            **Ideal für:** Excel-Dateien und Tabellenkalkulationsdaten
                            
                            **Eigenschaften:**
                            * Nutzt LlamaIndex nur für Excel-Dateien
                            * Erweiterte Excel-Fragmentierung (zeilenbasiert, spaltenbewusst)
                            * Bessere Genauigkeit für Tabellenkalkulationsabfragen
                            * Verarbeitet ALLE Excel-Dateien und ALLE ihre Daten
                            * Ignoriert den Parameter "Chunks abrufen" (verarbeitet alles)
                            
                            **Wann verwenden:**
                            * Abfragen von Excel-Dateien und Tabellenkalkulationen
                            * Finanzdatenanalyse
                            * Mitarbeiterakten, Inventar, Budgets
                            * Wenn Sie präzise Excel-Datenextraktion benötigen
                            
                            #### **Parameterauswirkung nach Methode**
                            
                            **Standard-Methode:**
                            *  **Chunks abrufen**: Steuert, wie viele Dokumentenfragmente durchsucht werden
                            *  **Temperatur**: Steuert die Kreativität der Antwort
                            *  **Max Tokens**: Steuert die Länge der Antwort
                            *  **Modell**: Steuert, welches LLM verwendet wird
                            
                            **Excel-spezifische Methode:**
                            *  **Chunks abrufen**: Ignoriert (verarbeitet ALLE Excel-Daten)
                            *  **Temperatur**: Steuert die Kreativität der Antwort
                            *  **Max Tokens**: Steuert die Länge der Antwort
                            *  **Modell**: Steuert, welches LLM verwendet wird
                            
                            #### **Die richtige Methode wählen**
                            
                            **Verwenden Sie Standard wenn:**
                            * Sie gemischte Dokumententypen haben
                            * Sie schnelle, allgemeine Abfragen wünschen
                            * Sie mit PDFs, Word-Dokumenten oder Textdateien arbeiten
                            
                            **Verwenden Sie Excel-spezifisch wenn:**
                            * Sie hauptsächlich mit Excel-Dateien arbeiten
                            * Sie präzise Tabellenkalkulationsdatenextraktion benötigen
                            * Sie die genaueste Excel-Abfrage wünschen
                            * Sie Finanz- oder Datenanalyse betreiben
                            """)
                        
                        with gr.Accordion("🚀 Erweiterte Funktionen", open=False):
                            gr.Markdown("""
                            ### Erweiterte Nutzung
                            
                            #### **Multi-Dokument-Abfragen**
                            * **Dokumentübergreifende Vergleiche**: "Vergleichen Sie die Methodologien in Dokument A und Dokument B"
                            * **Synthese-Abfragen**: "Fassen Sie die Hauptthemen über alle hochgeladenen Dokumente zusammen"
                            * **Widerspruchserkennung**: "Gibt es Widersprüche zwischen diesen Dokumenten?"
                            
                            #### **Forschungsworkflows**
                            1. **Laden Sie Forschungsartikel** in Ihrem Bereich hoch
                            2. **Bitten Sie um Zusammenfassungen** der wichtigsten Erkenntnisse
                            3. **Fordern Sie Vergleiche** zwischen verschiedenen Ansätzen an
                            4. **Generieren Sie Forschungsfragen** basierend auf Lücken in der Literatur
                            
                            #### **Inhaltsanalyse**
                            * **Sentiment-Analyse**: "Wie ist der allgemeine Ton dieses Dokuments?"
                            * **Schlüsselkonzept-Extraktion**: "Was sind die Hauptkonzepte, die diskutiert werden?"
                            * **Zeitlinien-Erstellung**: "Erstellen Sie eine Zeitlinie der in diesen Dokumenten erwähnten Ereignisse"
                            """)
                        
                        with gr.Accordion("ℹ️ Systeminformationen", open=False):
                            gr.Markdown("""
                            ### Technische Details
                            
                            **Dokumentenverarbeitung:**
                            * **Chunking-Strategie**: Satz-basiert mit Absatz-Bewusstsein
                            * **Chunk-Größe**: 400 Zeichen mit 100-Zeichen-Überlappung
                            * **Textextraktion**: Docling (primär) mit pypdf-Fallback für PDFs
                            * **Unterstützte Formate**: PDF, DOCX, XLSX, PPTX, TXT, MD, HTML, XHTML, CSV, PNG, JPEG, TIFF, BMP, WEBP, AsciiDoc, XML
                            * **Verarbeitungsabhängigkeiten**: Docling für komplexe Formate, pypdf für Web-gedruckte PDFs, direkte Textlesung für einfache Dateien
                            * **Embedding-Modell**: nomic-embed-text-v2-moe (768 Dimensionen)
                            * **Vektor-Datenbank**: LEANN mit ultra-effizienter Speicherung (97% Platzersparnis)
                            
                            **LLM-Integration:**
                            * **Lokale Verarbeitung**: Alle Abfragen lokal über Ollama verarbeitet
                            * **Modell-Unterstützung**: Jedes Ollama-kompatible Modell
                            * **GPU-Beschleunigung**: MPS-Unterstützung für Apple Silicon
                            * **Antwortzeit**: 15-30 Sekunden je nach Modell und Parametern
                            
                            **Testumgebung:**
                            * **Hardware**: Mac mini (Modell-Identifikator: Mac16,10)
                            * **Prozessor**: Apple M4-Chip
                            * **Speicher**: 16GB einheitlicher Speicher
                            * **Getestete Sprachen**: Englisch, Französisch, Spanisch, Deutsch
                            * **Leistung**: Hervorragend in allen unterstützten Sprachen
                            * **Architektur**: Lokale Verarbeitung für optimale Leistung und vollständigen Datenschutz
                            """)
                        
                        with gr.Accordion("🔧 Modell-Installation", open=False):
                            gr.Markdown("""
                            ### LLM-Modell-Installation
                            
                            **Um das System zu verwenden, müssen Sie Modelle in Ollama installieren:**
                            
                            #### **Beliebte Modell-Optionen:**
                            
                            **Für allgemeine Nutzung:**
                            ```bash
                            ollama pull llama3.2:3b    # Schnell, gute Qualität
                            ollama pull llama3.2:7b    # Bessere Qualität
                            ```
                            
                            **Für hohe Qualität:**
                            ```bash
                            ollama pull llama3.2:13b   # Hohe Qualität
                            ollama pull llama3.2:70b   # Beste Qualität (erfordert mehr Ressourcen)
                            ```
                            
                            **Für deutschen Inhalt:**
                            ```bash
                            ollama pull mistral:7b     # Gute deutsche Unterstützung
                            ollama pull mixtral:8x7b   # Ausgezeichnete mehrsprachige Unterstützung
                            ```
                            
                            #### **Installationsschritte:**
                            1. **Installieren Sie Ollama** von https://ollama.ai
                            2. **Laden Sie ein Modell herunter** mit den obigen Befehlen
                            3. **Aktualisieren Sie die Modellliste** in der Benutzeroberfläche
                            4. **Wählen Sie Ihr Modell** aus dem Dropdown-Menü
                            
                            **Hinweis**: Der Modell-Download kann je nach Größe und Internetgeschwindigkeit mehrere Minuten dauern.
                            """)
                        
                        with gr.Accordion("⚠️ Einschränkungen", open=False):
                            gr.Markdown("""
                            ### Aktuelle und wahrscheinliche Einschränkungen
                            
                            #### **Dokumentenverarbeitungs-Einschränkungen**
                            * **Gescannte PDFs**: Vollständig unterstützt mit automatischer OCR-Verarbeitung
                            * **Komplexe Layouts**: Tabellen, mehrspaltige Layouts können nicht perfekt erhalten bleiben
                            * **Große Dateien**: Sehr große Dokumente (>100MB) können Verarbeitungsverzögerungen verursachen
                            * **Bildinhalt**: Text in Bildern wird automatisch mit OCR extrahiert
                            
                            #### **Sprach- und Modell-Einschränkungen**
                            * **Sprachsensitivität**: Sprachübergreifende Abfragen können reduzierte Genauigkeit haben
                            * **Modellabhängigkeit**: Die Leistung variiert erheblich zwischen verschiedenen LLM-Modellen
                            * **Kontextfenster**: Sehr lange Dokumente können die Kontextgrenzen des Modells überschreiten
                            * **Spezialisierte Domänen**: Technische oder domänenspezifische Inhalte können spezialisierte Modelle erfordern
                            
                            #### **System-Einschränkungen**
                            * **Lokale Verarbeitung**: Alle Verarbeitung erfolgt lokal und erfordert angemessene Hardware-Ressourcen
                            * **Speichernutzung**: Große Dokumentensammlungen erfordern erheblichen RAM
                            * **Verarbeitungszeit**: Komplexe Abfragen mit großem Kontext können 30+ Sekunden dauern
                            * **Gleichzeitige Benutzer**: System für Einzelbenutzer oder kleines Team konzipiert
                            
                            """)
                        
                        with gr.Accordion("🔧 Systemverwaltung", open=False):
                            gr.Markdown("""
                            ### Ihr System verwalten
                            
                            Das System bietet umfassende Verwaltungstools für die Pflege Ihrer Dokumentsammlungen und Indizes.
                            
                            #### **Dokumentenverwaltung**
                            
                            **Dokumente anzeigen:**
                            * **Dokumentenliste**: Alle hochgeladenen und verarbeiteten Dokumente anzeigen
                            * **Statusinformationen**: Verarbeitungsstatus, Chunk-Anzahl und Dateigrößen anzeigen
                            * **Dokumente löschen**: Einzelne Dokumente aus dem System entfernen
                            
                            **Upload-Optionen:**
                            * **Upload und verarbeiten**: Dokumente sofort für Abfragen verarbeiten
                            * **Nur uploaden**: Dokumente ohne Verarbeitung speichern
                            * **Vorhandene verarbeiten**: Dokumente verarbeiten, die bereits in uploads sind
                            * **Nur hochgeladene verarbeiten**: Nur nicht verarbeitete Dokumente verarbeiten
                            
                            #### **Systemwartung**
                            
                            **Reset-Operationen (nur Index):**
                            * **LEANN-Index zurücksetzen**: Rekonstruiert LEANN-Index, bewahrt alle Daten
                            * **LlamaIndex Excel zurücksetzen (EXPERIMENTELL)**: Rekonstruiert LlamaIndex-Index, bewahrt alle Daten
                            
                            **Rebuild-Operationen (schnell, ohne Neuverarbeitung):**
                            * **LEANN-Index neu aufbauen**: Rekonstruiert aus vorhandenen verarbeiteten Dokumenten
                            * **LlamaIndex Excel neu aufbauen (EXPERIMENTELL)**: Rekonstruiert aus vorhandenen Excel-Verarbeitungsdateien
                            
                            **Lösch-Operationen (Index + Daten):**
                            * **LEANN-Dokumente löschen**: Entfernt LEANN-Index + nicht-Excel verarbeitete Dokumente
                            * **LlamaIndex Excel löschen (EXPERIMENTELL)**: Entfernt LlamaIndex-Index + verarbeitete Excel-Dateien
                            * **Alles löschen**: Entfernt alle Indizes und alle Daten
                            
                            #### **Vektorspeicher-Verwaltung**
                            
                            **LEANN-Vektorspeicher:**
                            * **Status-Überwachung**: Index-Status, Dokumentenanzahl und Größe anzeigen
                            * **Index-Informationen**: Konfiguration und Leistungsmetriken anzeigen
                            * **Status aktualisieren**: Statusinformationen in Echtzeit aktualisieren
                            
                            **LlamaIndex Excel-Speicher (EXPERIMENTELL):**
                            * **Status-Überwachung**: Excel-Index-Status und Dateianzahl anzeigen
                            * **Excel-Datei-Tracking**: Hochgeladene und verarbeitete Excel-Dateien überwachen
                            * **Status aktualisieren**: Excel-Index-Informationen aktualisieren
                            * **⚠️ Hinweis**: Excel-Verarbeitung mit LlamaIndex ist experimentell
                            
                            #### **Systeminformationen**
                            
                            **Modellverwaltung:**
                            * **Verfügbare Modelle**: Alle installierten Ollama-Modelle anzeigen
                            * **Modellauswahl**: Bestes Modell für Ihre Bedürfnisse wählen
                            * **Modelle aktualisieren**: Modellliste von Ollama aktualisieren
                            
                            **Verarbeitungsstatus:**
                            * **Echtzeit-Überwachung**: Aktuellen Verarbeitungsstatus anzeigen
                            * **Warteschlangen-Informationen**: Ausstehende Verarbeitungsaufgaben anzeigen
                            * **Leistungsmetriken**: Systemleistung überwachen
                            
                            #### **Best Practices für Systemverwaltung**
                            
                            **Regelmäßige Wartung:**
                            * Verwenden Sie "Rebuild"-Operationen für schnelle Index-Aktualisierung
                            * Verwenden Sie "Reset"-Operationen, wenn Indizes korrupt werden
                            * Verwenden Sie "Lösch"-Operationen nur, wenn Sie Daten entfernen möchten
                            
                            **Fehlerbehebung:**
                            * Wenn Abfragen schlechte Ergebnisse liefern, versuchen Sie den relevanten Index neu aufzubauen
                            * Wenn die Verarbeitung fehlschlägt, überprüfen Sie die Logs und versuchen Sie eine Neuverarbeitung
                            * Bei Speicherproblemen löschen Sie ungenutzte Daten und bauen Sie neu auf
                            
                            **Datensicherheit:**
                            * Sichern Sie immer wichtige Dokumente vor größeren Operationen
                            * Verwenden Sie "Reset" anstelle von "Löschen" wenn möglich
                            * Testen Sie Operationen zuerst an kleinen Datensätzen
                            """)
                    
                    # Language switching logic
                    def switch_language(language):
                        return (
                            gr.update(visible=(language == "English")),
                            gr.update(visible=(language == "Français")),
                            gr.update(visible=(language == "Español")),
                            gr.update(visible=(language == "Deutsch"))
                        )
                    
                    language_selector.change(
                        fn=switch_language,
                        inputs=language_selector,
                        outputs=[english_guide, french_guide, spanish_guide, german_guide]
                    )
            
            # Event handlers
            refresh_status_btn.click(
                fn=self.get_system_info,
                outputs=status_output
            )
            
            refresh_vector_store_btn.click(
                fn=self.get_vector_store_info,
                outputs=vector_store_info
            )
            

            
            refresh_docs_btn.click(
                fn=self.refresh_documents_with_dropdown,
                outputs=[doc_list_output, doc_dropdown]
            )
            
            delete_doc_btn.click(
                fn=self.delete_document_with_dropdown,
                inputs=doc_dropdown,
                outputs=[doc_list_output, doc_dropdown]
            )
            
            # Domain management button connections
            refresh_domain_stats_btn.click(
                fn=self.get_domain_statistics,
                outputs=domain_stats_output
            )
            
            reset_all_domains_btn.click(
                fn=self.reset_all_domain_indexes,
                outputs=domain_stats_output
            )
            
            
            
            
            # Upload with domain selection
            upload_btn.click(
                fn=self.upload_and_process_with_domain,
                inputs=[file_input, upload_domain_selector],
                outputs=[upload_output, upload_output]
            ).then(
                fn=self.refresh_documents_with_dropdown,
                outputs=[doc_list_output, doc_dropdown]
            )
            
            upload_only_btn.click(
                fn=self.upload_only_documents_with_domain,
                inputs=[file_input, upload_domain_selector],
                outputs=[upload_output, doc_list_output, doc_dropdown]
            )
            
            process_existing_btn.click(
                fn=self.process_existing_documents,
                outputs=upload_output
            )
            
            process_uploaded_btn.click(
                fn=self.process_uploaded_only_documents,
                outputs=upload_output
            )
            
            # Function to handle query mode changes
            
            
            # Enhanced query function that handles all modes
            def enhanced_query_documents(question, n_chunks, temperature, max_tokens, model_name, query_mode):
                if query_mode == "standard":
                    return self.query_documents(question, n_chunks, temperature, max_tokens, model_name, False, False, None)
                elif query_mode == "excel":
                    return self.query_documents(question, n_chunks, temperature, max_tokens, model_name, True, False, None)
                elif query_mode == "specialized_auto":
                    return self.query_documents(question, n_chunks, temperature, max_tokens, model_name, False, True, "auto")
                elif query_mode in ["financial", "legal", "medical", "academic"]:
                    return self.query_documents(question, n_chunks, temperature, max_tokens, model_name, False, True, query_mode)
                else:
                    return self.query_documents(question, n_chunks, temperature, max_tokens, model_name, False, False, None)
            
            query_btn.click(
                fn=enhanced_query_documents,
                inputs=[question_input, n_chunks_input, temperature_input, max_tokens_input, model_selector, query_mode],
                outputs=[query_output, query_status]
            )
            
            change_model_btn.click(
                fn=self.change_model,
                inputs=model_selector,
                outputs=model_change_output
            ).then(
                fn=lambda: gr.update(visible=True),
                outputs=model_change_output
            )
            
            refresh_models_btn.click(
                fn=self.refresh_models_list,
                outputs=model_selector
            )
            
            # Index Management Events
            index_selector.change(
                fn=self.update_index_description,
                inputs=index_selector,
                outputs=index_description
            )
            
            # Clear Operations Events
            clear_selector.change(
                fn=self.update_clear_description,
                inputs=clear_selector,
                outputs=clear_description
            )
            
            reset_index_btn.click(
                fn=self.reset_selected_index,
                inputs=[index_selector, reset_confirm],
                outputs=management_output
            )
            
            rebuild_index_btn.click(
                fn=self.rebuild_selected_index,
                inputs=index_selector,
                outputs=management_output
            )
            
            # Clear selected domain button
            clear_selected_btn.click(
                fn=self.clear_selected_data,
                inputs=[clear_selector, clear_confirm],
                outputs=management_output
            )
            
            # LlamaIndex Management Event Handlers
            refresh_llamaindex_btn.click(
                fn=self.get_llamaindex_status,
                outputs=llamaindex_info
            )
            
            
            # Initial status check
            interface.load(
                fn=self.get_system_info,
                outputs=status_output
            )
            
            interface.load(
                fn=self.list_documents,
                outputs=doc_list_output
            )
            
            # Initialize dropdowns on page load
            interface.load(
                fn=self.refresh_documents_with_dropdown,
                outputs=[doc_list_output, doc_dropdown]
            )
            
            # Copyright footer
            gr.Markdown("---")
            gr.Markdown(
                "<div style='text-align: center; color: #666; font-size: 12px; padding: 20px;'>"
                "© 2025 precellence.icu - Myr-Ag RAG System"
                "</div>"
            )
        
        return interface
    
    def launch(self, server_name="0.0.0.0", server_port=7860, share=False):
        """Launch the Gradio interface."""
        logger.info("Launching Gradio interface...")
        interface = self.create_interface()
        
        # Check API health before launching
        if not self.check_api_health():
            logger.warning("API server is not running!")
            print("⚠️ Warning: API server is not running!")
            print("Please start the FastAPI server first:")
            print("python run_api.py")
            print()
        
        logger.info(f"Starting Gradio interface on {server_name}:{server_port}")
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            show_error=True
        )


# Create global instance
frontend = GradioFrontend()
