from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from azure.ai.formrecognizer import DocumentAnalysisClient, DocumentModelAdministrationClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import json
import os
from typing import Optional
from dotenv import load_dotenv
from datetime import datetime, timedelta
import logging
import requests
import time
from azure.storage.blob import generate_container_sas, ContainerSasPermissions
from datetime import datetime, timedelta
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Azure Document Processing API")

# Configuration - Set these as environment variables
BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
BLOB_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME")
DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
DOCUMENT_INTELLIGENCE_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# Initialize clients
blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
document_analysis_client = DocumentAnalysisClient(
    endpoint=DOCUMENT_INTELLIGENCE_ENDPOINT,
    credential=AzureKeyCredential(DOCUMENT_INTELLIGENCE_KEY)
)
document_admin_client = DocumentModelAdministrationClient(
    endpoint=DOCUMENT_INTELLIGENCE_ENDPOINT,
    credential=AzureKeyCredential(DOCUMENT_INTELLIGENCE_KEY)
)
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# Request/Response Models
class DocumentRequest(BaseModel):
    folder_path: str
    filename: Optional[str] = None
    target_template: dict
    model_id: str  # Your custom model ID or prebuilt models (e.g., "prebuilt-invoice", "prebuilt-receipt", "prebuilt-layout", etc.)

class DocumentResponse(BaseModel):
    status: str
    transformed_data: dict
    original_analysis: Optional[dict] = None

class BatchDocumentResponse(BaseModel):
    status: str
    total_files: int
    processed_files: int
    results: list[dict]

# Predefined JSON template example (you can modify this)
DEFAULT_TEMPLATE = {
    "document_type": "",
    "document_date": "",
    "vendor_name": "",
    "total_amount": 0.0,
    "line_items": [],
    "extracted_fields": {}
}

@app.post("/process-document")
async def process_document(request: DocumentRequest):
    """
    Process a document from Azure Blob Storage using Document Intelligence and transform with OpenAI.
    If filename is provided, processes single file. If not, processes all files in folder_path.
    """
    try:
        # Step 1: Determine if processing single file or folder
        if request.filename:
            # Process single file
            blob_path = f"{request.folder_path}/{request.filename}".lstrip("/")
            return await process_single_document(blob_path, request.model_id, request.target_template)
        else:
            # Process all files in folder
            return await process_folder(request.folder_path, request.model_id, request.target_template)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

async def analyze_with_content_understanding(blob_url: str, model_id: str):
    """
    Analyze document using Content Understanding REST API for custom analyzers
    """
    # Content Understanding API endpoint (no project prefix needed)
    analyze_url = f"{DOCUMENT_INTELLIGENCE_ENDPOINT}contentunderstanding/analyzers/{model_id}:analyze"
    
    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": DOCUMENT_INTELLIGENCE_KEY
    }
    
    # Body format must be an array of inputs
    body = {
        "inputs": [
            {
                "url": blob_url
            }
        ]
    }
    
    params = {       
        "api-version": "2025-11-01"
    }
    
    logger.info(f"Calling Content Understanding API")
    logger.info(f"URL: {analyze_url}")
    logger.info(f"Params: {params}")
    logger.info(f"Body: {body}")
    
    # Start the analysis
    response = requests.post(analyze_url, headers=headers, json=body, params=params)
    
    logger.info(f"Response Status: {response.status_code}")
    logger.info(f"Response Headers: {dict(response.headers)}")
    logger.info(f"Response Body: {response.text[:1000]}")
    
    if response.status_code not in [200, 202]:
        error_msg = f"Analysis failed: {response.status_code} - {response.text}"
        logger.error(error_msg)
        raise HTTPException(status_code=response.status_code, detail=error_msg)
    
    # Get the operation location to poll for results
    operation_location = response.headers.get("Operation-Location") or response.headers.get("operation-location")
    
    if not operation_location:
        # Might be synchronous response
        result = response.json()
        logger.info("Received synchronous response")
        return result
    
    logger.info(f"Polling for results at: {operation_location}")
    
    # Poll for results - increase timeout for AI-powered analysis
    max_attempts = 120  # 4 minutes total (120 * 2 seconds)
    for attempt in range(max_attempts):
        time.sleep(2)
        result_response = requests.get(operation_location, headers=headers)
        
        if result_response.status_code == 200:
            result_data = result_response.json()
            status = result_data.get("status", "").lower()  # Convert to lowercase for comparison
            
            logger.info(f"Polling attempt {attempt + 1}/{max_attempts}, Status: {status}")
            
            if status == "succeeded":
                logger.info("Analysis succeeded!")
                return result_data.get("result", result_data)
            elif status == "failed":
                error = result_data.get("error", {})
                logger.error(f"Analysis failed: {error}")
                raise HTTPException(status_code=500, detail=f"Analysis failed: {error}")
            
            # Continue polling for running/notStarted status
        else:
            logger.warning(f"Polling attempt {attempt + 1} failed with status: {result_response.status_code}")
            logger.warning(f"Response: {result_response.text[:500]}")
    
    raise HTTPException(status_code=408, detail="Analysis timed out after 4 minutes")

async def process_single_document(blob_path: str, model_id: str, target_template: dict):
    """
    Process a single document from blob storage
    """
    logger.info(f"Processing document: {blob_path} with model: {model_id}")
    
    blob_client = blob_service_client.get_blob_client(
        container=BLOB_CONTAINER_NAME,
        blob=blob_path
    )
    
    # Check if blob exists
    if not blob_client.exists():
        logger.error(f"Blob not found: {blob_path}")
        raise HTTPException(status_code=404, detail=f"File not found: {blob_path}")
    
    logger.info(f"Blob found, generating SAS token...")
    
    # Generate SAS token for the blob (valid for 1 hour)
    sas_token = generate_blob_sas(
        account_name=blob_client.account_name,
        container_name=BLOB_CONTAINER_NAME,
        blob_name=blob_path,
        account_key=blob_service_client.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=1)
    )
    
    # Get blob URL with SAS token for Document Intelligence
    blob_url = f"{blob_client.url}?{sas_token}"
    logger.info(f"Generated blob URL with SAS token")
    
    # Step 2: Analyze document - check if using custom Content Understanding analyzer
    logger.info(f"Starting document analysis with model: {model_id}")
    
    # Try Content Understanding API first for custom models
    if not model_id.startswith("prebuilt-"):
        logger.info(f"Using Content Understanding REST API for custom model: {model_id}")
        result = await analyze_with_content_understanding(blob_url, model_id)
    else:
        # Use traditional Document Intelligence SDK for prebuilt models
        logger.info(f"Using Document Intelligence SDK for prebuilt model: {model_id}")
        poller = document_analysis_client.begin_analyze_document_from_url(
            model_id=model_id,
            document_url=blob_url
        )
        result = poller.result()
    
    logger.info(f"Document analysis completed")
        
    # Convert result to JSON-serializable format
    if isinstance(result, dict):
        # Already in dict format from Content Understanding API
        analysis_result = result
    else:
        # Convert from Document Intelligence SDK result object
        analysis_result = {
            "model_id": result.model_id,
            "content": result.content,
            "pages": [],
            "tables": [],
            "key_value_pairs": [],
            "documents": []
        }
        
        # Extract pages
        for page in result.pages:
            analysis_result["pages"].append({
                "page_number": page.page_number,
                "width": page.width,
                "height": page.height,
                "unit": page.unit,
                "lines": [{"content": line.content} for line in page.lines] if page.lines else []
            })
        
        # Extract tables
        for table in result.tables:
            table_data = {
                "row_count": table.row_count,
                "column_count": table.column_count,
                "cells": []
            }
            for cell in table.cells:
                table_data["cells"].append({
                    "row_index": cell.row_index,
                    "column_index": cell.column_index,
                    "content": cell.content
                })
            analysis_result["tables"].append(table_data)
        
        # Extract key-value pairs
        for kv_pair in result.key_value_pairs:
            if kv_pair.key and kv_pair.value:
                analysis_result["key_value_pairs"].append({
                    "key": kv_pair.key.content,
                    "value": kv_pair.value.content
                })
        
        # Extract document fields
        for doc in result.documents:
            doc_data = {"doc_type": doc.doc_type, "fields": {}}
            for name, field in doc.fields.items():
                if field.value:
                    doc_data["fields"][name] = str(field.value)
            analysis_result["documents"].append(doc_data)
    
    # Step 3: Transform with Azure OpenAI
    template = target_template if target_template else DEFAULT_TEMPLATE
    
    logger.info(f"Transforming data with OpenAI model: {AZURE_OPENAI_DEPLOYMENT}")
        
    prompt = f"""You are a data transformation assistant. 
        
I have extracted data from a document using Azure Document Intelligence. 
Please transform this data into the specified JSON template format.

Extracted Document Data:
{json.dumps(analysis_result, indent=2)}

Target JSON Template:
{json.dumps(template, indent=2)}

Instructions:
1. Map the extracted data to the template fields as accurately as possible
2. If a field cannot be found in the extracted data, leave it as null or empty based on the template
3. Maintain the exact structure of the template
4. Return ONLY the valid JSON object, no explanations or additional text

Transformed JSON:"""

    response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "You are a data transformation expert. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
    
    transformed_data = json.loads(response.choices[0].message.content)
    
    logger.info(f"Document processing completed successfully for: {blob_path}")
    
    return DocumentResponse(
        status="success",
        transformed_data=transformed_data,
        original_analysis=analysis_result
    )

async def process_folder(folder_path: str, model_id: str, target_template: dict):
    """
    Process all documents in a folder from blob storage
    """
    container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
    
    # List all blobs in the folder
    folder_prefix = folder_path.lstrip("/")
    if not folder_prefix.endswith("/"):
        folder_prefix += "/"
    
    blobs = container_client.list_blobs(name_starts_with=folder_prefix)
    
    results = []
    total_files = 0
    processed_files = 0
    
    for blob in blobs:
        # Skip folders (blobs ending with /)
        if blob.name.endswith("/"):
            continue
            
        total_files += 1
        try:
            result = await process_single_document(blob.name, model_id, target_template)
            results.append({
                "file": blob.name,
                "status": "success",
                "data": result.transformed_data,
                "analysis": result.original_analysis
            })
            processed_files += 1
        except Exception as e:
            results.append({
                "file": blob.name,
                "status": "failed",
                "error": str(e)
            })
    
    if total_files == 0:
        raise HTTPException(status_code=404, detail=f"No files found in folder: {folder_path}")
    
    return BatchDocumentResponse(
        status="completed",
        total_files=total_files,
        processed_files=processed_files,
        results=results
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/debug/models")
async def list_models():
    """List all available Document Intelligence models"""
    try:
        models = document_admin_client.list_document_models()
        custom_models = []
        for model in models:
            custom_models.append({
                "model_id": model.model_id,
                "description": model.description if model.description else "No description",
                "created_on": str(model.created_on)
            })
        return {
            "status": "success",
            "endpoint": DOCUMENT_INTELLIGENCE_ENDPOINT,
            "total_models": len(custom_models),
            "models": custom_models
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/debug/config")
async def debug_config():
    """Debug configuration"""
    return {
        "blob_container": BLOB_CONTAINER_NAME,
        "doc_intelligence_endpoint": DOCUMENT_INTELLIGENCE_ENDPOINT,
        "openai_deployment": AZURE_OPENAI_DEPLOYMENT,
        "openai_api_version": AZURE_OPENAI_API_VERSION,
        "has_blob_connection": bool(BLOB_CONNECTION_STRING),
        "has_doc_key": bool(DOCUMENT_INTELLIGENCE_KEY),
        "has_openai_key": bool(AZURE_OPENAI_KEY)
    }

@app.get("/debug/content-understanding")
async def test_content_understanding():
    """Test Content Understanding API connection"""
    try:
        # Try different API endpoints to find the right one
        test_endpoints = [
            {
                "name": "List Analyzers v2024-11-30",
                "url": f"{DOCUMENT_INTELLIGENCE_ENDPOINT}contentunderstanding/analyzers",
                "params": {"api-version": "2024-11-30"}
            },
            {
                "name": "List Analyzers v2024-07-31-preview",
                "url": f"{DOCUMENT_INTELLIGENCE_ENDPOINT}contentunderstanding/analyzers",
                "params": {"api-version": "2024-07-31-preview"}
            },
            {
                "name": "Test Analyzer Endpoint",
                "url": f"{DOCUMENT_INTELLIGENCE_ENDPOINT}contentunderstanding/analyzers/icDOBLAnalyzer",
                "params": {"api-version": "2024-11-30"}
            }
        ]
        
        headers = {
            "Ocp-Apim-Subscription-Key": DOCUMENT_INTELLIGENCE_KEY
        }
        
        results = []
        for test in test_endpoints:
            try:
                response = requests.get(test["url"], headers=headers, params=test["params"])
                results.append({
                    "name": test["name"],
                    "url": test["url"],
                    "status": response.status_code,
                    "response": response.json() if response.status_code == 200 else response.text[:500]
                })
            except Exception as e:
                results.append({
                    "name": test["name"],
                    "url": test["url"],
                    "error": str(e)
                })
        
        return {
            "status": "test_complete",
            "endpoint": DOCUMENT_INTELLIGENCE_ENDPOINT,
            "results": results,
            "instructions": "Check the Azure Content Understanding Studio at https://contentunderstanding.ai.azure.com/projects/ and look for 'Use model' or 'API' section to get the exact endpoint format."
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/debug/list-analyzers")
async def list_project_analyzers():
    """List all analyzers in the Content Understanding project"""
    try:
        project_name = os.getenv("CONTENT_UNDERSTANDING_PROJECT_NAME", "")
        
        if not project_name:
            return {"error": "CONTENT_UNDERSTANDING_PROJECT_NAME not set in .env"}
        
        # Try to list analyzers in the project
        url = f"{DOCUMENT_INTELLIGENCE_ENDPOINT}contentunderstanding/projects/{project_name}/analyzers"
        
        headers = {
            "Ocp-Apim-Subscription-Key": DOCUMENT_INTELLIGENCE_KEY
        }
        
        params = {
            "api-version": "2024-12-01-preview"
        }
        
        response = requests.get(url, headers=headers, params=params)
        
        return {
            "status": "success" if response.status_code == 200 else "error",
            "project": project_name,
            "url": url,
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/setup/content-understanding-deployments")
async def setup_content_understanding_deployments():
    """Configure Content Understanding GPT model deployments"""
    try:
        url = f"{DOCUMENT_INTELLIGENCE_ENDPOINT}contentunderstanding/defaults"
        
        headers = {
            "Ocp-Apim-Subscription-Key": DOCUMENT_INTELLIGENCE_KEY,
            "Content-Type": "application/json"
        }
        
        params = {
            "api-version": "2024-11-30"
        }
        
        # Configure model deployments - map Content Understanding model names to your Azure OpenAI deployments
        body = {
            "modelDeployments": {
                "gpt-4o": AZURE_OPENAI_DEPLOYMENT,  # Map to your actual deployment
            }
        }
        
        logger.info(f"Configuring Content Understanding deployments: {body}")
        
        response = requests.patch(url, headers=headers, json=body, params=params)
        
        if response.status_code in [200, 204]:
            return {
                "status": "success",
                "message": "Model deployments configured",
                "configured": body["modelDeployments"]
            }
        else:
            return {
                "status": "error",
                "status_code": response.status_code,
                "response": response.text
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Azure Document Processing API",
        "endpoints": {
            "process_document": "/process-document",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)