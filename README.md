# Azure Document Processing API - I&C Declaration Document Reading

FastAPI application for processing documents from Azure Blob Storage using Azure AI Content Understanding and transforming data with Azure OpenAI.

## Features

- Process single documents or batch process entire folders
- Custom document analysis using Azure AI Content Understanding
- Data transformation to custom JSON templates using Azure OpenAI
- SAS token-based secure blob access
- Comprehensive logging and debugging endpoints

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Azure credentials:
```env
AZURE_BLOB_CONNECTION_STRING=your_connection_string
AZURE_BLOB_CONTAINER_NAME=your_container_name
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=your_endpoint
AZURE_DOCUMENT_INTELLIGENCE_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_openai_endpoint
AZURE_OPENAI_KEY=your_openai_key
AZURE_OPENAI_DEPLOYMENT=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-12-01-preview
CONTENT_UNDERSTANDING_PROJECT_NAME=your_project_name
```

## Running the Application

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## API Endpoints

- `POST /process-document` - Process a single document or folder
- `GET /debug/models` - List available analysis models
- `GET /debug/config` - View current configuration
- `GET /debug/list-analyzers` - List Content Understanding analyzers
- `GET /setup/content-understanding-deployments` - List Content Understanding deployments

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
