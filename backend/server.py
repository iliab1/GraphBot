from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Depends
from fastapi.responses import RedirectResponse
from pathlib import Path
import aiofiles
import asyncio
from src.main import extract_graph_from_local_file
from src.main import get_sources
from src.validation.client_auth import ClientAuth
from src.validation.api_response import create_api_response
from config.logging_config import logger
from langserve import add_routes

from pydantic.v1 import BaseModel as BaseModelV1
import os
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.middleware.sessions import SessionMiddleware
from src.neo4j_functions import create_graph_database_connection, GraphDBFunctions
from fastapi.responses import JSONResponse
from pydantic import Field
import openai
from dotenv import load_dotenv

from src.main import run_entity_resolution
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage
from typing import Any, List, Union
# Chat chains
from src.chat.chains import typical_rag_chain, parent_rag_chain, create_graph_neighbour_chain
from src.chat.advanced_chain import multi_hop_qa_agent, create_advanced_rag_chain

load_dotenv()

app = FastAPI()

# Test logging at startup
logger.info("Dummy Info")
logger.error("Dummy Error")
logger.debug("Dummy Debug")
logger.warning("Dummy Warning")

# Allow CORS for all origins
# To ensure the API is accessible from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session management middleware
# To persist the client authentication details
app.add_middleware(SessionMiddleware, secret_key=os.urandom(24))


# Set authentication details
@app.post("/set_auth_details", tags=["Authentication"])
async def set_auth_details(
        request: Request,
        client_auth: ClientAuth
):
    try:
        # Test authentication details to make sure they are set correctly
        create_graph_database_connection(client_auth)
        client = openai.OpenAI(api_key=client_auth.openai_key)
        client.models.list()
        request.session['auth_details'] = client_auth.dict()
        logger.info("Authentication details set successfully")
        return create_api_response(status="success", message="Authentication details set successfully")

    except Exception as e:
        logger.error(f"Error setting authentication details: {e}")
        return JSONResponse(
            content=create_api_response(
                status="error",
                message="Error setting authentication details",
                error=str(e)))


# Get authentication details
# For testing purposes
@app.get("/get_auth_details", tags=["Authentication"])
async def get_auth_details(request: Request):
    auth_details = request.session.get('auth_details')
    if not auth_details:
        return create_api_response(status="error", message="Authentication details not found",
                                   error="Authentication details not found")
    return create_api_response(status="success", data=auth_details)


# Check authentication details
# For testing purposes
def get_current_user(request: Request):
    auth_details = request.session.get('auth_details')
    if not auth_details:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return ClientAuth(**auth_details)


# Test of a protected endpoint
# For testing purposes
@app.get("/protected_endpoint", tags=["Authentication"])
async def protected_endpoint(current_user: ClientAuth = Depends(get_current_user)):
    return create_api_response(status="success",
                               data={"message": "This is a protected endpoint", "user": current_user.dict()})


# Redirect root to docs for easier access to Swagger UI
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Test debug logging
# For testing purposes
@app.get("/test", tags=["Test"])
async def test(test_msg: str = Query(None)):
    logger.info("Received a test request")
    logger.debug(f"Test debug message: {test_msg}")
    return create_api_response(status="success", data={"test_msg": test_msg})


# File upload endpoint
@app.post("/uploadfile/", tags=["File Management"])
async def create_upload_file(
        file: UploadFile = File(...),
        current_user: ClientAuth = Depends(get_current_user)
):
    logger.info(f"Received file upload request: {file.filename}")
    upload_folder = Path("uploads")
    upload_folder.mkdir(exist_ok=True)
    file_path = upload_folder / file.filename

    async with aiofiles.open(file_path, 'wb') as out_file:
        while content := await file.read(1024):  # Read file in chunks
            await out_file.write(content)
    logger.info(f"File saved to {file_path}")

    try:
        current_source_node = await asyncio.to_thread(extract_graph_from_local_file, file_path,
                                                      current_user)  # Provide file_path as argument
        logger.info(f"Graph extracted for {current_source_node.file_name}")
        return create_api_response(status="success",
                                   message=f"Successfully processed file: {current_source_node.file_name}")
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return create_api_response(status="error", message=f"Error processing file {str(e)}", error=str(e))


# File upload endpoint for multiple files
@app.post("/uploadfiles/", tags=["File Management"])
async def create_upload_files(
        files: List[UploadFile] = File(...),
        current_user: ClientAuth = Depends(get_current_user)
):
    upload_folder = Path("uploads")
    upload_folder.mkdir(exist_ok=True)

    file_responses = []
    for file in files:
        file_path = upload_folder / file.filename

        try:
            async with aiofiles.open(file_path, 'wb') as out_file:
                while content := await file.read(1024):  # Read file in chunks
                    await out_file.write(content)
            logger.info(f"File saved to {file_path}")

            current_source_node = await asyncio.to_thread(extract_graph_from_local_file, file_path,
                                                          current_user)  # Provide file_path as argument
            logger.info(f"Graph extracted for {current_source_node.file_name}")
            file_responses.append({"file_name": current_source_node.file_name, "status": "success"})
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            file_responses.append({"file_name": file.filename, "status": "error", "error": str(e)})

    return create_api_response(status="success", data=file_responses)


# Delete file endpoint
@app.post("/deletefile/", tags=["File Management"])
async def delete_file(
        file_name: str = Query(...),
        current_user: ClientAuth = Depends(get_current_user)
):
    logger.info(f"Received request to delete file: {file_name}")
    file_path = Path("uploads") / file_name

    if file_path.exists():
        try:
            file_path.unlink()  # Delete the file
            logger.info(f"File {file_name} deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting local file: {e}")
            return create_api_response(status="error", message="Error deleting local file", error=str(e))
    else:
        logger.warning(f"File {file_name} not found locally but will continue to database deletion.")
        #return create_api_response(status="error", message=f"File '{file_name}' not found", error="File not found")

    try:
        graph = create_graph_database_connection(current_user)
        result = GraphDBFunctions(graph).delete_source_node(file_name)
        logger.info(f"Deleted '{file_name}' from database with result: {result}")
        return create_api_response(status="success", message=f"File '{file_name}' deleted successfully", data=result)
    except Exception as e:
        logger.error(f"Error deleting file from database: {e}")
        return create_api_response(status="error", message="Error deleting file from database", error=str(e))


# Get sources list endpoint
@app.post("/sources_list", tags=["File Management"])
async def get_sources_list(current_user: ClientAuth = Depends(get_current_user)):
    try:
        result = await asyncio.to_thread(get_sources, current_user)
        logger.info("Received request to get sources list")
        return create_api_response(status="success", data=result)
    except Exception as e:
        logger.exception(f'Exception: {e}')
        return create_api_response(status="error", message="Error getting sources list", error=str(e))


@app.post("/perform_entity_resolution", tags=["Post-Processing"])
async def entity_resolution(current_user: ClientAuth = Depends(get_current_user)):
    try:
        duplicate_entities = run_entity_resolution(current_user)
        return create_api_response(status="success", data=duplicate_entities)
    except Exception as e:
        logger.exception(f'Exception: {e}')
        return create_api_response(status="error", message="Error resolving entities", error=str(e))


# TODO: Finish community creation function
@app.post("/create_communities", tags=["Post-Processing"])
async def create_communities(current_user: ClientAuth = Depends(get_current_user)):
    await create_communities(current_user)


class AgentInput(BaseModelV1):
    input: str
    #chat_history: List[Union[HumanMessage, AIMessage, FunctionMessage]] = Field(
    #    ...)


class AgentOutput(BaseModelV1):
    output: Any


class ChainInput(BaseModelV1):
    question: str


# Initialise routes endpoint for Langserve chatbot integration
@app.get("/initialize_routes", tags=["Langserve"])
async def initialize_routes(
        current_user: ClientAuth = Depends(get_current_user)
):
    try:
        # Initialise chains
        typical_rag = await asyncio.to_thread(typical_rag_chain, current_user)
        parent_rag = await asyncio.to_thread(parent_rag_chain, current_user)
        graph_neighbour = await asyncio.to_thread(create_graph_neighbour_chain, current_user)
        advanced_rag = await asyncio.to_thread(create_advanced_rag_chain, current_user)

        add_routes(app, typical_rag.with_config(input_type=ChainInput), path="/typical_chain")
        add_routes(app, parent_rag.with_config(input_type=ChainInput), path="/parent_chain")
        add_routes(app, graph_neighbour.with_config(input_type=ChainInput), path="/graph_chain")
        add_routes(app, advanced_rag.with_config(input_type=ChainInput), path="/advanced_chain")

        # Initialize agent
        qa_agent = await asyncio.to_thread(multi_hop_qa_agent, current_user)
        add_routes(app,
                   qa_agent.with_types(input_type=AgentInput, output_type=AgentOutput).with_config({"run_name": "Agent"}),
                   path="/qa_agent"
                   )

        logger.info("Routes initialized successfully")
        return create_api_response(status="success", message="Routes initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing routes: {e}")
        return create_api_response(status="error", message="Error initializing routes", error=str(e))
