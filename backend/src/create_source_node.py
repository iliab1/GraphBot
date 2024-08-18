from langchain_core.documents import Document
from typing import List
from datetime import datetime
import os
import hashlib
from config.logging_config import logger
from src.validation.source_node import SourceNode


# This module contains functions for creating source nodes from documents.
# This node will then be stored in the Neo4j database.

def create_source_node_from_document(document: List[Document]) -> SourceNode:
    """
    Creates a source node from the document.

    Attributes of the source node:
    - file_name: The name of the file from the document's metadata
    - langchain_document: The list of documents for later use
    - text: The concatenated text content of all pages in the document
    - hash: SHA-256 hash of the text content
    - embedding: Initialized to None
    - created_at: Timestamp when the source node is created
    - updated_at: Initialized to None
    """
    # Create a source node
    current_source_node = SourceNode()
    logger.info(f"Creating source node from {document[0].metadata['source']}...")
    # Extract the file name from the document metadata
    current_source_node.file_name = os.path.basename(document[0].metadata["source"])
    logger.info(f"File name: {current_source_node.file_name}")
    # Store the langchain document inside the source node for later use
    current_source_node.langchain_document = document
    logger.info(f"Langchain document stored in source node.")
    # Concatenate the text content of all pages in the document
    full_text = "".join(doc.page_content for doc in document)
    current_source_node.text = full_text

    # Generate a hash for the document based on the content
    hasher = hashlib.sha256()
    hasher.update(full_text.encode('utf-8'))
    current_source_node.hash = hasher.hexdigest()
    logger.info(f"Hash generated for the document.")
    # Initialize embedding as None
    current_source_node.embedding = None

    # Set the current date and time for created_at
    current_source_node.created_at = datetime.now()

    # Set updated_at as None
    current_source_node.updated_at = datetime.now()

    logger.info(f"Created source node from {current_source_node.file_name} successfully.")

    return current_source_node
