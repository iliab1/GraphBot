from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from config.logging_config import logger

# This module contains functions for processing raw documents
# It includes utilities for loading and handling various format files


def create_document_from_pdf(path: Path) -> List[Document]:
    """
    Creates a Document from the PDF file.
    """
    try:
        loader = PyPDFLoader(str(path))
        document = loader.load()
        logger.info(f"Document created from {path}")
        return document

    except Exception as e:
        logger.error(f"Error creating document from {path}: {e}")