from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument


class SourceNode(BaseModel):
    """
    SourceNode class for creating a source node in the Neo4j graph database.
    """
    file_name: Optional[str] = Field(default=None)
    file_path: Optional[str] = Field(default=None)
    status: Optional[str] = Field(default=None)
    langchain_document: Optional[List[Document]] = Field(default=None)
    langchain_parent_chunks: Optional[List[Document]] = Field(default=None)
    langchain_child_chunks: Optional[List[Document]] = Field(default=None)
    llm: Optional[str] = Field(default=None)
    langchain_graph_documents: Optional[List[GraphDocument]] = Field(default=None)
    text: Optional[str] = Field(default=None)
    hash: Optional[str] = Field(default=None)
    embedding: Optional[list] = Field(default=None)
    created_at: Optional[datetime] = Field(default=None)
    updated_at: Optional[datetime] = Field(default=None)

    def __str__(self):
        return str(self.dict())

    def __repr__(self):
        return self.__str__()
