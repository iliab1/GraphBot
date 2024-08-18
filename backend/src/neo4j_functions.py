from neo4j.exceptions import ClientError
from src.validation.source_node import SourceNode
from datetime import datetime
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument
from typing import List
import logging
from src.validation.client_auth import ClientAuth


# This module contains functions for interacting with the Neo4j graph database.
# It was inspired by https://github.com/neo4j-labs/llm-graph-builder/blob/main/backend/src/graphDB_dataAccess.py

def create_graph_database_connection(current_user: ClientAuth) -> Neo4jGraph:
    """
    Creates and returns a connection to the Neo4j graph database
    using the credentials of the current user.
    """
    try:
        graph = Neo4jGraph(
            url=current_user.uri,
            database=current_user.database,
            username=current_user.userName,
            password=current_user.password
        )

        return graph
    except Exception as e:
        logging.error(f"Failed to create graph database connection: {e}")
        raise


def save_graph_documents_to_database(graph: Neo4jGraph, graph_document_list: List[GraphDocument]) -> None:
    """
    Saves Graph Documents to the Neo4j graph database.
    """
    graph.add_graph_documents(graph_document_list)


# Main class for interacting with the Neo4j graph database
class GraphDBFunctions:
    """
    Initializes GraphDBFunctions class.

    Args:
        graph (Neo4jGraph): The graph database connection object.
    """

    def __init__(self, graph: Neo4jGraph):
        self.graph = graph

    def execute_query(self, query, param=None):
        """
        Executes a query on the graph database.
        """
        return self.graph.query(query, param)

    def create_source_node(self, current_source_node: SourceNode):
        """
        Creates a source node in the graph database from the SourceNode class.
        """

        try:
            params = {
                "file_name": current_source_node.file_name,
                "text": current_source_node.text,
                "hash": current_source_node.hash,
                "embedding": current_source_node.embedding,
                "created_at": current_source_node.created_at,
                "updated_at": current_source_node.updated_at
            }

            self.graph.query(
                """
                MERGE (d:Document {hash :$hash})
                SET d.fileName = $file_name,
                    d.text = $text,
                    d.embedding = $embedding,
                    d.createdAt = $created_at,
                    d.updatedAt = $updated_at
                RETURN d
                """,
                params,
            )

        except ClientError as e:
            error_message = str(e)
            raise Exception(error_message)

    def delete_source_node(self, file_name: str):
        """
        Deletes source node from the graph database and all related dependencies (chunks, entities and relationships).
        """
        query_to_delete_document_and_entities = """ 
            MATCH (d:Document {fileName: $file_name})
            OPTIONAL MATCH (d)-[:HAS_PARENT]->(p:Parent)
            OPTIONAL MATCH (p)-[:HAS_CHILD]->(c:Child)
            // Nested query to delete entities that are part of the parent
            CALL {
                WITH  p, d
                MATCH (e)-[:PART_OF]->(p)
                // Do not delete entities that are
                // also connected to other documents
                WHERE NOT EXISTS {
                    MATCH (e)-[:PART_OF]->(other_p:Parent)
                    MATCH (other_p)<-[:HAS_PARENT]-(other_d:Document)
                    WHERE other_d <> d
                }

                DETACH DELETE e
                RETURN count(*) as entities
            } 
            // Delete children, parent, and document nodes
            DETACH DELETE c, p, d
            RETURN sum(entities) as deletedEntities, count(*) as deletedChunks
            """

        param = {"file_name": file_name}

        result = self.execute_query(query_to_delete_document_and_entities, param)

        return result

    def get_current_status_document_node(self, file_name):
        """
        Returns the current status of the source document node in the graph database.
        Useful for tracking the progress of the document processing.
        """
        query = """
                MATCH(d:Document {fileName : $file_name}) 
                RETURN d.status AS Status , 
                       d.processingTime AS processingTime, 
                       d.nodeCount AS nodeCount,
                       d.model as model,
                       d.relationshipCount as relationshipCount,
                       d.total_pages AS total_pages,
                       d.total_chunks AS total_chunks,
                       d.fileSize as fileSize,
                       d.is_cancelled as is_cancelled,
                       d.processed_chunk as processed_chunk,
                       d.fileSource as fileSource
                """
        param = {"file_name": file_name}
        return self.execute_query(query, param)

    def update_source_node(self, current_source_node: SourceNode):
        """
        Updates the source node in the graph database.
        """

        try:
            params = {}
            # Read the source node parameters
            if current_source_node.file_name is not None and current_source_node.file_name != '':
                params["fileName"] = current_source_node.file_name

            if current_source_node.status is not None and current_source_node.status != '':
                params["status"] = current_source_node.status

            if current_source_node.text is not None and current_source_node.text != '':
                params["text"] = current_source_node.text

            if current_source_node.hash is not None and current_source_node.hash != '':
                params["hash"] = current_source_node.hash

            if current_source_node.embedding is not None and current_source_node.embedding != '':
                params["embedding"] = current_source_node.embedding

            if current_source_node.created_at is not None:
                params['createdAt'] = current_source_node.created_at

            if current_source_node.updated_at is not None:
                params['updatedAt'] = datetime.now()

            param = {"props": params}

            # Update the source node properties
            query = "MERGE(d:Document {hash :$props.hash}) SET d += $props"
            self.graph.query(query, param)

        except ClientError as e:
            error_message = str(e)
            raise Exception(error_message)

    def get_source_list(self):
        """
        Returns a list of all source nodes from the graph database in json format.
        """

        query = "MATCH(d:Document) WHERE d.fileName IS NOT NULL RETURN d ORDER BY d.updatedAt DESC"
        result = self.graph.query(query)
        list_of_json_objects = [entry['d'] for entry in result]
        return list_of_json_objects
