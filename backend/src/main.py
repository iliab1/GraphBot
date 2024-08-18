from src.create_document import create_document_from_pdf
from src.create_source_node import create_source_node_from_document
from src.neo4j_functions import GraphDBFunctions, create_graph_database_connection
from src.create_chunks import create_chunks_from_source
from src.extract_relationships import create_graph_documents
from src.validation.client_auth import ClientAuth
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from config.logging_config import logger

from src.graph_post_processing.entity_resolution import create_entity_embeddings
from src.graph_post_processing.entity_resolution import find_similar_entities
from src.graph_post_processing.entity_resolution import perform_entity_resolution


def extract_graph_from_local_file(file_path, current_client: ClientAuth):
    """
    The full pipeline for extracting graph from local document.
    :param file_path:
    :param current_client:
    :return: SourceNode
    """
    # Load the document
    document = create_document_from_pdf(file_path)
    # Create the source node
    current_source_node = create_source_node_from_document(document)

    # Create graph database connection
    try:
        graph = create_graph_database_connection(current_client)
    except Exception as e:
        logger.error(f"Failed to create graph database connection: {e}")

    neo4j_functions = GraphDBFunctions(graph)
    # Create the source node in the graph database
    neo4j_functions.create_source_node(current_source_node)
    # Extract the graph from the document

    embeddings = OpenAIEmbeddings(openai_api_key=current_client.openai_key)
    embedding_dimension = 1536
    # Embed the text of the source node
    current_source_node.embedding = embeddings.embed_query(current_source_node.text)
    neo4j_functions.update_source_node(current_source_node)
    logger.info(f"Generated embeddings for {current_source_node.file_name}")
    # Create the chunks from the source node
    current_source_node = create_chunks_from_source(current_source_node, graph, embedding_dimension, embeddings)
    # Extract entities and rels from parent chunks of the document
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, openai_api_key=current_client.openai_key)
    current_source_node = create_graph_documents(current_source_node, llm)
    # Store the graph documents in the graph database
    graph.add_graph_documents(current_source_node.langchain_graph_documents, baseEntityLabel=True)
    logger.info(f"Nodes and Rels added to Database for {current_source_node.file_name}")
    # Create relationships between chunks and entities
    graph.query(
        """
        MATCH (e:__Entity__), (p:Parent)
        WHERE e.parent_hash = p.hash
        MERGE (e)-[:PART_OF]->(p)
        """
    )
    logger.info(f"Completed {current_source_node.file_name}")
    return current_source_node


# Function to post process the graph
def run_entity_resolution(current_client):
    try:
        duplicate_entities = perform_entity_resolution(current_client)
        return duplicate_entities
    except Exception as e:
        logger.error(f"Failed to perform entity resolution: {e}")
        return None


def get_sources(current_client: ClientAuth):
    """
    Get sources list from graph database.
    :param current_client:
    :return: sources_list
    """
    # Create graph database connection
    try:
        graph = create_graph_database_connection(current_client)
    except Exception as e:
        logger.error(f"Failed to create graph database connection: {e}")

    neo4j_functions = GraphDBFunctions(graph)
    # Get sources list
    sources_list = neo4j_functions.get_source_list()
    return sources_list
