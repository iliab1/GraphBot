from src.validation.source_node import SourceNode
import hashlib
from langchain_text_splitters import TokenTextSplitter
from src.neo4j_functions import ClientError
from config.logging_config import logger
from tqdm import tqdm


# Inspired by langchain template:
# https://github.com/langchain-ai/langchain/blob/master/templates/neo4j-advanced-rag/neo4j_advanced_rag

def create_chunks_from_source(source_node_instance: SourceNode, graph, embedding_dimension, embeddings):
    """
    Create Chunk nodes from Source Node
    """
    # get Document from Source Node
    document = source_node_instance.langchain_document

    # Define the splitters
    parent_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    child_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=24)

    # Split into parent chunks
    parent_chunks = parent_splitter.split_documents(document)
    # Add parent chunks to source node for later use
    source_node_instance.langchain_parent_chunks = parent_chunks

    # Added tqdm progress bar
    for i, parent in tqdm(enumerate(parent_chunks), total=len(parent_chunks), desc="Processing Parent Chunks"):
        # Add doc number to metadata
        parent_chunks[i].metadata["number"] = i

        # Generate a hash for the document based on the content
        hasher = hashlib.sha256()
        hasher.update(parent.page_content.encode('utf-8'))
        parent_hash = hasher.hexdigest()
        parent_chunks[i].metadata["hash"] = parent_hash

        # Generate a unique parent ID using hash and number
        unique_parent_id = f"{source_node_instance.hash}_{i}"

        # Split parent into child chunks
        child_chunks = child_splitter.split_documents([parent])
        # Add child chunks to source node for later use
        source_node_instance.langchain_child_chunks = child_chunks

        params = {
            "source_document_name": str(source_node_instance.file_name),
            "source_document_hash": source_node_instance.hash,
            "parent_hash": parent_hash,
            "parent_text": parent.page_content,
            "parent_id": unique_parent_id,
            "parent_embedding": embeddings.embed_query(parent.page_content),
            "children": [
                {
                    "text": c.page_content,
                    "id": f"{unique_parent_id}-{ic}",
                    "child_embedding": embeddings.embed_query(c.page_content),
                }
                for ic, c in enumerate(child_chunks)
            ],
        }
        try:
            # Ingest data
            graph.query(
                """
                // Create Parent
                MERGE (p:Parent {id: $parent_id})
                SET p.text = $parent_text,
                    p.source_name = $source_document_name,
                    p.source_hash = $source_document_hash,
                    p.hash = $parent_hash
                WITH p
                CALL db.create.setVectorProperty(p, 'embedding', $parent_embedding)
                YIELD node
                WITH p
    
                // Link to Source Document
                MATCH (d:Document {hash: $source_document_hash}) // Find the source document by hash
                MERGE (p)<-[:HAS_PARENT]-(d)
                WITH p, d
    
                // Create children
                UNWIND $children AS child
                MERGE (c:Child {id: child.id})
                SET c.text = child.text
                MERGE (c)<-[:HAS_CHILD]-(p)
                WITH c, child
                CALL db.create.setVectorProperty(c, 'embedding', child.child_embedding)
                YIELD node
                RETURN count(*)
                """,
                params,
            )
        except ClientError as e:  # something went wrong with the query
            error_message = str(e)
            logger.error("Failed to create chunks: " + error_message)

        # Create vector index for child with name child_index
        try:
            graph.query(
                "CALL db.index.vector.createNodeIndex('child_index', "
                "'Child', 'embedding', $dimension, 'cosine')",
                {"dimension": embedding_dimension},
            )
        except ClientError:  # already exists
            logger.info("Vector index for children already exists")
            pass

        # Create vector index for parents with name parent_index
        try:
            graph.query(
                "CALL db.index.vector.createNodeIndex('parent_index', "
                "'Parent', 'embedding', $dimension, 'cosine')",
                {"dimension": embedding_dimension},
            )
        except ClientError:  # already exists
            logger.info("Vector index for parents already exists")
            pass

        logger.info(f"Split {source_node_instance.file_name} into chunks successfully.")
    return source_node_instance
