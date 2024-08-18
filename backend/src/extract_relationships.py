from src.validation.source_node import SourceNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from config.logging_config import logger
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
from tqdm import tqdm

# This module contains functions for extracting entities and relationships from the source node.
# It uses Langchain llm_transformer to convert the text chunks into graph documents and add them to the source node.
# From https://neo4j.com/developer-blog/global-graphrag-neo4j-langchain/


system_prompt = (
    "# Knowledge Graph Instructions for GPT-4\n"
    "## 1. Overview\n"
    "You are a top-tier algorithm designed for extracting information in structured "
    "formats to build a knowledge graph.\n"
    " Your primary goal is to capture as much information from the text as possible without "
    "sacrificing accuracy. Do not add any information that is not explicitly "
    "mentioned in the text.\n"
    "- **Nodes** represent entities and concepts.\n"
    "- The aim is to achieve simplicity and clarity in the knowledge graph, making it\n"
    "accessible for a vast audience.\n"
    "## 2. Labeling Nodes\n"
    "- **Consistency**: Ensure you use available types for node labels.\n"
    "Ensure you use basic or elementary types for node labels.\n"
    "- For example, when you identify an entity representing a person, "
    "always label it as **'person'**. Avoid using more specific terms "
    "like 'mathematician' or 'scientist'."
    "- **Node IDs**: Never utilize integers as node IDs. Node IDs should be "
    "names or human-readable identifiers found in the text.\n"
    "- **Relationships** represent connections between entities or concepts.\n"
    "Ensure consistency and generality in relationship types when constructing "
    "knowledge graphs. Instead of using specific and momentary types "
    "such as 'BECAME_PROFESSOR', use more general and timeless relationship types "
    "like 'PROFESSOR'. Make sure to use general and timeless relationship types!\n"
    "## 3. Coreference Resolution\n"
    "- **Maintain Entity Consistency**: When extracting entities, it's vital to "
    "ensure consistency.\n"
    'If an entity, such as "John Doe", is mentioned multiple times in the text '
    'but is referred to by different names or pronouns (e.g., "Joe", "he"),'
    "always use the most complete identifier for that entity throughout the "
    'knowledge graph. In this example, use "John Doe" as the entity ID.\n'
    "Remember, the knowledge graph should be coherent and easily understandable, "
    "so maintaining consistency in entity references is crucial.\n"
    "## 4. Node descriptions\n"
    "- **Description as property**: Description should be incorporated as attribute with a key 'description'"
    "## 4. Strict Compliance\n"
    "Adhere to the rules strictly. Non-compliance will result in termination."
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        (
            "human",
            (
                "Tip: Make sure to answer in the correct format and do "
                "not include any explanations. "
                "Tip: Analyse everything after the 'following input'."
                "Use the given format to extract information from the "
                "following input: {input}"
            ),
        ),
    ]
)


def create_graph_documents(source_node_instance: SourceNode, llm):
    """
    Create graph documents from parent documents using the LLMGraphTransformer.

    Args:
        source_node_instance (SourceNode): The source node object containing parent chunks.
        llm (object): The LLM model object.

    Returns:
        SourceNode: The updated source node instance with added graph documents.
    """
    logger.info(f"Creating graph documents for source node {source_node_instance.file_name}")

    parent_chunks = source_node_instance.langchain_parent_chunks

    # Pre-process chunks
    for chunk in parent_chunks:
        chunk.page_content = chunk.page_content.replace("\n", " ")

    llm_transformer = LLMGraphTransformer(
        llm=llm,
        strict_mode=False,
        # Encourage LLM to generate descriptions for nodes and relationships
        # This would help with entity resolution and RAG
        node_properties=["description"],
        relationship_properties=["description"],
        # Optional parameter for passing custom prompt
        prompt=prompt,  # Optional parameter for passing custom prompt
    )
    futures = []
    graph_document_list = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        for chunk in parent_chunks:
            # Convert each chunk to langchain Document
            chunk_doc = Document(page_content=chunk.page_content.encode("utf-8"), metadata=chunk.metadata)  # metadata=chunk.metadata
            #graph_chunk = llm_transformer.convert_to_graph_documents(parent_chunks)
            # Run the conversion for chunks in parallel
            futures.append(executor.submit(llm_transformer.convert_to_graph_documents, [chunk_doc]))

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            graph_document = future.result()
            graph_document_list.append(graph_document[0])  # Each chunk only has one graph document

    logger.info(f"Graph documents created for source node {source_node_instance.file_name}")


    # Add hash as a property to nodes for easy reference to the parent document
    for graph_document in graph_document_list:
        parent_hash = graph_document.source.metadata["hash"]
        # Add parent_id to each entity in the graph document
        for entity in graph_document.nodes:
            entity.properties["parent_hash"] = parent_hash

    source_node_instance.langchain_graph_documents = graph_document_list
    source_node_instance.llm = str(llm.model_name)

    # TODO: Display what entities and rels were extracted

    return source_node_instance
