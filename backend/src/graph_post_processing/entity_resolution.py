from graphdatascience import GraphDataScience
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Neo4jVector
from src.validation.client_auth import ClientAuth
from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import Field as FieldV1
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain.prompts import ChatPromptTemplate
from config.logging_config import logger
from src.neo4j_functions import create_graph_database_connection


def create_entity_embeddings(current_client: ClientAuth):
    """
    Generate the embeddings for the entities based on ID and description
    """
    entity_vector = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(openai_api_key=current_client.openai_key),
        url=current_client.uri,
        username=current_client.userName,
        password=current_client.password,
        database=current_client.database,
        node_label='__Entity__',
        text_node_properties=['id', 'description'],
        embedding_node_property='embedding',
        keyword_index_name='entity_index'
    )

    return entity_vector


def find_similar_entities(graph, current_client: ClientAuth):
    # Initialize the GraphDataScience object
    gds = GraphDataScience(
        current_client.uri,
        database=current_client.database,
        auth=(current_client.userName, current_client.password)
    )

    # Project entities and their embeddings
    gds.graph.drop("entities")

    G, result = gds.graph.project(
        "entities",  # Graph name
        "__Entity__",  # Node projection
        "*",  # Relationship projection
        nodeProperties=["embedding"]  # Configuration parameters
    )

    # Run the k-Nearest Neighbors algorithm to find similar entities
    similarity_threshold = 0.85
    gds.knn.mutate(
        G,
        nodeProperties=['embedding'],
        mutateRelationshipType='SIMILAR',
        mutateProperty='score',
        similarityCutoff=similarity_threshold
    )

    # Write the results back to the graph
    gds.wcc.write(
        G,
        writeProperty="wcc",
        relationshipTypes=["SIMILAR"]
    )

    # Run fuzzy matching instead (maybe not)
    word_edit_distance = 4
    potential_duplicate_candidates = graph.query(
        """MATCH (e:`__Entity__`)
        WHERE size(e.id) > 3 // longer than 3 characters
        WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
        WHERE count > 1
        UNWIND nodes AS node
        // Add text distance
        WITH distinct
          [n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance 
                      OR node.id CONTAINS n.id | n.id] AS intermediate_results
        WHERE size(intermediate_results) > 1
        WITH collect(intermediate_results) AS results
        // combine groups together if they share elements
        UNWIND range(0, size(results)-1, 1) as index
        WITH results, index, results[index] as result
        WITH apoc.coll.sort(reduce(acc = result, index2 IN range(0, size(results)-1, 1) |
                CASE WHEN index <> index2 AND
                    size(apoc.coll.intersection(acc, results[index2])) > 0
                    THEN apoc.coll.union(acc, results[index2])
                    ELSE acc
                END
        )) as combinedResult
        WITH distinct(combinedResult) as combinedResult
        // extra filtering
        WITH collect(combinedResult) as allCombinedResults
        UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
        WITH allCombinedResults[combinedResultIndex] as combinedResult, combinedResultIndex, allCombinedResults
        WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
            WHERE x <> combinedResultIndex
            AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
        )
        RETURN combinedResult
        """, params={'distance': word_edit_distance})

    return potential_duplicate_candidates


# LLM prompt for guiding the final decision on merging nodes
system_prompt = """You are a data processing assistant. Your task is to identify duplicate entities in a list and decide which of them should be merged.
The entities might be slightly different in format or content, but essentially refer to the same thing. Use your analytical skills to determine duplicates.

Here are the rules for identifying duplicates:
1. Entities with minor typographical differences should be considered duplicates.
2. Entities with different formats but the same content should be considered duplicates.
3. Entities that refer to the same real-world object or concept, even if described differently, should be considered duplicates.
4. If it refers to different numbers, dates, or products, do not merge results
"""

user_template = """
Here is the list of entities to process:
{entities}

Please identify duplicates, merge them, and provide the merged list.
"""


class DuplicateEntities(BaseModelV1):
    entities: List[str] = FieldV1(
        description="Entities that represent the same object or real-world entity and should be merged"
    )


class Disambiguate(BaseModelV1):
    merge_entities: Optional[List[DuplicateEntities]] = FieldV1(
        description="Lists of entities that represent the same object or real-world entity and should be merged"
    )


def llm_entity_resolution(entities: List[str], current_client: ClientAuth) -> Optional[List[List[str]]]:
    extraction_llm = ChatOpenAI(
        model_name="gpt-4-turbo",
        openai_api_key=current_client.openai_key
    ).with_structured_output(Disambiguate)

    extraction_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", user_template),
        ]
    )
    extraction_chain = extraction_prompt | extraction_llm

    result = extraction_chain.invoke({"entities": entities})
    if result and result.merge_entities:
        return [el.entities for el in result.merge_entities]
    return []


def perform_entity_resolution(current_client: ClientAuth):
    try:
        graph = create_graph_database_connection(current_client)
    except Exception as e:
        logger.error(f"Failed to create graph database connection: {e}")
        return None

    try:
        create_entity_embeddings(current_client)
        logger.info("Generated embeddings for nodes")
    except Exception as e:
        logger.error(f"Failed to create entity embeddings: {e}")
        return None

    try:
        potential_duplicate_candidates = find_similar_entities(graph, current_client)
        logger.info(f"Found potential duplicates: {potential_duplicate_candidates}")
    except Exception as e:
        logger.error(f"Failed to find similar entities: {e}")
        return None

    if potential_duplicate_candidates:
        try:
            logger.info("Found potential duplicate candidates")
            duplicate_entities = llm_entity_resolution(potential_duplicate_candidates, current_client)
            logger.info("Performed entity resolution")

            # Parallelize the entity resolution process
            merged_entities = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(llm_entity_resolution, el['combinedResult'], current_client)
                    for el in potential_duplicate_candidates
                ]

                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing documents"):
                    try:
                        to_merge = future.result()
                        if to_merge:
                            merged_entities.extend(to_merge)
                    except Exception as e:
                        logger.error(f"Error processing future: {e}")

            # Merge the entities in the graph
            graph.query("""
            UNWIND $data AS candidates
            CALL {
              WITH candidates
              MATCH (e:__Entity__) WHERE e.id IN candidates
              RETURN collect(e) AS nodes
            }
            CALL apoc.refactor.mergeNodes(nodes, {properties: {
                description:'combine',
                `.*`: 'discard'
            }})
            YIELD node
            RETURN count(*)
            """, params={"data": merged_entities})

            logger.info("Merged duplicated entities")
            return duplicate_entities
        except Exception as e:
            logger.error(f"Failed to merge duplicated entities: {e}")
            return None
    else:
        logger.info("No potential duplicate candidates found")
        return None
