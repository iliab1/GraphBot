from src.validation.client_auth import ClientAuth
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from src.neo4j_functions import create_graph_database_connection
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import RunnablePassthrough
from typing import List
from pydantic.v1 import BaseModel, Field
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings


def setup_regular_index(current_client: ClientAuth):
    # Vector Search Parent Nodes

    try:
        typical_rag = Neo4jVector.from_existing_index(
            url=current_client.uri,
            username=current_client.userName,
            password=current_client.password,
            database=current_client.database,
            embedding=OpenAIEmbeddings(openai_api_key=current_client.openai_key),
            index_name="parent_index",
            # retrieval_query = retrieval_query,
        )
        return typical_rag

    except Exception as e:
        raise e


def setup_parent_index(current_client: ClientAuth):
    """
    Set up the retrievers for the chatbot
    :param current_client:
    :return:
    """
    # Vector Search Child Nodes and retrieve their Parents
    try:
        parent_query = """
            MATCH (node)<-[:HAS_CHILD]-(parent)
            WITH parent, max(score) AS score // deduplicate parents
            RETURN parent.text AS text, score, {} AS metadata LIMIT 1

            """
        parent_vectorstore = Neo4jVector.from_existing_index(
            url=current_client.uri,
            username=current_client.userName,
            password=current_client.password,
            database=current_client.database,
            embedding=OpenAIEmbeddings(openai_api_key=current_client.openai_key),
            index_name="child_index",
            retrieval_query=parent_query,
        )
        return parent_vectorstore
    except Exception as e:
        raise e


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def typical_rag_chain(current_user: ClientAuth):
    # setup vector index
    retriever = setup_regular_index(current_user).as_retriever()

    template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(openai_api_key=current_user.openai_key, model="gpt-4-turbo")

    chain = (
            RunnableParallel(
                {
                    "context": itemgetter("question") | retriever | format_docs,
                    "question": itemgetter("question"),
                }
            )
            | prompt
            | model
            | StrOutputParser()
    )
    return chain


def typical_rag_chain_with_sources(current_user: ClientAuth):
    retriever = setup_regular_index(current_user).as_retriever()

    template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(openai_api_key=current_user.openai_key, model="gpt-4-turbo")

    chain = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | model
            | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {
            "context": lambda d: retriever.invoke(itemgetter("question")(d)),  # Get the context
            "question": RunnablePassthrough()  # Pass through the question
        }
    ).assign(answer=chain)  # Assign the answer output from the previous chain

    return rag_chain_with_source


def parent_rag_chain(current_user: ClientAuth):
    retriever = setup_parent_index(current_user).as_retriever()

    template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(openai_api_key=current_user.openai_key, model="gpt-4-turbo")

    chain = (
            RunnableParallel(
                {
                    "context": itemgetter("question") | retriever | format_docs,
                    "question": itemgetter("question"),
                }
            )
            | prompt
            | model
            | StrOutputParser()
    )

    return chain


def parent_rag_chain_with_sources(current_user: ClientAuth):
    retriever = setup_parent_index(current_user).as_retriever()

    template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(openai_api_key=current_user.openai_key, model="gpt-4-turbo")

    chain = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | model
            | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {
            "context": lambda d: retriever.invoke(itemgetter("question")(d)),  # Get the context
            "question": RunnablePassthrough()  # Pass through the question
        }
    ).assign(answer=chain)  # Assign the answer output from the previous chain

    return rag_chain_with_source


class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
                    "appear in the text",
    )


# Adapted from
class GraphNeighbourSearch:
    def __init__(self, current_user: ClientAuth):
        self.current_user = current_user
        self.model = ChatOpenAI(openai_api_key=current_user.openai_key, model="gpt-4-turbo")
        self.graph = create_graph_database_connection(current_user)

    def generate_full_text_query(self, input) -> str:
        """
        Generate a full-text search query for a given input string.

        This function constructs a query string suitable for a full-text search.
        It processes the input string by splitting it into words and appending a
        similarity threshold (~2 changed characters) to each word, then combines
        them using the AND operator. Useful for mapping entities from user questions
        to database values, and allows for some misspellings.
        """
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    def create_entity_chain(self):
        self.graph.query(
            "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are extracting organization and person entities from the text.",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following "
                    "input: {question}",
                ),
            ]
        )
        entity_chain = prompt | self.model.with_structured_output(Entities)
        return entity_chain

    def structured_retriever(self, question: str) -> str:
        """
        Collects the neighborhood of entities mentioned
        in the question
        """
        result = ""
        entity_chain = self.create_entity_chain()
        entities = entity_chain.invoke({"question": question})

        for entity in entities.names:
            response = self.graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                  WITH node
                  MATCH (node)-[r]->(neighbor)
                  RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                  UNION ALL
                  WITH node
                  MATCH (node)<-[r]-(neighbor)
                  RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": self.generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])
        return result

    def graph_neighbour_chain(self):
        template = """Answer the question based only on the following context:
            {context}
            Question: {question}
            """
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
                RunnableParallel(
                    {
                        "context": lambda d: self.structured_retriever(itemgetter("question")(d)),
                        "question": itemgetter("question"),
                    }
                )
                | prompt
                | self.model
                | StrOutputParser()
        )
        return chain

    def graph_neighbour_chain_with_sources(self):
        template = """Answer the question based only on the following context:
            {context}
            Question: {question}
            """
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
                prompt
                | self.model
                | StrOutputParser()
        )

        # Define a chain to process the context and question for RAG
        rag_chain_with_source = RunnableParallel(
            {
                "context": lambda d: self.structured_retriever(itemgetter("question")(d)),  # Get the context
                "question": RunnablePassthrough()  # Pass through the question
            }
        ).assign(answer=chain)  # Assign the answer output from the previous chain

        # Return the chain with structured retrieval output
        return rag_chain_with_source


def create_graph_neighbour_chain(current_user: ClientAuth):
    return GraphNeighbourSearch(current_user).graph_neighbour_chain()


def create_graph_neighbour_chain_with_sources(current_user: ClientAuth):
    return GraphNeighbourSearch(current_user).graph_neighbour_chain_with_sources()
