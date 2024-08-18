# Structured retrieval (use neighborhood of entities)
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from src.neo4j_functions import create_graph_database_connection
from src.validation.client_auth import ClientAuth
from pydantic.v1 import BaseModel, Field
from typing import List, Dict
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import StdOutCallbackHandler
from src.neo4j_functions import ClientError

# https://medium.com/@pierre.oberholzer/iso-20022-a-ready-to-use-knowledge-graph-9a7955f8ea7b
# From https://github.com/langchain-ai/langchain/blob/master/templates/neo4j-cypher/neo4j_cypher/chain.py
# https://github.com/tomasonjo/blogs/blob/master/llm/enhancing_rag_with_graph.ipynb

class Entities(BaseModel):
    """Identifying entities in the text"""
    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
                    "appear in the text",
    )


class AdvancedRAGChain:
    def __init__(self, current_user: ClientAuth):
        self.current_user = current_user
        self.model = ChatOpenAI(openai_api_key=current_user.openai_key, model="gpt-4-turbo")
        self.graph = create_graph_database_connection(current_user)

    @staticmethod
    def generate_full_text_query(input) -> str:
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    def create_entity_extraction_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an AI specialized in extracting key entities from text."
                    "Your task is to identify and list people, organizations, concepts, and any other relevant "
                    "entities.",
                ),
                (
                    "human",
                    ""
                    "Extract all entities from the input question below"
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
        try:
            self.graph.query(
                "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
        except ClientError:
            pass


        result = ""
        entity_chain = self.create_entity_extraction_chain()
        entities = entity_chain.invoke({"question": question})
        # :!MENTIONS
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

    # Unstructured retrieval
    # https://github.com/neo4j-labs/llm-graph-builder/blob/21f945cdd172d2ad931c89e92edc3bfcb56f5739/backend/src/QA_optimization.py
    def vector_retriever(self):

        retrieval_query1 = """
        WITH node, score, apoc.text.join([ (node)<-[:PART_OF]-(e) | head(labels(e)) + ": "+ e.id],", ") as entities
        MATCH (node)<-[:HAS_PARENT]-(d:Document)
        WITH d, apoc.text.join(collect(node.text + "\n" + entities),"\n----\n") as text, avg(score) as score
        RETURN text, score, {source: d.fileName} AS metadata
        """

        try:
            typical_rag = Neo4jVector.from_existing_index(
                url=self.current_user.uri,
                username=self.current_user.userName,
                password=self.current_user.password,
                database=self.current_user.database,
                embedding=OpenAIEmbeddings(openai_api_key=self.current_user.openai_key),
                index_name="parent_index",
                retrieval_query=retrieval_query1,
            )

            retriever = typical_rag.as_retriever(
                search_kwargs={'k': 3, "score_threshold": 0.5},
                return_source_documents=True
            )

            return retriever

        except Exception as e:
            raise e

    def unstructured_retriever(self, question: str):
        vector_retriever = self.vector_retriever()
        return vector_retriever.invoke(question)

    def combined_retriever(self, question: str):
        #vector_retriever = self.vector_retriever()
        structured_data = self.structured_retriever(question)
        unstructured_data = [el.page_content for el in self.unstructured_retriever(question)]
        final_data = f"""Structured data:
        {structured_data}
        Unstructured data:
        {"#Document ".join(unstructured_data)}
            """
        return final_data

    # Not using this method
    def combined_retriever_dict(self, question: str):
        structured_data = self.structured_retriever(question)
        unstructured_data = [el.page_content for el in self.unstructured_retriever(question)]
        final_data = {
            "structured_data": structured_data,
            "unstructured_data": unstructured_data
        }

        return final_data


# We need to create a QA chain
def create_advanced_rag_chain_with_sources(current_user: ClientAuth):
    retriever = AdvancedRAGChain(current_user)

    template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(openai_api_key=current_user.openai_key, model="gpt-4-turbo")

    # Define the QA chain
    qa_chain = (
            RunnablePassthrough()  # Pass data through without modification
            | prompt  # Format the data using the prompt template
            | model  # Use the model to generate an answer
            | StrOutputParser()  # Parse the model's output to a string
    )
    # "context": lambda d: retriever.combined_retriever(itemgetter("question")(d)),
    # Create the RAG chain
    rag_chain_with_source = RunnableParallel(
        {
            "context": lambda d: retriever.combined_retriever(itemgetter("question")(d)),  # Retrieve context
            "question": itemgetter("question"),  # Pass through the question
        }
    ).assign(answer=qa_chain)  # Assign the answer from the QA chain

    return rag_chain_with_source


def create_advanced_rag_chain(current_user: ClientAuth):
    retriever = AdvancedRAGChain(current_user)

    template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(openai_api_key=current_user.openai_key, model="gpt-4-turbo")

    # Due to weird behavior on the frontend we use .get instead of itemgetter
    # Used to be: "context": lambda d: retriever.combined_retriever(itemgetter("question")(d)),
    # Now: "context": lambda d: retriever.combined_retriever(d.get("question", "")),
    # "question": RunnablePassthrough(),
    qa_chain = (
            RunnableParallel(
                {
                    "context": lambda d: retriever.combined_retriever(itemgetter("question")(d)),
                    "question": itemgetter("question"),
                }
            )
            | prompt
            | model
            | StrOutputParser()
    )
    return qa_chain


# Create multi hop QA chain
# We can use ReAct Agent for this (Reason + Act) agent
# https://towardsdatascience.com/using-langchain-react-agents-for-answering-multi-hop-questions-in-rag-systems-893208c1847e
def multi_hop_qa_agent(current_user):
    # Create custom tool
    retriever = AdvancedRAGChain(current_user)
    rag_chain = create_advanced_rag_chain(current_user)

    class SearchInput(BaseModel):
        question: str = Field(description="should be a question to search for in the database")
    @tool("search_tool", args_schema=SearchInput, return_direct=False)
    def search(question: str) -> str:  # not async
        """Look up things in database."""
        return rag_chain.invoke({"question": question})  # retriever.combined_retriever(question)

    # Create custom tool from advanced_rag_chain_with_sources
    chain = create_advanced_rag_chain(current_user)
    # Chain as tool
    as_tool = chain.as_tool(
        name="search_tool", description="Looks things up in the database based on a question."
    )

    # Pulling the prompt from hub
    prompt_1 = hub.pull("hwchase17/react")

    # Add The Final Answer must come in JSON format. to the prompt. This solved the issue described in the blog post
    # https://github.com/langchain-ai/langchain/issues/1358

    template = '''Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Begin!

    Question: {input}
    Thought:{agent_scratchpad}'''

    prompt = PromptTemplate.from_template(template)

    model = ChatOpenAI(openai_api_key=current_user.openai_key, model="gpt-4-turbo")

    # Use the correct tool reference without quotes and hyphenation
    react_agent = create_react_agent(model, [search], prompt_1)
    handler = StdOutCallbackHandler()
    agent_executor = AgentExecutor(
        agent=react_agent,
        callbacks=[handler],
        tools=[search],
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,  # For debugging
        max_iterations=5  # useful when agent is stuck in a loop
    )

    return agent_executor
