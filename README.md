# GraphBot
Transform unstructured PDF documents into insightful knowledge graphs.

![Python](https://img.shields.io/badge/Python-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-green)
![Langchain](https://img.shields.io/badge/Langchain-blue)
![Neo4j](https://img.shields.io/badge/Neo4j-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-red)
![Docker](https://img.shields.io/badge/Docker-blue)

## Overview
GraphBot is a powerful tool that converts unstructured PDF documents into insightful knowledge graphs. By leveraging advanced graph processing and natural language processing (NLP) techniques, GraphBot enables users to extract meaningful insights, perform complex multi-hop reasoning, and interact with the knowledge graph through intuitive queries.

## Key Features
- **📊 Knowledge Graphs:**  
  Transform unstructured PDF documents into insightful knowledge graphs.

- **💬 Chat with Graph:**  
  Ask questions about the document and receive instant answers.

- **🔍 Multi-hop Reasoning:**  
  Perform complex analyses to uncover deep insights.

- **📄 Graph-Grounding:**  
  Access reliable and precise information grounded in the knowledge graph.

- **☁️ Store Knowledge:**  
  Store your graphs in the Neo4j Aura cloud and access them from anywhere.

## Usage
To get started with GraphBot, follow these steps:

### 1. Log In to GraphBot
When you first launch the application, you will be prompted to enter your credentials.

#### Local Neo4j Setup
If you're running Neo4j in a local container, use the following default credentials:
- **URL:** `bolt://localhost:7687`
- **Username:** `neo4j`
- **Password:** `pleaseletmein`
- **Database:** `neo4j`

#### Cloud Neo4j Setup
To use a cloud database, sign up for a [free Neo4j Aura account](https://neo4j.com/cloud/platform/aura-graph-database/) and use the provided credentials.

#### OpenAI API Key
GraphBot also requires an OpenAI API key. You can create one on the [OpenAI website](https://platform.openai.com/signup).

### 2. Upload a PDF Document
To convert a PDF document into a knowledge graph:
1. Navigate to the `📁 Document Upload` tab.
2. Drag and drop your file.
3. Click the `Upload` button.

You can also explore the entity disambiguation feature by clicking on the `Entity Disambiguation` button.

### 3. View the Knowledge Graph
After uploading the document, view the knowledge graph in the Neo4j browser.

### 4. Chat with the Graph
To interact with the knowledge graph:
1. Go to the `💬 Chat with Graph` tab.
2. Ask questions about the document.

For more complex multi-hop reasoning, use the `🧠 Agent` tab.

## Installation
### 1. Clone the Repository
First, clone the GraphBot repository to your local machine:
```bash
git clone https://github.com/yourusername/graphbot.git
cd graphbot
```
### 2. Build the Application
To build the application using Docker, run the following command:
```bash
docker-compose up -d --build
```
- You also deploy local neo4j instance in a separate container by running the following command:
```bash
docker-compose --profile local_neo4j up -d --build
```
To stop the application, run the following command:
```bash
docker-compose down
```


