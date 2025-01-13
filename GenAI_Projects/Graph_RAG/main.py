# import os
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain.prompts import PromptTemplate
# from langchain_community.vectorstores import Neo4jVector
# from langchain_community.chat_models import ChatOpenAI
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.graphs import Neo4jGraph
# from langchain_experimental.graph_transformers import LLMGraphTransformer
# from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
# import streamlit as st
# import tempfile
# from neo4j import GraphDatabase

# def main():
#     st.set_page_config(
#         layout="wide",
#         page_title="Graphy v1",
#         page_icon=":graph:"
#     )
#     st.sidebar.image('logo.png', use_column_width=True) 
#     with st.sidebar.expander("Expand Me"):
#         st.markdown("""
#     This application allows you to upload a PDF file, extract its content into a Neo4j graph database, and perform queries using natural language.
#     It leverages LangChain and OpenAI's GPT models to generate Cypher queries that interact with the Neo4j database in real-time.
#     """)
#     st.title("Graphy: Realtime GraphRAG App")

#     # Load environment variables
#     load_dotenv()
    
#     # Get credentials from environment variables
#     api_key = os.getenv("API_KEY")
#     embed_model = os.getenv("EMBED_MODEL")
#     model = os.getenv("MODEL")
#     organization = os.getenv("ORGANIZATION")
#     anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
#     anthropic_model = os.getenv("MODEL_NAME")
#     neo4j_url = os.getenv('NEO4J_URL')
#     neo4j_username = os.getenv('NEO4J_USERNAME')
#     neo4j_password = os.getenv('NEO4J_PASSWORD')

#     # Validate required credentials
#     if not all([api_key, neo4j_url, neo4j_username, neo4j_password]):
#         st.error("Missing required environment variables. Please check your .env file.")
#         return

#     # Initialize OpenAI components
#     if 'embeddings' not in st.session_state:
#         # Initialize embeddings
#         embed = OpenAIEmbeddings(
#             api_key=api_key,
#             model=embed_model,
#             organization=organization
#         )

#         # Initialize LLM
#         chat_llm = ChatOpenAI(
#             api_key=api_key,
#             organization=organization, 
#             temperature=0
#         )
        
#         st.session_state['embeddings'] = embed
#         st.session_state['llm'] = chat_llm
#     else:
#         embed = st.session_state['embeddings']
#         chat_llm = st.session_state['llm']

#     # Initialize Neo4j connection
#     if 'neo4j_connected' not in st.session_state:
#         try:
#             graph = Neo4jGraph(
#                 url=neo4j_url, 
#                 username=neo4j_username, 
#                 password=neo4j_password
#             )
#             st.session_state['graph'] = graph
#             st.session_state['neo4j_connected'] = True
#             st.sidebar.success("Connected to Neo4j database.")
#         except Exception as e:
#             st.error(f"Failed to connect to Neo4j: {e}")
#             return
#     else:
#         graph = st.session_state['graph']

#     # File uploader
#     uploaded_file = st.file_uploader("Please select a PDF file.", type="pdf")

#     if uploaded_file is not None and 'qa' not in st.session_state:
#         with st.spinner("Processing the PDF..."):
#             # Save uploaded file to temporary file
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#                 tmp_file.write(uploaded_file.read())
#                 tmp_file_path = tmp_file.name

#             # Load and split the PDF
#             loader = PyPDFLoader(tmp_file_path)
#             pages = loader.load_and_split()

#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
#             docs = text_splitter.split_documents(pages)

#             lc_docs = []
#             for doc in docs:
#                 lc_docs.append(Document(page_content=doc.page_content.replace("\n", ""), 
#                 metadata={'source': uploaded_file.name}))

#             # Clear the graph database
#             cypher = """
#               MATCH (n)
#               DETACH DELETE n;
#             """
#             graph.query(cypher)

#             # Define allowed nodes and relationships
#             allowed_nodes = ["Patient", "Disease", "Medication", "Test", "Symptom", "Doctor"]
#             allowed_relationships = ["HAS_DISEASE", "TAKES_MEDICATION", "UNDERWENT_TEST", "HAS_SYMPTOM", "TREATED_BY"]

#             # Transform documents into graph documents
#             transformer = LLMGraphTransformer(
#                 llm=chat_llm,
#                 allowed_nodes=allowed_nodes,
#                 allowed_relationships=allowed_relationships,
#                 node_properties=False, 
#                 relationship_properties=False
#             ) 

#             graph_documents = transformer.convert_to_graph_documents(lc_docs)
#             graph.add_graph_documents(graph_documents, include_source=True)

#             # Create vector index
#             index = Neo4jVector.from_existing_graph(
#                 embedding=embed,
#                 url=neo4j_url,
#                 username=neo4j_username,
#                 password=neo4j_password,
#                 database="neo4j",
#                 node_label="Patient",
#                 text_node_properties=["id", "text"], 
#                 embedding_node_property="embedding", 
#                 index_name="vector_index", 
#                 keyword_index_name="entity_index", 
#                 search_type="hybrid" 
#             )

#             st.success(f"{uploaded_file.name} preparation is complete.")

#             # Retrieve the graph schema
#             schema = graph.get_schema

#             # Set up the QA chain
#             template = """
#             Task: Generate a Cypher statement to query the graph database.

#             Instructions:
#             Use only relationship types and properties provided in schema.
#             Do not use other relationship types or properties that are not provided.

#             schema:
#             {schema}

#             Note: Do not include explanations or apologies in your answers.
#             Do not answer questions that ask anything other than creating Cypher statements.
#             Do not include any text other than generated Cypher statements.

#             Question: {question}""" 

#             question_prompt = PromptTemplate(
#                 template=template, 
#                 input_variables=["schema", "question"] 
#             )

#             qa = GraphCypherQAChain.from_llm(
#                 llm=chat_llm,
#                 graph=graph,
#                 cypher_prompt=question_prompt,
#                 verbose=True,
#                 allow_dangerous_requests=True
#             )
#             st.session_state['qa'] = qa

#     if 'qa' in st.session_state:
#         st.subheader("Ask a Question")
#         with st.form(key='question_form'):
#             question = st.text_input("Enter your question:")
#             submit_button = st.form_submit_button(label='Submit')

#         if submit_button and question:
#             with st.spinner("Generating answer..."):
#                 res = st.session_state['qa'].invoke({"query": question})
#                 st.write("\n**Answer:**\n" + res['result'])

# if __name__ == "__main__":
#     main()


import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
import streamlit as st
import tempfile
from langchain_neo4j import Neo4jGraph

def main():
    st.set_page_config(
        layout="wide",
        page_title="Graphy v1",
        page_icon=":graph:"
    )
    
    # Updated parameter from use_column_width to use_container_width
    st.sidebar.image('logo.png', use_container_width=True)
    
    with st.sidebar.expander("Expand Me"):
        st.markdown("""
    This application allows you to upload a PDF file, extract its content into a Neo4j graph database, and perform queries using natural language.
    It leverages LangChain and OpenAI's GPT models to generate Cypher queries that interact with the Neo4j database in real-time.
    """)
    st.title("Graphy: Realtime GraphRAG App")

    # Load environment variables
    load_dotenv()
    
    # Get credentials from environment variables
    api_key = os.getenv("API_KEY")
    embed_model = os.getenv("EMBED_MODEL")
    model = os.getenv("MODEL")
    organization = os.getenv("ORGANIZATION")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model = os.getenv("MODEL_NAME")
    neo4j_url = os.getenv('NEO4J_URL')
    print(neo4j_url)
    neo4j_username = os.getenv('NEO4J_USERNAME')
    neo4j_password = os.getenv('NEO4J_PASSWORD')

    # Validate required credentials
    if not all([api_key, neo4j_url, neo4j_username, neo4j_password]):
        st.error("Missing required environment variables. Please check your .env file.")
        return

    # Initialize OpenAI components
    if 'embeddings' not in st.session_state:
        # Initialize embeddings with updated class
        embed = OpenAIEmbeddings(
            api_key=api_key,
            model=embed_model,
            organization=organization
        )

        # Initialize LLM with updated class
        chat_llm = ChatOpenAI(
            api_key=api_key,
            organization=organization, 
            temperature=0
        )
        
        st.session_state['embeddings'] = embed
        st.session_state['llm'] = chat_llm
    else:
        embed = st.session_state['embeddings']
        chat_llm = st.session_state['llm']

    # # Initialize Neo4j connection with updated class
    # if 'neo4j_connected' not in st.session_state:
    #     try:
    #         graph = Neo4jGraph(
    #             url=neo4j_url, 
    #             username=neo4j_username, 
    #             password=neo4j_password
    #         )
    #         print(neo4j_url)
    #         st.session_state['graph'] = graph
    #         st.session_state['neo4j_connected'] = True
    #         st.sidebar.success("Connected to Neo4j database.")
    #     except Exception as e:
    #         st.error(f"Failed to connect to Neo4j: {e}")
    #         return
    # else:
    #     graph = st.session_state['graph']
    
    
    if 'neo4j_connected' not in st.session_state:
        try:
            # For Neo4j Aura, make sure the URL is in the correct format
            if not neo4j_url.startswith(('neo4j+s://', 'neo4j+ssc://')):
                neo4j_url = f"neo4j+s://{neo4j_url}"
            
            st.info(f"Attempting to connect to Neo4j at {neo4j_url}")
            
            graph = Neo4jGraph(
                url=neo4j_url, 
                username=neo4j_username, 
                password=neo4j_password,
                database="neo4j"  # Explicitly specify the database
            )
            
            # Test the connection with a simple query
            test_query = "RETURN 1 as test"
            try:
                graph.query(test_query)
                st.session_state['graph'] = graph
                st.session_state['neo4j_connected'] = True
                st.sidebar.success("Connected to Neo4j database successfully!")
            except Exception as query_error:
                st.error(f"Connected but failed to execute test query: {str(query_error)}")
                return
                
        except Exception as e:
            st.error(f"""
            Failed to connect to Neo4j. Please check:
            1. Your Neo4j URL format (should be neo4j+s://xxxxx.databases.neo4j.io)
            2. Your username and password
            3. That your IP is whitelisted in Neo4j Aura
            
            Error: {str(e)}
            """)
            return
    else:
        graph = st.session_state['graph']

    # File uploader
    uploaded_file = st.file_uploader("Please select a PDF file.", type="pdf")

    if uploaded_file is not None and 'qa' not in st.session_state:
        with st.spinner("Processing the PDF..."):
            # Save uploaded file to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Load and split the PDF
            loader = PyPDFLoader(tmp_file_path)
            pages = loader.load_and_split()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
            docs = text_splitter.split_documents(pages)

            lc_docs = []
            for doc in docs:
                lc_docs.append(Document(page_content=doc.page_content.replace("\n", ""), 
                metadata={'source': uploaded_file.name}))

            # Clear the graph database
            cypher = """
              MATCH (n)
              DETACH DELETE n;
            """
            graph.query(cypher)

            # Define allowed nodes and relationships
            allowed_nodes = ["Patient", "Disease", "Medication", "Test", "Symptom", "Doctor"]
            allowed_relationships = ["HAS_DISEASE", "TAKES_MEDICATION", "UNDERWENT_TEST", "HAS_SYMPTOM", "TREATED_BY"]

            # Transform documents into graph documents
            transformer = LLMGraphTransformer(
                llm=chat_llm,
                allowed_nodes=allowed_nodes,
                allowed_relationships=allowed_relationships,
                node_properties=False, 
                relationship_properties=False
            ) 

            graph_documents = transformer.convert_to_graph_documents(lc_docs)
            graph.add_graph_documents(graph_documents, include_source=True)

            # Create vector index
            index = Neo4jVector.from_existing_graph(
                embedding=embed,
                url=neo4j_url,
                username=neo4j_username,
                password=neo4j_password,
                database="neo4j",
                node_label="Patient",
                text_node_properties=["id", "text"], 
                embedding_node_property="embedding", 
                index_name="vector_index", 
                keyword_index_name="entity_index", 
                search_type="hybrid" 
            )

            st.success(f"{uploaded_file.name} preparation is complete.")

            # Retrieve the graph schema
            schema = graph.get_schema

            # Set up the QA chain
            template = """
            Task: Generate a Cypher statement to query the graph database.

            Instructions:
            Use only relationship types and properties provided in schema.
            Do not use other relationship types or properties that are not provided.

            schema:
            {schema}

            Note: Do not include explanations or apologies in your answers.
            Do not answer questions that ask anything other than creating Cypher statements.
            Do not include any text other than generated Cypher statements.

            Question: {question}""" 

            question_prompt = PromptTemplate(
                template=template, 
                input_variables=["schema", "question"] 
            )

            qa = GraphCypherQAChain.from_llm(
                llm=chat_llm,
                graph=graph,
                cypher_prompt=question_prompt,
                verbose=True,
                allow_dangerous_requests=True
            )
            st.session_state['qa'] = qa

    if 'qa' in st.session_state:
        st.subheader("Ask a Question")
        with st.form(key='question_form'):
            question = st.text_input("Enter your question:")
            submit_button = st.form_submit_button(label='Submit')

        if submit_button and question:
            with st.spinner("Generating answer..."):
                res = st.session_state['qa'].invoke({"query": question})
                st.write("\n**Answer:**\n" + res['result'])

if __name__ == "__main__":
    main()