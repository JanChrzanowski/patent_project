# Import necessary libraries langchain, vectorstores, and other dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from pypdf import PdfReader
from typing import List
from langchain.schema import Document
from langchain_core._api.deprecation import LangChainDeprecationWarning
from sklearn.metrics.pairwise import cosine_similarity

# Phonetic and semantic similarity function
from model_phonetic import phonetic_similarity
from model_sense import trademark_similarity

# Other modules and packages
import os
import tempfile
import pandas as pd
from dotenv import load_dotenv
import warnings
import shutil
import logging

# ----------------------------------------
# Determine path to the parent directory
# ----------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))   
parent_dir = os.path.dirname(current_dir)                  
logs_dir = os.path.join(parent_dir, "logs")             

# Create the logs directory if it doesn't exist
os.makedirs(logs_dir, exist_ok=True)

# Define the full path to the log file
log_path = os.path.join(logs_dir, "app.log")

# ----------------------------------------
# Logging configuration

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_path),     
        logging.StreamHandler()            
    ]
)
logger = logging.getLogger(__name__)


# Ignore deprecation warnings
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)


# Locate espeak-ng.exe
espeak_path = shutil.which("espeak-ng")
if espeak_path:
    logger.info(f"Found espeak-ng at: {espeak_path}")
else:
    logger.warning("Did not find espeak-ng. Please install it or provide the correct path.")

# -------------------------------
# Load environment variable
# -------------------------------
def get_dotenv_variable(variable_name: str) -> str:
    """Retrieve a variable from the .env file."""
    if load_dotenv():
        logger.info("Environment variables loaded successfully.")
        value = os.getenv(variable_name)
        if value is None:
            logger.error(f"Environment variable '{variable_name}' not found in .env file.")
            raise EnvironmentError(f"Environment variable '{variable_name}' not found in .env file.")
        logger.debug(f"Retrieved environment variable: {variable_name}")
        return value
    else:
        logger.error("Failed to load environment variables. Please check your .env file.")
        return None

# -------------------------------
# Connect to OpenAI Chat Model
# -------------------------------
def connect_to_model(api_key: str, model: str = "gpt-4o-mini") -> ChatOpenAI:
    """Connect to the OpenAI API using the provided API key."""
    logger.info(f"Connecting to OpenAI model: {model}")
    return ChatOpenAI(api_key=api_key, model=model)

# -------------------------------
# Create embeddings
# -------------------------------
def get_embedding_function(api_key: str, model: str = "text-embedding-ada-002") -> OpenAIEmbeddings:
    """Return an OpenAIEmbeddings object, which is used to create vector embeddings from text."""
    logger.info(f"Creating embedding function with model: {model}")
    embeddings = OpenAIEmbeddings(
        model=model, openai_api_key=api_key
    )
    return embeddings


# -------------------------------
# Create vectorstore from PDF instructions
# -------------------------------
def create_instruction_vectorstore(
    embedding_function,
    pdf_path: str ,
    vectorstore_dir_name: str
):
    """
    Creates a vectorstore from a given PDF file if it does not already exist.
    The vectorstore is stored in a folder located one level above this script.
    """
    try:
        # Determine vectorstore path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        vectorstore_path = os.path.join(parent_dir, vectorstore_dir_name)

        # Check if the vectorstore already exists
        if os.path.exists(vectorstore_path) and os.path.isdir(vectorstore_path) and os.listdir(vectorstore_path):
            logger.info(f"Vectorstore already exists at: {vectorstore_path}. Skipping creation.")
            return

        logger.info(f"Creating vectorstore at: {vectorstore_path} from PDF: {pdf_path}")

        # Load the PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        logger.info(f"Loaded {len(pages)} page(s) from PDF.")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " "]
        )
        chunks = text_splitter.split_documents(pages)
        logger.info(f"Split document into {len(chunks)} chunk(s).")

        # Create vectorstore and persist it
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=vectorstore_path
        )
        vectorstore.persist()
        logger.info(f"Vectorstore successfully created and persisted at: {vectorstore_path}")

    except Exception as e:
        logger.exception(f"Failed to create vectorstore: {e}")

# -------------------------------
# Load the vectorstore from the specified directory
# -------------------------------       

def load_instruction_vectorstore(
    embedding_function,
    vectorstore_dir_name: str
):
    """
    Loads an existing Chroma vectorstore from a folder located one level above this script.
    Returns the vectorstore object, or None if not found or empty.
    """
    try:
        # Determine full path to the vectorstore
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        vectorstore_path = os.path.join(parent_dir, vectorstore_dir_name)

        # Check if the directory exists and contains files
        if not os.path.exists(vectorstore_path) or not os.listdir(vectorstore_path):
            logger.warning(f"Vectorstore directory not found or empty: {vectorstore_path}")
            return None

        # Load the vectorstore
        vectorstore = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=embedding_function
        )
        logger.info(f"Successfully loaded vectorstore from: {vectorstore_path}")
        return vectorstore

    except Exception as e:
        logger.exception(f"Failed to load vectorstore: {e}")
        return None
    

def query_instruction_vectorstore(query: str, embedding_function, vectorstore_dir_name, k: int = 3):
    """
    Loads the instruction vectorstore and performs a similarity search for the given query.
    
    Parameters:
        query (str): The query string.
        embedding_function: Embedding function used to load the vectorstore.
        k (int): Number of top results to return.
    
    Returns:
        List of strings: The content of relevant document chunks.
    """
    instructionstore = load_instruction_vectorstore(embedding_function, vectorstore_dir_name)

    if instructionstore is None:
        logger.error("Instruction vectorstore is not available.")
        return []

    try:
        retriever = instructionstore.as_retriever(search_type="similarity")
        relevant_chunks = retriever.invoke(query)
        
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunk(s) for query: \"{query}\"")
        return [chunk.page_content for chunk in relevant_chunks[:k]]
    
    except Exception as e:
        logger.exception(f"Error while querying the vectorstore: {e}")
        return []

# Final setup for the script
logger.info("Setting up the model_combined module...")

# -------------------------------
# Get API key from .env
# -------------------------------
api_key = get_dotenv_variable("openai_key")
logger.debug(f"API key loaded: {'Yes' if api_key else 'No'}")

# -------------------------------
# Create vectorstore for instructions if needed
# -------------------------------
create_instruction_vectorstore(
    embedding_function=get_embedding_function(api_key),
    pdf_path="patent_project/data/Instructions.pdf",
    vectorstore_dir_name="instructions_vectorstore"
)

# Create second vectorstore
create_instruction_vectorstore(
    embedding_function=get_embedding_function(api_key),
    pdf_path="patent_project/data/Instructions_2.pdf",
    vectorstore_dir_name="instructions_vectorstore_2"
)

# -------------------------------
# Finalize setup
# -------------------------------

def compare_trademarks(
    query_name: str,
    comparison_names: list,
    embedding_function,
    api_key: str,
    vectorstore_dir_name: str = "instructions_vectorstore"
):
    logger.info(f"Comparing trademark: '{query_name}' with {len(comparison_names)} other names.")

    phonetic_results = []
    semantic_results = []

    for name in comparison_names:

        try:
            phon_sim, ipa1, ipa2 = phonetic_similarity(query_name, name)
        except Exception as e:
            logger.error(f"Phonetic similarity failed for {name}: {e}")
            phon_sim, ipa1, ipa2 = None, None, None

       
        try:
            sem_sim = trademark_similarity(query_name, name)
        except Exception as e:
            logger.error(f"Semantic similarity failed for {name}: {e}")
            sem_sim = None

        phonetic_results.append((name, phon_sim, ipa1, ipa2))
        semantic_results.append((name, sem_sim))

  
    context_chunks = query_instruction_vectorstore(
        query=f"Jak ocenić podobieństwo znaków towarowych? {query_name} vs {', '.join(comparison_names)}",
        embedding_function=embedding_function,
        vectorstore_dir_name=vectorstore_dir_name
    )
    context = "\n\n".join(context_chunks)


    similarity_report = ""
    for i, name in enumerate(comparison_names):
        similarity_report += (
            f"Porównanie z: {name}\n"
            f"Fonetyczne podobieństwo: {phonetic_results[i][1]} (IPA: {phonetic_results[i][2]} vs {phonetic_results[i][3]})\n"
            f"Znaczeniowe podobieństwo: {semantic_results[i][1]}\n\n"
        )

    PROMPT_TEMPLATE = f"""
        Jesteś ekspertem w dziedzinie znaków towarowych.
        Twoim zadaniem jest określić prawdopodobieństwo w jakim stopniu dwa znaki są do siebie podobne, fonetycznie i znaczeniowo.
        Na wejściu otrzymujesz jeden znak towarowy oraz listę znaków do porównania.
        Uwzględnij kontekst użytkowania znaków.

        Znak towarowy: {query_name}
        Znaki do porównania: {', '.join(comparison_names)}

        Podobieństwa:

        {similarity_report}

        Kontekst (fragmenty z instrukcji):

        {context}

        Na podstawie powyższych informacji odpowiedz:
        """
    
    logger.info("Prompt prepared. Sending to LLM...")

    
    chat = connect_to_model(api_key)
    response = chat.invoke(PROMPT_TEMPLATE)

    logger.info("LLM response received.")
    return response.content


result = compare_trademarks(
    query_name="Mięsny u Olgi",
    comparison_names=["Mięsny u Zosi", "Mięsny u Jadzi", "Mięsny u Zdzisia"],
    embedding_function=get_embedding_function(api_key),
    api_key=api_key,
    vectorstore_dir_name="instructions_vectorstore"
)

print(result)