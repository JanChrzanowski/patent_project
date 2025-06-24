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


def create_instruction_vectorstore(
    embedding_function,
    pdf_path: str = "patent_project/data/Instructions.pdf",
    vectorstore_dir_name: str = "instructions_vectorstore"
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



# Final setup for the script
logger.info("Setting up the model_combined module...")

# -------------------------------
# Get API key from .env
# -------------------------------
api_key = get_dotenv_variable("openai_key")
logger.debug(f"API key loaded: {'Yes' if api_key else 'No'}")

# -------------------------------
# Create vectorstore for instructions
# -------------------------------

create_instruction_vectorstore(
    embedding_function=get_embedding_function(api_key))