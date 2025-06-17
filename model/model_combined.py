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

# Loand local models
#from model_sense import trademark_similarity
#from model_phonetic import phonetic_similarity

warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
espeak_path = "C:\Program Files\eSpeak NG\espeak-ng.exe" # This should be the path to your eSpeak NG executable if its not in PATH


def get_dotenv_variable(variable_name: str) -> str:
    """Retrieve a variable from the .env file."""
    if load_dotenv() is True:
        print("Environment variables loaded successfully.")
        value = os.getenv(variable_name)
        if value is None:
            raise EnvironmentError(f"Environment variable '{variable_name}' not found in .env file.")
        return value
    else:
        print("Failed to load environment variables. Please check your .env file.")
        return None
    

def connect_to_model(api_key: str, model: str = "gpt-4o-mini") -> ChatOpenAI:
    """Connect to the OpenAI API using the provided API key."""
    return ChatOpenAI(api_key=api_key, model=model)


def get_embedding_function(api_key: str, model: str = "text-embedding-ada-002") -> OpenAIEmbeddings:
    """Return an OpenAIEmbeddings object, which is used to create vector embeddings from text."""
    embeddings = OpenAIEmbeddings(
        model=model, openai_api_key=api_key
    )
    return embeddings
