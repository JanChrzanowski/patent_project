# Fonetic and semantic similarity model for trademark data


# Artificial Intelligence liberaries
#import transformers
#import torch
#import tensorflow as tf

# Data processing libraries
import pickle
import numpy as np
import pandas as pd

# Os and Path libraries
import subprocess
from dotenv import load_dotenv
from pathlib import Path
import os


base_dir = Path(__file__).parent
filepath = base_dir.parent / 'data' / 'trademark_dataZT20-2024-04-01.pkl'

df = pd.read_pickle(filepath)
