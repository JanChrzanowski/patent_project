# Fonetic and semantic similarity model for trademark data
import subprocess
import jellyfish

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


def get_polish_ipa(word):
    """ Get the International Phonetic Alphabet (IPA) representation of a Polish word using eSpeak NG."""
    espeak_path = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
    cmd = [espeak_path, "-v", "pl", "--ipa=3", "-q", word]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    ipa = result.stdout.strip().replace("ˈ", "").replace(" ", "")
    return ipa


def phonetic_similarity(word1, word2):

    """ Calculate the phonetic similarity between two Polish words using their IPA representations."""
    
    ipa1 = get_polish_ipa(word1)
    ipa2 = get_polish_ipa(word2)
    dist = jellyfish.levenshtein_distance(ipa1, ipa2)
    max_len = max(len(ipa1), len(ipa2))
    similarity = 1 - dist / max_len 
    return similarity, ipa1, ipa2
