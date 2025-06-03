# Artificial Intelligence libraries
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import cos_sim
from sentence_transformers import InputExample
import torch

# Data processing libraries
import pickle
import numpy as np
import pandas as pd

# Os and Path libraries
import subprocess
from dotenv import load_dotenv
from pathlib import Path
import os



train_examples = [
    InputExample(texts=["Adidas", "Adidos"], label=1.0),
    InputExample(texts=["Nike", "Adidas"], label=0.0),
    InputExample(texts=["Żywiec", "Zywiec"], label=1.0),
    InputExample(texts=["Puma", "Tymbark"], label=0.0),
]


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer("sdadas/st-polish-paraphrase-from-mpnet",device=device)


def trademark_similarity(name1, name2):
    results = model.encode([name1, name2], convert_to_tensor=True, show_progress_bar=False)
    similarity = util.cos_sim(results[0], results[1]).item()
    return round(similarity, 3)




print(trademark_similarity('Kowalski', 'Chrzanowski'))

# base_dir = Path(__file__).parent
# filepath = base_dir.parent / 'data' / 'trademark_dataZT20-2024-04-01.pkl'

# df = pd.read_pickle(filepath)

# word_to_compare = df.loc[0]
# trademark = word_to_compare['WordMark']
# category = [str(c) for c in word_to_compare['ClassNumbers']]

# filtered_df = df[df['ClassNumbers'].apply(lambda x: any(str(c) in x for c in category))]
# list = filtered_df['WordMark'].tolist()

# for word in list: 
#     score = trademark_similarity(trademark, word)
#     print(f"Similarity between '{trademark}' and '{word}': {score}")

