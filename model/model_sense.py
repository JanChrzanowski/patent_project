# Artificial Intelligence libraries
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import cos_sim
from sentence_transformers import InputExample
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer("sdadas/st-polish-paraphrase-from-mpnet",device=device)


def trademark_similarity(name1, name2):
    results = model.encode([name1, name2], convert_to_tensor=True, show_progress_bar=False)
    similarity = util.cos_sim(results[0], results[1]).item()
    return round(similarity, 3)


