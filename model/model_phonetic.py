# Fonetic and semantic similarity model for trademark data
import subprocess
import jellyfish
import shutil
import os


def get_polish_ipa(word, espeak_path=None):
    """ Get the International Phonetic Alphabet (IPA) representation of a Polish word using eSpeak NG."""
    if espeak_path is None:
        espeak_path = shutil.which("espeak-ng")

    if not espeak_path or not os.path.exists(espeak_path):
        raise FileNotFoundError("eSpeak NG executable not found. Please install it or provide the correct path.")
    
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