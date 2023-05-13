import argparse
import json
import openai
from babydragon.apps.auto_perspective.perspective import PerspectivePromptGenerator

from babydragon.chat.chat import Chat
from babydragon.utils.multithreading import RateLimitedThreadPoolExecutor
from concurrent.futures import as_completed
from time import perf_counter
import json
import time
from babydragon.memory.indexes.memory_index import MemoryIndex

import numpy as np
import hdbscan
import umap
from babydragon.memory.indexes.memory_index import MemoryIndex
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from babydragon.models.embedders.cohere import CohereEmbedder
from babydragon.models.generators.cohere import cohere_summarize

def main():
    openai.api_key =  "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"

    base_filename="prompt_"

    subjects=[
        "Sustainability-Focused",
        "Technological Optimism",
        "Existential Risk Theory",
        "Social Constructivism",
        "Cosmological Anthropic Principle",
        "Bioconservatism",
        "Biopsychosocial Model",
        "Economic Libertarianism",
        "Techno-Social Interactionism",
        "Deep Ecology",
        "Humanistic Psychology",
        "Technological Determinism",
        "Techno-Utopianism",
        "Transhumanism",
        "Positive Psychology",
        "Quantum Mysticism",
        "Physical Realism",
        "Critical Theory",
        "Media Ecology",
        "Anthropocentrism",
    ]

    perspectives=[
        "Artificial Intelligence",
        "Climate Change",
        "Quantum Computing",
        "Space Exploration",
        "Genetic Engineering",
        "Cryptocurrency",
        "Biodiversity",
        "Mental Health",
        "Social Media",
        "Education",
        "Renewable Energy",
        "Virtual Reality",
        "Internet Privacy",
        "Globalization",
        "Autonomous Vehicles",
        "Urban Planning",
        "Food Security",
        "Cultural Diversity",
        "Cybersecurity",
        "Aging Population",
    ]
    dataset_url = "Cohere/wikipedia-22-12-simple-embeddings"
    start = perf_counter()
    try:
        index = MemoryIndex(name="wiki_index", load=True, is_batched=True,embedder=CohereEmbedder)
        if len(index.values)>0:
            loaded = True
        else:
            loaded = False
    except:
        loaded = False

    if not loaded:
        print("Index not found, creating new index")
        index = MemoryIndex.from_hf_dataset(dataset_url, ["title", "text"],embeddings_column= "emb", name="wiki_index", is_batched=True,embedder=CohereEmbedder)
    end = perf_counter()
    print("Time to index: ", end - start)
    #delete the index to save memory
    del index
    generator = PerspectivePromptGenerator(subjects, perspectives, max_workers=12, calls_per_minute=4)
    prompts = generator.generate_prompts()

if __name__ == "__main__":
    main()
