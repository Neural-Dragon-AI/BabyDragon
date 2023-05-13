

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

import tiktoken
from typing import List, Tuple, Dict


class SubjectPerspectiveAnalyzer:
    def __init__(self, chatbot: 'Chat'):
        self.chatbot = chatbot

    def analyze_subject_perspective(self, user_subject: str, user_perspective: str) -> dict:
        prompts = [
            f"Generate ideas and concepts that explore the connection between {user_subject} and {user_perspective}, considering both traditional and unconventional approaches.",
            f"List key concepts or topics that would help analyze {user_subject} through the lens of {user_perspective}, including relevant principles, theories, or models.",
            f"Identify potential areas or research topics where {user_subject} and {user_perspective} intersect, highlighting intriguing or innovative perspectives."
        ]

        output = {}

        with RateLimitedThreadPoolExecutor(max_workers=3, calls_per_minute=20, verbose = False) as executor:
            future_to_prompt = {executor.submit(self._analyze_prompt, prompt): prompt for prompt in prompts}
            for future in as_completed(future_to_prompt):
                prompt = future_to_prompt[future]
                try:
                    output[prompt] = future.result()
                except Exception as exc:
                    counter = 0
                    print(f'An exception occurred while analyzing prompt "{prompt}": {exc}')
                    while counter < 3:
                        try:
                            output[prompt] = self._analyze_prompt(prompt)
                            break
                        except Exception as exc:
                            counter += 1
                            print(f'An exception occurred while analyzing prompt "{prompt}": {exc}')
                    if counter == 3:
                        output[prompt] = []

        return output

    def _analyze_prompt(self, prompt: str) -> list:
        response = self.chatbot.reply(prompt, verbose=False)
        return self._format_response(response)

    def _format_response(self, response: str) -> list:
        formatted_response = response.strip().split('\n')
        return formatted_response
    

class Ideation:
    def __init__(self, memory_index: MemoryIndex):
        self.memory_index = memory_index

    def retrieve_ideas(self, queries: Dict, k: int = 30, max_tokens: int = 10000):
        """
        Generate ideas based on the given list of queries.

        Args:
            queries: The list of queries for generating ideas.
            k: The number of top search results to consider.
            max_tokens: The maximum number of tokens to return.

        Returns:
            A list of ideas generated based on the queries.
        """
        ideas = []
        for key, queries in queries.items():
            for query in queries:
                if query is None or len(query) < 10:
                    continue
                top_k_hints, scores, indices = self.memory_index.token_bound_query(
                    query, k=k, max_tokens=max_tokens
                )
                last_query = self.memory_index.query_history[-1]
                hints_tokens = last_query["hints_tokens"]
                returned_tokens = last_query["returned_tokens"]

                ideas.append({"key_task": key, "query": query, "hints": top_k_hints, "scores": scores, "hints_tokens": hints_tokens, "returned_tokens": returned_tokens})

        return ideas


tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

class IdeaCluster:
    def __init__(self, ideas: list, max_tokens_per_cluster: int):
        self.ideas = ideas
        self.max_tokens_per_cluster = max_tokens_per_cluster
        self.idea_index = self.create_idea_index()
        self.cluster_labels = None

    def create_idea_index(self):
        gathered_docs = []
        for idea in self.ideas:
            for hint in idea["hints"]:
                gathered_docs.append(hint)
        self.gathered_docs = set(gathered_docs)
        idea_index = MemoryIndex(values=self.gathered_docs, is_batched=True, name="ideas")
        return idea_index

    def cluster_embeddings(self, n_neighbors: int = 10, min_cluster_size: int = 5):
        reducer = umap.UMAP(n_neighbors=n_neighbors)
        reduced_embeddings = reducer.fit_transform(self.idea_index.get_all_embeddings())

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(reduced_embeddings)

        token_count_per_cluster = self.count_tokens_per_cluster(labels)
        print(token_count_per_cluster)
        if max(token_count_per_cluster.values()) <= self.max_tokens_per_cluster:
            self.cluster_labels = labels
            print("Clusters created successfully.")
        else:
            print("Clusters exceed the maximum token count.")

    def count_tokens_per_cluster(self, labels):
        token_count_per_cluster = {}

        for label, doc in zip(labels, self.gathered_docs):
            if label not in token_count_per_cluster:
                token_count_per_cluster[label] = len(tokenizer.encode(doc))
            else:
                token_count_per_cluster[label] += len(tokenizer.encode(doc))
        return token_count_per_cluster

    def create_minimum_spanning_paths(self):
        if self.cluster_labels is None:
            raise ValueError("You must run cluster_embeddings() before creating minimum spanning paths.")

        unique_labels = np.unique(self.cluster_labels)
        min_span_paths = []

        for label in unique_labels:


            # Get the indices of the current cluster
            cluster_indices = np.where(self.cluster_labels == label)[0]

            # Calculate the pairwise distances between embeddings in the cluster
            cluster_embeddings = self.idea_index.embeddings[cluster_indices]
            dist_matrix = squareform(pdist(cluster_embeddings))

            # Create a graph from the distance matrix
            graph = nx.from_numpy_array(dist_matrix)

            # Compute the minimum spanning tree of the graph
            min_span_tree = nx.minimum_spanning_tree(graph)

            # Get the minimum spanning paths
            min_span_paths_cluster = []
            visited = set()
            for u, v in min_span_tree.edges():
                if u not in visited and v not in visited:
                    orig_u = cluster_indices[u]
                    orig_v = cluster_indices[v]
                    min_span_paths_cluster.append(orig_u)
                    visited.add(u)
                    visited.add(v)
            # Add the last node to complete the path
            min_span_paths_cluster.append(orig_v)

            min_span_paths.append(min_span_paths_cluster)

        self.min_span_paths = min_span_paths

    
    def plot_embeddings_with_path(self):
        paths = self.min_span_paths
        embeddings = self.idea_index.embeddings
        title = "Minimum Spanning Paths"
        tsne = TSNE(n_components=2, random_state=42)
        reduced_embeddings = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        colors = cm.rainbow(np.linspace(0, 1, len(paths)))
        for i, path in enumerate(paths):
            path_embeddings = reduced_embeddings[path]
            plt.scatter(
                path_embeddings[:, 0],
                path_embeddings[:, 1],
                color=colors[i],
                label=f"Cluster {i}",
            )
            for j in range(len(path) - 1):
                plt.plot(
                    [path_embeddings[j, 0], path_embeddings[j + 1, 0]],
                    [path_embeddings[j, 1], path_embeddings[j + 1, 1]],
                    color=colors[i],
                )
        plt.title(title)
        plt.legend()
        plt.show()
    def get_clustered_ideas(self):
        if self.cluster_labels is None:
            raise ValueError("You must run cluster_embeddings() before getting clustered ideas.")
        
        clustered_ideas = {}
        for label, idea in zip(self.cluster_labels, self.idea_index.values):
            if label not in clustered_ideas:
                clustered_ideas[label] = [idea]
            else:
                clustered_ideas[label].append(idea)
        
        # Convert the dictionary to a list of lists (each list corresponds to a cluster)
        return list(clustered_ideas.values())


class Summarizer:
    def __init__(self, texts: list):
        self.texts = texts

    def summarize_texts(self) -> dict:
        output = {}

        with RateLimitedThreadPoolExecutor(max_workers=12, calls_per_minute=200, verbose = False) as executor:
            future_to_text = {executor.submit(self._summarize_text, text): text for text in self.texts}
            for future in as_completed(future_to_text):
                text = future_to_text[future]
                try:
                    output[text] = future.result()
                except Exception as exc:
                    print(f'An exception occurred while summarizing text: {exc}')

        return output

    def _summarize_text(self, text: str) -> str:
        summary = cohere_summarize(text, model="summarize-xlarge", length="auto", extractiveness="low", format="auto")
        return summary


def generate_perspective_prompt(user_subject, user_perspective, seed_model = "gpt-3.5-turbo"):
    start = perf_counter()
    chat_instance = Chat(model=seed_model)
    analyzer = SubjectPerspectiveAnalyzer(chat_instance)
    output = analyzer.analyze_subject_perspective(user_subject, user_perspective)
    end = perf_counter()
    print("Time to analyze_perspective: ", end - start)

    dataset_url = "Cohere/wikipedia-22-12-simple-embeddings"
    start = perf_counter()
    index = MemoryIndex(name="wiki_index", load=True, is_batched=True,embedder=CohereEmbedder)
    if len(index.values)>0:
        loaded = True
    else:
        loaded = False

    if not loaded:
        print("Index not found, creating new index")
        index = MemoryIndex.from_hf_dataset(dataset_url, ["title", "text"],embeddings_column= "emb", name="wiki_index", is_batched=True,embedder=CohereEmbedder)
    end = perf_counter()
    print("Time to index: ", end - start)

    start = perf_counter()
    ideation = Ideation(memory_index=index)
    ideas = ideation.retrieve_ideas(output, k=40, max_tokens=10000)
    token_count = 0
    for idea in ideas:
        token_count += idea["returned_tokens"]
    end = perf_counter()
    print("Time to retrieve ideas: ", end - start)
    print("Number of tokens: ", token_count)

    start = perf_counter()
    max_tokens_per_cluster = 20000
    idea_cluster = IdeaCluster(ideas, max_tokens_per_cluster)
    idea_cluster.cluster_embeddings()
    time.sleep(0.5)
    ideas = idea_cluster.get_clustered_ideas()
    end = perf_counter()
    combined_idea = []
    ideas_tokens = []
    for idea in ideas:
        combined_idea.append("\n".join(idea))
        ideas_tokens.append(tokenizer.encode(combined_idea[-1]))
    print("Time to cluster ideas: ", end - start)
    print("Number of clusters: ", len(ideas))
    print("Number of tokens in each cluster: ", [len(tokens) for tokens in ideas_tokens])
    start = perf_counter()
    summarizer = Summarizer(combined_idea)
    summaries = summarizer.summarize_texts()
    end = perf_counter()
    print("Time to summarize: ", end - start)
    print("Number of summaries: ", len(summaries))
    print("Number of tokens in each summary: ", [len(tokenizer.encode(summary)) for summary in summaries.values()])
    start = perf_counter()
    system_prompt = f"""With the summarized information from Cohere about the essential ideas, concenpts, priciples, and intersection points between {user_subject} and {user_perspective}, construct an appealing and context-aware chatbot prompt that impels the chatbot to respond with insights and perspectives born from the synergy of these two domains. Use the tips below to help you create an effective chatbot prompt: 1. Begin with a concise introduction: Initiate the chatbot prompt by setting the context, encompassing the user's specified subject and perspective. 2. Accentuate the intersection: Secure that the chatbot prompt underlines the connection between the user_subject and user_perspective, leading to more pertinent and perceptive responses. 3. Foster exploration: Ensure the chatbot prompt provokes the chatbot to delve into the main ideas, principles, and concepts from the summaries with a thoughtful and reflective approach. 4. Pose open-ended questions: Incorporate open-ended queries in the chatbot prompt, stimulating the chatbot to contemplate beyond the summaries and offer comprehensive responses. 5. Prioritize simplicity: Maintain the chatbot prompt's clarity and brevity, assisting the chatbot in comprehending the context and reacting suitably. 6. Steer the conversation: Craft the chatbot prompt in a manner that subtly directs the chatbot's answers, confirming they consistently focus on the user_subject and user_perspective. By leveraging these tips, develop a chatbot prompt that generates responses illustrative of the user's specified subject and perspective, culminating in a customized and significant interaction. Remember to conclude the prompt with 7 content pillars that will help the chatbot use the perspective at the best of its capacity. """.format(user_subject=user_subject, user_perspective=user_perspective)
    for i, summary in enumerate(summaries.values()):
        system_prompt += summary + "\n\n"
    prompt_generator = Chat(name= "prompt generator",system_prompt=system_prompt, max_output_tokens= 2000, model = "gpt-4")
    perspective_prompt = prompt_generator.reply("", verbose=False)
    additional_prompt =  """ \n\n Bear in mind that while engaging with the user, questions may arise that seem unrelated to the initial prompt. It's crucial to maintain flexibility and creativity, interpreting and addressing these topics through the unique prism provided by the initial prompt."""
    perspective_prompt += additional_prompt
    end = perf_counter()
    print("Time to generate prompt: ", end - start)
    return perspective_prompt


import json
import traceback

import threading

class PerspectivePromptGenerator:
    def __init__(self, subjects, perspectives, max_workers=10, calls_per_minute=20, base_filename="prompt_"):
        self.subjects = subjects
        self.perspectives = perspectives
        self.executor = RateLimitedThreadPoolExecutor(
            max_workers=max_workers, 
            calls_per_minute=calls_per_minute
        )
        self.prompts = []
        self.base_filename = base_filename
        self.lock = threading.Lock()  # create a lock

    def handle_future(self, future):
        try:
            result = future.result()
            self.prompts.append(result)
            complete_filename = self.base_filename + "results.json"
            with self.lock:  # acquire the lock before writing to the file
                self.save_prompts_to_json(complete_filename)
        except Exception as e:
            error_report = {"error": str(e), "traceback": traceback.format_exc()}
            self.prompts.append(error_report)
            complete_filename = self.base_filename + "errors.json"
            with self.lock:  # acquire the lock before writing to the file
                self.save_prompts_to_json(complete_filename)
    
    def generate_prompts(self):
        for subject in self.subjects:
            for perspective in self.perspectives:
                future = self.executor.submit(
                    generate_perspective_prompt, 
                    subject, 
                    perspective
                )
                future.add_done_callback(self.handle_future)
        self.executor.shutdown(wait=True)
        return self.prompts

    def save_prompts_to_json(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.prompts, f)

