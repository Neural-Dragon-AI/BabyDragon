# BabyDragon Package Overview

The `babydragon` package is designed to handle heterogeneous data in an efficient and intuitive way. It combines the power of polar frames, machine learning techniques, neural network embedders, and language models. This package is very flexible and could significantly facilitate the handling of different types of data. 

## MemoryFrame

A central component of this package is the `MemoryFrame` class. This frame classifies data into three groups of columns: `meta_columns`, `value_columns`, and `embedding_columns`.

### Meta Columns

The `meta_columns` are used to store metadata about each sample. These columns might contain:

- ID
- Name
- Source
- Author
- Created At
- Last Modified At

### Value Columns

The `value_columns` are subdivided into two types: `context_columns` and `embeddable_columns`.

#### Context Columns

The `context_columns` contain data of any polar type. It can also be auto-generated based on the data type, like applying pre-trained classifiers to get their prediction or applying some transformation or augmentation to the data.

#### Embeddable Columns

The `embeddable_columns` contain data of stricter `babydragon` types, which will allow automatic inference of embedders and interaction with language models. 

### Embedding Columns

The `embedding_columns` store the vector for the corresponding value in the `embeddable_columns`. There will be the same number of embedding columns as there are embeddable columns.

## BabyDragon Auto-Data Types

BabyDragon supports several auto-data types for automatic embedder inference:

- **Text**: This includes natural language text, audio-transcripts, and python-code. They can be represented as strings or lists of strings. Audio transcripts also have support for timestamps and diarization, while Python uses `libcst` to parse code syntax trees and automatically create a rich context of variables associated with a script.

- **Finite Alphabet Discrete Sequences**: These can be represented as a list of strings or a list of integers. This data type can be used to store replay buffers or datasets for tabular reinforcement learning environments. The package supports learning epsilon machines, hidden Markov models (HMM), and small-scale transformers for model-based prediction.

- **Episodic Time Series**: These are multiple realizations of the same dynamical system with float or int values. They can be represented as arrays or lists of floats. The package supports 1D search using dynamic time warping (DTW) and DTW-based kernels. In addition, time-series features using inverse Fourier features of the DTW kernel for 1D are supported. The package also facilitates learning n-dimensional auto-regressive forecasting with traditional methods, XGBoost, and neural differential equations/operators.

- **Images**: The package supports a variety of images, including natural images (for feature extraction, segmentation, classification), medical images, and text-rich images (such as slides and plots).

- **Audio**: The package supports audio data, and differentiates between Speech and Music This includes speech recognition, speaker diarization, and possibly emotion or sentiment analysis. As well as conenction to speec2text api. For music data, the package might offer features like genre classification, beat detection, and music recommendation based on audio and lyrics feature analysis.

| BD Data Type                   | Representation           | Supported Operations/Features                                                                                                                |
|-------------------------------|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Text                          | String / List of Strings | Natural Language Processing, Audio Transcript (with support for timestamps and diarization), Python Code (with `libcst` parsing)              |
| Finite Alphabet Discrete Seq. | List of Strings/Integers | Replay Buffers, Tabular Reinforcement Learning Environments, Epsilon Machines, HMM, Small-scale Transformers                                  |
| Episodic Time Series          | Array / List of Floats   | Dynamic Time Warping (DTW), Inverse Fourier Features of DTW Kernel, Auto-regressive Forecasting (traditional, XGBoost, Neural Diff. Equations)|
| Images                        | -                        | Natural Images (feature extraction, segmentation, classification), Medical Images, Text-rich Images                                           |
| Audio                         | -                        | Speech (Speech Recognition, Speaker Diarization, Emotion/Sentiment Analysis), Music (Genre Classification, Beat Detection)                     |


## Bridging Structured and Unstructured Data


`BabyDragon` extends typical database operations such as `select`, `group_by`, `filter`, and `sort` by incorporating top-k nearest neighbor search capabilities over vector representations of column values. This integration enables the manipulation and querying of both structured and unstructured data within a unified framework. Users can create dynamic query pipelines, first employing SQL-like operations to filter and organize the data, followed by vector search operations to extract information based on similarity in the vector space. This is especially beneficial when dealing with unstructured data like text or images. Furthermore, `BabyDragon` supports advanced analytical tasks across data types. The fusion of these techniques enables efficient processing and advanced exploration of heterogeneous data.


## LLM-Ready Processing

`BabyDragon` is designed to efficiently work with Large Language Models (LLMs) like OpenAI, Cohere, and Google, as well as locally hosted GPU models. It allows parallelized calls to these LLMs to process data and create new rows. 

### Supporting Retrieval and Grounding for LLM Agents

The `MemoryFrame` can be used as a support for retrieval and grounding of LLM agents or chatbots. It provides an efficient way to use the knowledge stored in the `MemoryFrame` to inform the responses of these agents, leading to more context-aware and grounded interactions.

### LLMs for Data Processing and Feature Creation

The LLMs can also be used to elaborate the data and produce new columns. This includes tasks like summarizing a text column, interpreting a mix of text and numerical columns, or using the LLM as a few-shot classifier. In fact, any task that depends on row-wise inputs can be achieved through the integrated processing with LLMs. 

### Efficiency and Error Handling

All this processing happens under the hood, with multi-threading and batching for efficiency, and robust error handling to ensure the smooth processing of data. All the steps are saved to optimize the process, ensuring that the package makes the best use of computational resources.

