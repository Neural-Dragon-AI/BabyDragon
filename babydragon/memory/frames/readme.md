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
| Episodic Time Series          | Array / List of Floats   | Dynamic Time Warping (DTW), Inverse Fourier Features of DTW Kernel, Auto-regressive Forecasting (traditional, XGBoost, Neural Differential Equations)|
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

# Stratified Sampling in MemoryFrame

Stratified sampling is a powerful technique used to ensure that the subset (or subsets) of data you're working with is representative of the whole in certain specified respects. When applied to the task of cross-validation in machine learning, it helps ensure that each fold is representative of the whole dataset, improving the reliability of the validation process.

In the context of MemoryFrame, a data structure which holds in-memory datasets augmented with additional contextual and embedded features, we are interested in performing stratified cross-validation using all available data. This includes categorical context columns, embedded high-dimensional data, as well as quantized real-valued columns.

## 1. Quantization of Real-Valued Columns

Real-valued columns can be discretized into "bins", essentially turning them into categorical variables for the purpose of creating strata. This process, known as quantization, converts a continuous range of values into a finite number of intervals. The resulting binned data can then be used like any other categorical variable in the stratification process.

## 2. Entropy and Mutual Information Computation

With all categorical data at hand, which includes the original categorical columns and the newly quantized real-valued columns, we compute the entropy and mutual information between these columns. 

Entropy gives us a measure of the uncertainty or randomness of a single variable, while mutual information measures the amount of information you can obtain about one random variable by observing another. By calculating these measures, we gain insights into the relationships between variables, and can identify those which are conditionally independent.

This information is crucial when creating the strata, as we strive to create strata that are as informative as possible. The concept here is to ensure that each stratum, or subset of data, is as similar as possible internally, while being as different as possible from the other strata.

## 3. High Dimensional Vector Fields

For high-dimensional embedded columns, we must take a slightly different approach. These embeddings might represent complex data like text or images in a condensed form, and can't be treated directly as categorical data. 

Instead, we propose to use a clustering algorithm to partition the high-dimensional space into discrete clusters, and use these clusters as strata. At the same time, we compute a non-parametric estimate of the entropy of these vector fields to get a measure of their inherent diversity and complexity.

This approach allows us to treat high-dimensional data in a way that's compatible with our stratified sampling methodology, ensuring that our cross-validation process remains robust and reliable, no matter the nature of the data in the MemoryFrame.

# Artificial Transfer Learning Experiments with Stratified Sampling

Stratified sampling not only ensures that each subset of data is representative of the whole dataset, but it can also be leveraged to create interesting and insightful artificial transfer learning experiments. This process allows for the exploration of invariances and equivalence classes across different strata, which can lead to deeper understanding and new scientific inquiries.

In the classic approach of stratified cross-validation, we strive to ensure that each fold is representative of the whole dataset. But by strategically varying the distribution of strata in training and validation sets, we can set up experiments to test how well models generalize under specific conditions.

## 1. Setting Up the Experiment

Let's say we have identified a certain stratum (or a set of strata) that is of particular interest. For instance, these could be strata that represent a certain demographic in a social science study, certain types of transactions in a financial analysis, or specific categories of images in a computer vision task.

We can construct our training and validation folds such that this stratum is over-represented in the training set and under-represented in the validation set, or vice versa. 

## 2. Transfer Learning and Invariances

In this setup, a model trained on the modified training set is in effect a model trained with a biased view of the world. When this model is evaluated on the validation set, any significant drop in performance can be interpreted as the model's inability to generalize beyond the bias in the training data.

By conducting a series of such experiments, systematically varying the strata that are over- and under-represented, we can gain insights into what invariances the model is able to learn, and which ones it struggles with. 

This forms the basis for a form of scientific inquiry, where we're not just interested in creating a model that performs well overall, but one that performs well under specific conditions. It provides an empirical basis for identifying equivalence classes of strata, where an equivalence class consists of strata that a model treats as effectively the same.

## 3. Beyond Cross-Validation

While the above discussion focused on the context of cross-validation in a single dataset, the same principles can be applied in a broader context. For instance, they can be used to create "artificial" transfer learning scenarios, where a model is trained on one dataset and evaluated on another, to test how well it generalizes across different but related domains.

Overall, while the standard use of stratified sampling is to ensure robustness in cross-validation, its strategic use can lead to deeper insights about the models and the phenomena they are trying to capture.
