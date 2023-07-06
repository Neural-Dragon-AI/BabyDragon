# BabyDragon: Comprehensive Data Management and Processing

## Introduction

BabyDragon is a versatile package designed for handling heterogeneous data in an efficient and intuitive manner. It brings together the capabilities of Polars Frames, machine learning techniques, neural network embedders, and language models to streamline the handling of diverse data types.

## Data Types and Corresponding Pydantic Classes

Each BabyDragon data type (`bd_type`) is coupled with a specific Pydantic class, providing comprehensive data validation:

- **NaturalLanguage**: Utilized for natural language text, including prose, speech transcripts. Its associated Pydantic class uses the `tiktoken` library for validation.
- **PythonCode**: For managing Python code, it leverages `libcst` for parsing code syntax trees.
- **Conversation**: Designed for OpenAI's chat markup language. The Pydantic class verifies a list of message objects.

These `bd_types` are flexible, supporting either single or multiple strings. 

## Additional Modalities and Their Pydantic Classes

Apart from text data, BabyDragon supports numerous other modalities:

- **Finite Alphabet Discrete Sequences**: Stored as lists of strings or integers. Useful for replay buffers or tabular reinforcement learning environments.
- **Episodic Time Series**: Stored as arrays or lists of floats, these represent multiple realizations of the same dynamical system.
- **Images**: Natural images for feature extraction, segmentation, classification, medical images, and text-rich images.
- **Audio**: BabyDragon differentiates between Speech and Music data, offering features like speech recognition and sentiment analysis for speech and genre classification for music.

## MemoryFrame: The Backbone of BabyDragon

BabyDragon's core construct, the `MemoryFrame`, serves as a bridge between Pydantic and Polars, facilitating data validation and storage. `MemoryFrame` is essentially a Polars DataFrame wrapped with additional functionalities. It classifies data into three groups of columns: `meta_columns`, `value_columns`, and `embedding_columns`.

### Meta Columns

Meta columns carry metadata pertaining to each data entry. This can include information such as:

- ID
- Name
- Source
- Author
- Created At
- Last Modified At

### Value Columns

Value columns are further subdivided into two categories: `context_columns` and `embeddable_columns`.

#### Context Columns

Context columns house data of any polar type. They can contain auto-generated data based on specific data type applications like pre-trained classifiers' predictions or data transformations/augmentations.

#### Embeddable Columns

Embeddable columns, on the other hand, hold data of stricter `babydragon` types. This rigidity allows for automatic inference of embedders and seamless interaction with language models.

### Embedding Columns

These columns store vector representations corresponding to the values in the `embeddable_columns`. The number of embedding columns mirrors the number of embeddable columns.

## Intelligent Selection of Vector Embedders

BabyDragon boasts a feature of automatic selection of vector embedders based on the `bd_type`. This feature identifies the best-suited embedder for the data, facilitating the conversion of complex, diverse data into formats suitable for machine learning algorithms and operations like vector search. It streamlines preprocessing and optimizes the data processing pipeline.

## Unifying Structured and Unstructured Data

BabyDragon augments conventional database operations like `select`, `group_by`, `filter`, and `sort` with top-k nearest neighbor search over vector representations of column values. This enhancement allows manipulation of both structured and unstructured data within the same framework, making it possible to perform advanced analytical tasks across data types. Users can create dynamic query pipelines: employing SQL-like operations to filter and organize data before leveraging vector search operations to draw insights based on vector space similarity. This proves particularly advantageous when handling unstructured data like text or images, paving the way for efficient processing and sophisticated exploration of heterogeneous data.


# Column Generators
The concept of column generators is a key feature in the BabyDragon package. Column generators are arbitrary generator classes that can take existing columns in a MemoryFrame and use them to generate new columns. A wide range of generator classes are supported, allowing you to utilize different types of data and models to generate new columns in your MemoryFrame.

For example, you could use a Large Language Model (LLM) as a column generator to take a text column and generate a summary column. This involves feeding the text from the original column into the LLM, having it generate a summary, and then storing that summary in a new column in the MemoryFrame.

But column generation is not limited to just text-to-text operations. You can also use column generators for image-to-text, text-to-image, or text-to-speech tasks. This enables you to extract and store diverse types of information from your original data in a structured and easily accessible format.

Moreover, you can use auto-regressive forecasters as column generators to create a time series column from another one. This allows you to make predictions about future data points based on the existing data in a column, and then store those predictions in a new column for further analysis.

### Efficiency and Error Handling

All this processing happens under the hood, with multi-threading and batching for efficiency, and robust error handling to ensure the smooth processing of data. All the steps are saved to optimize the process, ensuring that the package makes the best use of computational resources.

# Stratified Sampling for Multimodal Classification
The stratification part of BabyDragon leverages a novel concept: learning supervised classifiers for all data types by combining multimodal datasets via embeddings and traditional categorical/numerical features for multimodal classification.

The package supports stratified sampling over this combined feature set, which includes traditional structured data and unstructured data represented through embeddings. This provides a more nuanced and comprehensive view of your data, as it encapsulates different types of data and the relationships between them.

The package uses stratified sampling to ensure each subset of data is representative of the entire dataset, improving the reliability of the validation process. Moreover, the stratified sampling technique is flexible and can adapt to different kinds of data, making it particularly suitable for handling multimodal datasets.

By leveraging column generators and stratified sampling in this way, BabyDragon provides a powerful tool for handling, analyzing, and making predictions based on heterogeneous datasets. The flexibility and capabilities of the package make it a potent tool for data analysis and machine learning tasks.

## 1. Quantization of Real-Valued Columns

Real-valued columns can be discretized into "bins", essentially turning them into categorical variables for the purpose of creating strata. This process, known as quantization, converts a continuous range of values into a finite number of intervals. The resulting binned data can then be used like any other categorical variable in the stratification process.

## 2. Quantization of High Dimensional Vector Fields

For high-dimensional embedded columns, we must take a slightly different approach. These embeddings might represent complex data like text or images in a condensed form, and can't be treated directly as categorical data. 

Instead, we propose to use a clustering algorithm to partition the high-dimensional space into discrete clusters, and use these clusters as strata. At the same time, we compute a non-parametric estimate of the entropy of these vector fields to get a measure of their inherent diversity and complexity.

This approach allows us to treat high-dimensional data in a way that's compatible with our stratified sampling methodology, ensuring that our cross-validation process remains robust and reliable, no matter the nature of the data in the MemoryFrame.

## 3. Entropy and Mutual Information Computation

With all categorical data at hand, which includes the original categorical columns and the newly quantized real-valued columns, we compute the entropy and mutual information between these columns. 

Entropy gives us a measure of the uncertainty or randomness of a single variable, while mutual information measures the amount of information you can obtain about one random variable by observing another. By calculating these measures, we gain insights into the relationships between variables, and can identify those which are conditionally independent.

This information is crucial when creating the strata, as we strive to create strata that are as informative as possible. The concept here is to ensure that each stratum, or subset of data, is as similar as possible internally, while being as different as possible from the other strata.


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

# Chatbot Memory Support and Grounding with Multimodal Data
BabyDragon offers strong capabilities when it comes to supporting chatbot functionalities, particularly in the context of grounding responses and retrieval from multimodal datasets.

## Retrieval and Grounding for Chatbot Agents
The MemoryFrame in BabyDragon provides a robust structure for organizing and accessing information, making it an excellent support system for chatbot agents. The data stored in the MemoryFrame can be used to inform a chatbot's responses, leading to more context-aware and grounded interactions.

For instance, suppose a chatbot is asked a question about a specific data point or trend within the stored data. The MemoryFrame can be queried to retrieve the relevant information, which can then be used to formulate a grounded and accurate response.

## Chatbots and Multimodal Data
BabyDragon extends the concept of chatbot memory support to multimodal datasets. Multimodal data includes data of different types - such as text, images, and audio - that can be processed and represented together in the MemoryFrame.

A chatbot supported by BabyDragon can, therefore, leverage this rich, multimodal context to provide more comprehensive and accurate responses. For example, a chatbot can use text-based data in conjunction with relevant image or audio data to answer a query, resulting in a more informative response.