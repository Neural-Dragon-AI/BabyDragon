# BabyDragon

## Introduction

BabyDragon is a comprehensive package designed to facilitate the handling, analysis, and processing of heterogeneous data in a joint probabilistic framework. By integrating the power of Polars DataFrames, machine learning techniques, neural network embedders, and language models, BabyDragon is capable of handling and modeling a wide array of data types. Each type is represented as dimensions of a joint PDF, enabling automatic inference for efficient marginalization and conditioning across these dimensions.

## Data Types and Corresponding Pydantic Classes

Each BabyDragon data type (`bd_type`) is paired with a specific Pydantic class, providing rigorous data validation and handling:

- **NaturalLanguage**: This `bd_type` is used for processing natural language text, such as prose and speech transcripts. Its corresponding Pydantic class leverages the `tiktoken` library from OpenAI for validation and the `transformers` library from Hugging Face for advanced natural language processing tasks like tokenization and embedding. This category further branches into:
    - `BaseText`: General textual content that could be any free text.
    - `Book`: Extended textual content which could include chapters or sections.
    - `PythonCode`: Designed for Python scripts and snippets, it leverages `libcst` for parsing code syntax trees.
    - `Chat Conversation`: Designed for OpenAI's chat markup language, it supports back-and-forth conversations.
    - `Conversation Transcript`: This type includes a time, datetime, or delta-time index and diarization information for managing transcripts of conversations.

- **DiscreteData**: This `bd_type` handles discrete data, like labels, categories, and sequences. It can be stored as integers or strings or lists of these, and further breaks down into:
    - `Univariate`: Discrete data that can be modeled using a single random variable.
    - `Joint Discrete`: Discrete data represented by a set of discrete random variables that could be correlated.
    - `1D Sequence`: Represents auto-regressive/sequential nature of a sequence of discrete variables.
    - `ND Sequence`: Such as reinforcement learning (RL) environments, handling multiple sequences of discrete variables.

- **RealValuedData**: This type handles numerical data and extends into:
    - `Univariate`: Single, real-valued measurements.
    - `Multivariate`: This subtype is used for groups of real-valued measurements from the same sensor, such as multiple blood tests or the SNPs readings from a genetic test.

- **EpisodicTimeSeries**: This data type represents multiple realizations of the same dynamical system and requires an associated time, datetime, or delta-time object. It breaks down into:
    - `Univariate`: Single variable time series data.
    - `Multivariate`: Multivariate time series data that share the same time object.

- **Images**: This `bd_type` is used for images and it differentiates into:
    - `Natural Image`: Typical images, which can be used for tasks such as feature extraction, segmentation, classification, etc.
    - `GigaPixel Medical Image`: These images are too big to load in memory and require modality-specific pre-processing, e.g., using the histolab package for histopathological images. They are typically represented as arrays or sets of embeddings instead of a single embedding.
    - `Text Rich Images`: Images that contain significant textual content. This subtype can have pixel-level or bounding box annotations.

- **Sound**: Treated similarly to time-series, this `bd_type` can further branch into:
    - `Speech`: This subtype is used for spoken language, often associated with conversation transcripts.
    - `Music`: This subtype is often associated with lyrics and utilizes specialized```markdown
# MemoryFrame: Unifying Joint PDFs and Data Types

The MemoryFrame, serving as the heart of BabyDragon, bridges the gap between complex data types (`bd_types`) and the joint probabilistic modeling framework. As a dynamic, multidimensional probabilistic data container, it unifies the rigidity of structured data with the flexibility of unstructured data by providing a structured mapping between `bd_types` and Polars DataFrame columns.

## Architecture of MemoryFrame

A MemoryFrame is composed of `meta_columns`, `value_columns`, and `embedding_columns`, each of which serves a specific role in managing the different aspects of data and their representation.

### Meta Columns

`Meta_columns` encapsulate the metadata associated with each data instance, such as data source, timestamp, and unique identifiers. This information aids in data management and organization within the MemoryFrame.

### Value Columns

`Value_columns` are further sub-divided into `context_columns` and `embeddable_columns`.

`Context_columns` accommodate any data type supported by Polars and can be utilized to store transformed or augmented data like pre-trained classifier predictions. These columns provide additional context about the data.

`Embeddable_columns` specifically house data associated with defined `bd_type` classifications. This categorization enables automatic inference of the appropriate embedding techniques, allowing seamless integration with diverse language models. Such a structured approach facilitates the management of complex data types, such as natural language text or images.

### Embedding Columns

`Embedding_columns` preserve the high-dimensional vector representations of data from the `embeddable_columns`. The coherent mapping between `embeddable_columns` and `embedding_columns` ensures that each data instance is connected to its vector representation within the joint PDF framework.

## Automating Embedder Selection

BabyDragon's standout feature is its ability to automatically select the suitable vector embedder based on the `bd_type` of the data. This capability allows for the transformation of diverse data into a format compatible with machine learning algorithms and vector search operations, effectively reducing the preprocessing workload.

## Bridging Data Types with Joint PDF Modelling

The innovative architecture of the MemoryFrame augments traditional data operations with the power of vector-space operations, such as top-k nearest neighbor searches over vector representations. It allows users to handle structured and unstructured data within a single framework, thereby providing the capability to perform a diverse set of analytical tasks across different data types.

The MemoryFrame's role in the joint PDF framework is to map each `bd_type` to a dimension in a shared high-dimensional space. For instance, an `EpisodicTimeSeries` bd_type could correspond to one or more continuous dimensions in this space, while a `DiscreteData` bd_type could map to a discrete dimension. This mapping ensures that complex data analysis tasks, such as marginalization and conditioning, can be performed efficiently across these dimensions.

Through a balance of structured, SQL-like operations and vector space operations, BabyDragon provides a comprehensive and unified platform for sophisticated data processing and analysis. Whether you are dealing with structured data like `RealValuedData` or unstructured data like `NaturalLanguage`, BabyDragon's MemoryFrame offers a versatile solution for modeling and managing your data in the joint probabilistic space.



# Column Generators: Powering Dynamic Data Transformation and Infilling in Joint PDFs

In BabyDragon, column generators serve as dynamic tools for creating new columns in a MemoryFrame based on existing ones. These generator classes can perform a wide range of data transformations, offering a flexible solution to manage various data types and models. 

One such application of column generators is the utilization of Large Language Models (LLMs) for text transformations. An LLM can ingest a text column from a MemoryFrame, generate a summarized version of that text, and store this summary in a new column. This offers a seamless way to handle complex text-to-text transformations within a data frame.

The use of column generators isn't restricted to text operations. They can also be employed for image-to-text, text-to-image, and even text-to-speech tasks. Such capabilities enable the extraction of diverse types of information from the original data, enriching the MemoryFrame with varied, structured, and easily accessible data.

For time-series data, auto-regressive forecasters can serve as column generators, creating a predictive time series column based on an existing time series data column. These predictions can then be stored in a new column for further analysis.

## Infilling Missing Values

Column generators can also be utilized for infilling missing values in a column. Leveraging a generative model of the column, missing data can be generated conditionally based on all or a subset of other columns. This technique enables sophisticated handling of incomplete data, offering a robust solution for maintaining data integrity and quality.

## Efficiency and Error Handling

Column generation in BabyDragon is built for efficiency and reliability. It uses multi-threading and batching techniques for efficient data processing, coupled with robust error handling protocols to ensure smooth operations. To optimize computational resources, the package saves and reuses steps, enhancing its performance. This makes BabyDragon a highly reliable platform for dynamic data management, data transformation, and infilling missing values in joint PDFs.

# Non-parametric Mutual Information and Entropy

The BabyDragon package stands out for its unique non-parametric approach to computing mutual information and entropy across various types of columns in a dataset. This methodology provides a flexible and robust measure to assess the complexity and interdependence within your data.

Entropy, which quantifies the uncertainty or randomness inherent in a variable, and mutual information, which gauges the amount of information about one random variable obtained by observing another, are both pivotal in comprehending the relationships between different variables in a dataset. They are particularly essential in identifying conditionally independent variables.

The non-parametric approach to these computations ensures that BabyDragon is capable of handling a wide variety of data types and complexities. For instance:

- For **discrete variables**, the package can compute entropy and mutual information directly from frequency tables.

- For **continuous variables**, BabyDragon employs the concept of kernel density estimation, a non-parametric way of estimating the probability density function of a random variable. Using this, it can estimate entropy and mutual information.

- For **high-dimensional vectors** (like embeddings), BabyDragon uses clustering algorithms (such as K-means) to partition the high-dimensional space into discrete clusters. Then, it treats these clusters as discrete categories and calculates entropy and mutual information.

- For **mixed types**, BabyDragon's strategy is to train supervised predictors. For example, it can use a classifier to predict a discrete column based on other columns and then estimate the mutual information between the predicted and actual values of the discrete column.

This non-parametric approach allows BabyDragon to handle various data types and to provide a more robust and flexible understanding of your data's complexity and interrelationships.
## Stratified Sampling for Multimodal Predictions and Semi-parametric Models

BabyDragon's stratification capability embodies a groundbreaking concept. It allows the training of supervised predictors across all data types by merging multimodal datasets with embeddings and traditional categorical/numerical features for comprehensive predictions.

This process enables stratified sampling over this blended feature set, incorporating both standard structured data and unstructured data manifested through embeddings. This methodology gives a more complex and complete portrayal of your data, encapsulating a wide array of data types and their interconnections.

Stratified sampling, as implemented by BabyDragon, ensures that each data subset mirrors the entire dataset accurately, thereby enhancing the validation process's reliability. Furthermore, the stratified sampling technique is highly adaptable and can manage various data types, making it exceptionally suitable for multimodal datasets.

This innovative strategy coincides with the complexities that arise in statistical inference for semi-parametric models when the sampling method deviates from complete randomness. In a semi-parametric model, the conditional distribution of the outcome given the predictors, f(y|x, Θ), is parameterized, while the marginal distribution of the predictors, h(x), is not.

When observations are randomly sampled, inferring the unknown parameter Θ becomes simpler as we can estimate it by maximizing the conditional likelihood function. This is achievable without making assumptions about the marginal distribution of the predictors, an advantage as h(x) is unknown.

However, complications arise if the sampling process is not random or if it is 'endogenous'—where the probability of an observation being included in the sample depends on both the predictors and the outcome. In this situation, h(x) can't be factored out easily, making assumptions about h(x) necessary to estimate Θ.

BabyDragon provides techniques for dealing with these complexities:

1. **Quantization of Real-Valued Columns**: BabyDragon can discretize real-valued columns into "bins", converting them into categorical variables for the purpose of creating strata. This process, called quantization, transforms a continuous spectrum of values into a finite number of intervals. The resulting binned data can then be used like any other categorical variable in the stratification process.

2. **Quantization of High Dimensional Vector Fields**: When working with high-dimensional embedded columns, a slightly different approach is required. BabyDragon employs a clustering algorithm to divide the high-dimensional space into discrete clusters, which are then used as strata. Simultaneously, a non-parametric estimate of these vector fields' entropy is calculated to gauge their inherent diversity and complexity.

The combination of these methods introduces a unique strategy for dealing with high-dimensional data, ensuring that the cross-validation process remains robust and reliable, regardless of the data's nature in the MemoryFrame.

## Artificial Transfer Learning Experiments with Stratified Sampling and Semi-parametric Models

Stratified sampling in BabyDragon not only ensures each data subset accurately represents the entire dataset, but it also enables the exploration of artificial transfer learning experiments. This process lets us examine invariances and equivalence classes across different strata, leading to a deeper understanding and novel scientific discoveries.

In the conventional methodology of stratified cross-validation, the objective is to make sure each fold is a good representation of the entire dataset. By purposefully manipulating the distribution of strata in the training and validation sets, we can devise experiments to evaluate the models' ability to generalize under specific conditions.

This experimental framework is highly flexible and can be adapted to a variety of contexts. It can help uncover hidden dataset biases, understand model vulnerabilities, or reveal surprising data patterns. Additionally, it can support exploratory data analysis and hypothesis testing, all under the unified framework offered by BabyDragon.


## 1. Setting Up the Experiment

Assume we have identified a certain stratum (or a set of strata) that is particularly interesting. For instance, these could be strata representing specific age groups in a demographics study, various transaction types in financial analysis, or distinct categories of images in a computer vision task.

We can arrange our training and validation folds such that this stratum is over-represented in the training set and under-represented in the validation set, or vice versa.

## 2. Transfer Learning and Invariances

Under this arrangement, a model trained on the altered training set effectively learns from a biased perspective of the world. When this model is evaluated against the validation set, any significant decline in performance can be interpreted as the model's inability to generalize beyond the training data's bias.

By conducting a sequence of such experiments, with systematic variation of the over- and under-represented strata, we can glean insights into the invariances the model successfully learns, and those it struggles with.

This methodology forms the foundation of a scientific inquiry. Our goal extends beyond just creating a model that excels overall; we aim for a model that performs well under specific conditions. This offers an empirical basis for identifying equivalence classes of strata, where an equivalence class comprises strata that a model treats as essentially identical.

## 3. Beyond Cross-Validation

Although the previous discussion centered on the context of cross-validation within a single dataset, the same principles can be extended to broader contexts. For example, these principles can be employed to construct "artificial" transfer learning scenarios, where a model is trained on one dataset and evaluated on another, to test its generalization ability across different yet related domains.

In conclusion, while the conventional use of stratified sampling is to ensure robustness in cross-validation, its strategic application can yield deeper insights about the models and the phenomena they aim to simulate. This experimental approach allows us to estimate the invariances or equivalences a model can learn, and those it struggles with. Such insights can guide the creation of better models, influence the design of future data collection endeavors, or even inform the generation of new hypotheses in the underlying domain of study.

This experimental framework is extremely versatile and can be adapted to a plethora of contexts. It can assist in uncovering hidden dataset biases, understanding model vulnerabilities, or revealing surprising data patterns. Furthermore, it can support exploratory data analysis and hypothesis testing, all under the unified framework provided by BabyDragon.
