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