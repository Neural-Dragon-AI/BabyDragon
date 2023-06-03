.. BabyDragon documentation master file, created by
   sphinx-quickstart on Tue May 30 17:45:43 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BabyDragon's documentation!
======================================

.. toctree::
   :maxdepth: 2

   authors

.. toctree::
   :maxdepth: 2
   :caption: Tasks:

   tasks/llm_task
   tasks/multi_kernel_task
   tasks/base_task
   tasks/topic_tree_task
   tasks/embedding_task

.. toctree::
   :maxdepth: 2
   :caption: Memory/Kernels:

   memory/kernels/memory_kernel
   memory/kernels/kernel_clustering
   memory/kernels/multi_kernel
   memory/kernels/multi_kernel_visualization
   memory/kernels/kernel_utils

.. toctree::
   :maxdepth: 2
   :caption: Memory/Threads:

   memory/threads/base_thread
   memory/threads/fifo_thread
   memory/threads/vector_thread

.. toctree::
   :maxdepth: 2
   :caption: Memory/Indexes:

   memory/indexes/python_index
   memory/indexes/memory_index
   memory/indexes/pandas_index

.. toctree::
   :maxdepth: 2
   :caption: Chat:

   chat/base_chat
   chat/memory_chat
   chat/chat


.. toctree::
   :maxdepth: 2
   :caption: Utils:

   utils/multithreading
   utils/pandas
   utils/hf_datasets
   utils/chatml


.. toctree::
   :maxdepth: 2
   :caption: Models/embedders:

   models/embedders/ada2
   models/embedders/cohere
   models/embedders/sbert

.. toctree::
   :maxdepth: 2
   :caption: Models/generators:

   models/generators/cohere
   models/generators/chatgpt

.. toctree::
   :maxdepth: 2
   :caption: Processors:

   processors/github_processors
   processors/os_processor
   processors/parsers/git_metadata
   processors/parsers/visitors
   processors/parsers/python_parser


.. toctree::
   :maxdepth: 2
   :caption: Apps:

   apps/auto_perspective/perspective
