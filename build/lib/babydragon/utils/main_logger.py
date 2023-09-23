import logging

# Create a custom logger
logger = logging.getLogger(__name__)

# Set the level of logging.
# DEBUG, INFO, WARNING, ERROR, CRITICAL are the levels in increasing order of severity.
logger.setLevel(logging.INFO)

# Create handlers
c_handler = logging.StreamHandler()  # Console handler
f_handler = logging.FileHandler("file.log")  # File handler
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.ERROR)

# Create formatters and add them to handlers
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)
