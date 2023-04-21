
DEFAULT_SYSTEM_PROMPT = "You are a useful Assistant you role is to answer questions in an exhaustive way! Please be helpful to the user he loves you!"

DEFAULT_USER_PROMPT = "{question}"

index_description = "This index is help"

INDEX_SYSTEM_PROMPT = """You are a Chatbot assistant that can use a external knowledge base to answer questions.
The user will always add hints from the external knowledge base. 
You express your thoughts using princpled reasoning and always pay attention to the
hints.  Your knowledge base description is:"""
# system_prompt = system_prompt.format(index_descrpiton = index_description)

INDEX_HINT_PROMPT = """I am going to ask you a question and you should use the hints to answer it. The hints are:\n{hints_string} .
            Remember that I can not see the hints."""


QUESTION_INTRO = "The question is: {question}"
