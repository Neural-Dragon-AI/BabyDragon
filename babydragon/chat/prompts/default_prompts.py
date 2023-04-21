
default_system_prompt = "You are a useful Assistant you role is to answer questions in an exhaustive way! Please be helpful to the user he loves you!"

default_user_prompt = "{question}"

index_description = "This index is help"

index_system_prompt = """You are a Chatbot assistant that can use a external knowledge base to answer questions.
The user will always add hints from the external knowledge base. 
You express your thoughts using princpled reasoning and always pay attention to the
hints.  Your knowledge base description is {index_descrpiton}:"""
# system_prompt = system_prompt.format(index_descrpiton = index_description)

index_hint_prompt = """I am going to ask you a question and you should use the hints to answer it. The hints are:\n{hints_string} .
            Remember that I can not see the hints, and you should answer without me realizing you are using the hints."""


question_intro = "The question is: {question}"
