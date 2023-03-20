import openai


class OpenAiEmbedder:
    def get_embedding_size(self):
        return 1536
    def embed(self, data, embed_mark = True, verbose = False):
        try:
            if embed_mark is False and type(data) is dict and "content" in data:
                print("Embedding without mark", data["content"])
                out = openai.Embedding.create(input=data["content"], engine='text-embedding-ada-002')
            else:
                if verbose is True:
                    print("Embedding without preprocessing the input", data)
                out = openai.Embedding.create(input=str(data), engine='text-embedding-ada-002')
        except:
            raise ValueError("The data  is not valid")
        return out.data[0].embedding
    def embed_list(self,data):
        #use the batched version of the API by giving a list as input
        #che that is listo of strings
        if type(data) is not list:
            raise ValueError("The data  is not valid")
        out = openai.Embedding.create(input=data, engine='text-embedding-ada-002')


def mark_system(system_prompt):
    return {"role": "system", "content": system_prompt}
def mark_answer(answer):
    return {"role": "assistant", "content": answer}
def mark_question(question):
    return {"role": "user", "content": question}
def check_dict(message_dict):
        if type(message_dict) is list and len(message_dict) == 1 and type(message_dict[0]) is dict:
            message_dict = message_dict[0]
        elif type(message_dict) is not dict:
            raise Exception("The message_dict should be a dictionary or a [dictionary] instead it is ", message_dict, type(message_dict))  
        return message_dict