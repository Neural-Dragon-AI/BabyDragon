from typing import List

def convert_mark_to_str_prompt(messages: List[dict], prompt: str = "") -> str:
    prompt = ""
    for message in messages:
        role = message["role"].upper()
        content = message["content"]
        prompt += f" #{role}: {content}"

    return prompt

def mark_system(system_prompt):
    return {"role": "system", "content": system_prompt}


def mark_answer(answer):
    return {"role": "assistant", "content": answer}


def mark_question(question):
    return {"role": "user", "content": question}


def check_dict(message_dict):
    if (
        type(message_dict) is list
        and len(message_dict) == 1
        and type(message_dict[0]) is dict
    ):
        message_dict = message_dict[0]
    elif type(message_dict) is not dict:
        raise Exception(
            "The message_dict should be a dictionary or a [dictionary] instead it is ",
            message_dict,
            type(message_dict),
        )
    return message_dict


def get_mark_from_response(response, model = "gpt"):
    # return the answer from the response
    if "gpt" in model:
        role = response["choices"][0]["message"]["role"]
        message = response["choices"][0]["message"]["content"]
    elif "command" in model:
        role = "assistant"
        message = response[0].text 
    else:
        raise Exception("Unknown model type")
    return {"role": role, "content": message}


def get_str_from_response(response, model = "gpt"):
    # return the answer from the response
    if "gpt" in model:
        return response["choices"][0]["message"]["content"]
    elif "command" in model:
        text = response[0].text
        text_without_assistant = text.replace("#ASSISTANT", "")
        return text_without_assistant
    else:
        raise Exception("Unknown model type")