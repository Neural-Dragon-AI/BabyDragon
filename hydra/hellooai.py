import modal

stub = modal.Stub(image=modal.Image.debian_slim().pip_install("openai"))


@stub.function(secret=modal.Secret.from_name("oaisecret"))
def complete_text(prompt):
    import openai
    openai.api_key = "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"
    
    completion = openai.Completion.create(engine="ada", prompt=prompt)
    return completion.choices[0].text

@stub.local_entrypoint
def main():
    print(complete_text.call("Orc statblocks:"))