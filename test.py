import code
import os

import openai

from babydragon.working_memory.short_term_memory.git_memory import GitMemory


def open_repl(local_vars=None):
    """
    Opens an interactive Python shell (REPL) with the given local variables.

    Args:
        local_vars (dict): A dictionary containing the local variables to be available in the REPL.
    """
    if local_vars is None:
        local_vars = {}

    # Update the local variables with the globals, so that all functions and variables are accessible
    local_vars.update(globals())

    # Start the REPL
    code.interact(local=local_vars)


if __name__ == "__main__":
    openai.api_key = "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"

    username = "Danielpatrickhug"
    repo_name = "GitModel"
    base_directory = "work_folder"

    # Make sure the work folder exists
    if not os.path.exists(base_directory):
        os.mkdir(base_directory)

    git_memory = GitMemory(username, repo_name)
    git_memory.create_code_index(base_directory)

    print(git_memory)
    print(type(git_memory.code_index.index))
    """
    #ic = Chat (model='gpt-3.5-turbo-0301',index_dict={"babydragon": pind})
    vci = FifoVectorChat(model='gpt-3.5-turbo-0301',index_dict={"GITMODEL": git_memory.code_index.index}, max_output_tokens=500, max_index_memory= 1000, max_memory=2000, longterm_frac=0.3)
    vci.gradio()
    # Add this line at the end of your script
    #open_repl(locals())
    """
