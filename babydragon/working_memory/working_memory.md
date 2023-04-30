# Working Memory Concept for Language Models

The working memory class in large language models is a system that combines
various components to generate contextually appropriate and coherent responses
based on user inputs. This concept is inspired by Minsky's Society of Mind
theory, which breaks down complex tasks into smaller, specialized agents and
sub-agents working together. The working memory class includes the following
agents:

1. Short-term Memory
2. Episodic Memory
3. Associative Memory
4. Goal Definition
5. Self-prompting
6. Output Format Definition

Each of these agents is further divided into sub-agents responsible for specific
tasks within their respective domain. The sub-agents are:

1. Short-term Memory: a. Input capture b. Memory decay c. Context retrieval

2. Episodic Memory: a. Event storage b. Memory retrieval c. Event pruning

3. Associative Memory: a. Knowledge indexing b. Query formulation c. Knowledge
   retrieval

4. Goal Definition: a. Goal extraction b. Goal prioritization c. Goal-driven
   response generation

5. Self-prompting: a. Ambiguity detection b. Internal question generation c.
   Response refinement

6. Output Format Definition: a. Format identification b. Response structuring c.
   Format validation

Now, let's examine the multi-threaded flow for the inner loop process.

## Multi-threaded Inner Loop Process

The multi-threaded inner loop process aims to handle complex tasks more
efficiently by leveraging concurrency. This process enables faster and more
responsive language model interactions. The following steps outline the
multi-threaded inner loop process:

```
function processInput(userInput):
    preprocessedInput = preprocess(userInput)

    # Concurrent processing - Step 3
    parallel {
        goals = extractGoals(preprocessedInput)
        outputFormat = identifyOutputFormat(preprocessedInput)
        memoryContext = retrieveMemoryContext(preprocessedInput)
    }

    query = formulateQuery(preprocessedInput)

    # Concurrent processing - Step 6
    parallel {
        ambiguities = detectAmbiguities(preprocessedInput)
        retrievedKnowledge = searchExternalKnowledge(query)
    }

    if ambiguities:
        internalQuestions = generateInternalQuestions(ambiguities)

    refinedInput = refineInput(preprocessedInput, internalQuestions, retrievedKnowledge)

    # Concurrent processing - Step 10
    parallel {
        goalDrivenResponse = generateGoalDrivenResponse(refinedInput, goals)
        structuredResponse = structureResponse(goalDrivenResponse, outputFormat)
    }

    workingMemory = constructWorkingMemory(memoryContext, refinedInput, goals, outputFormat)

    finalPrompt = generateFinalPrompt(workingMemory)

    response = generateResponse(finalPrompt)

    return response
```

1. User input: Receive a text-based query or statement from the user.

2. Preprocessing: Preprocess and tokenize the input request for further
   processing by sub-agents.

3. Concurrent processing starts: a. Goal extraction: Identify user's goals and
   objectives. b. Format identification: Determine the desired output format for
   the response. c. Memory retrieval: Retrieve relevant context and past events
   from Short-term Memory and Episodic Memory.

4. Synchronization: Wait for all concurrent processes in step 3 to complete.

5. Query formulation: Transform the user's input into a suitable query format
   for the Associative Memory.

6. Concurrent processing starts: a. Ambiguity detection: Check for unclear or
   ambiguous elements in the user's input. b. Knowledge retrieval: Search for
   relevant information in external knowledge sources based on the query.

7. Synchronization: Wait for all concurrent processes in step 6 to complete.

8. Internal question generation: Generate internal questions or prompts if
   ambiguities are detected.

9. Response refinement: Update and refine the model's understanding of the
   user's input using internal questions and retrieved knowledge.

10. Concurrent processing starts: a. Goal-driven response generation: Generate a
    response addressing the user's objectives based on the identified goals. b.
    Response structuring: Organize the generated response according to the
    identified output format.

11. Synchronization: Wait for all concurrent processes in step 10 to complete.

12. Working memory construction: Integrate the retrieved short-term memory,
    episodic memory, associative memory, goal definition, self-prompting, and
    output format definition into the working memory, while staying within the
    maximum token limit.

13. Final prompt generation: Create the final prompt using the working memory,
    which serves as the input for the language model's response generation
    process.

14. Output: Generate a response based on the final prompt, ensuring it is
    coherent, contextually appropriate, and adheres to the desired output
    format.

| Method                                                                               | Input                                                                       | Output                                       | Function                                                                                                                                    |
| ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| preprocess(userInput)                                                                | Raw user input (text)                                                       | Preprocessed and tokenized input             | Preprocesses and tokenizes the user input for further processing                                                                            |
| extractGoals(preprocessedInput, memoryContext)                                       | Preprocessed input, memory context                                          | Identified goals and objectives              | Extracts the user's goals and objectives from the input and memory context                                                                  |
| identifyOutputFormat(preprocessedInput, memoryContext)                               | Preprocessed input, memory context                                          | Desired output format                        | Determines the desired output format for the response using input and memory context                                                        |
| retrieveMemoryContext(preprocessedInput, memoryType)                                 | Preprocessed input, memory type (STM, LTM, etc.)                            | Relevant context and past events             | Retrieves context and past events from specified memory type                                                                                |
| formulateQuery(preprocessedInput, memoryContext)                                     | Preprocessed input, memory context                                          | Query format suitable for Associative Memory | Transforms the input into a query format for the Associative Memory using memory context                                                    |
| detectAmbiguities(preprocessedInput, memoryContext)                                  | Preprocessed input, memory context                                          | Detected ambiguities                         | Checks for unclear or ambiguous elements in the input and memory context                                                                    |
| searchExternalKnowledge(query, memoryType)                                           | Query suitable for Associative Memory, memory type (e.g., vector storage)   | Retrieved knowledge                          | Searches specified external knowledge sources for relevant information                                                                      |
| generateInternalQuestions(ambiguities, memoryContext)                                | Detected ambiguities, memory context                                        | Internal questions or prompts                | Generates internal questions or prompts if ambiguities are detected, using memory context                                                   |
| refineInput(preprocessedInput, internalQuestions, retrievedKnowledge, memoryContext) | Preprocessed input, internal questions, retrieved knowledge, memory context | Refined input                                | Updates and refines the model's understanding of the input using internal questions, retrieved knowledge, and memory context                |
| generateGoalDrivenResponse(refinedInput, goals, memoryContext)                       | Refined input, identified goals, memory context                             | Goal-driven response                         | Generates a response addressing the user's objectives based on the identified goals and memory context                                      |
| structureResponse(goalDrivenResponse, outputFormat, memoryContext)                   | Goal-driven response, output format, memory context                         | Structured response                          | Organizes the generated response according to the identified output format, using memory context                                            |
| constructWorkingMemory(memoryContext, refinedInput, goals, outputFormat)             | Memory context, refined input, goals, output format                         | Working memory                               | Integrates context, refined input, goals, and output format into the working memory                                                         |
| generateFinalPrompt(workingMemory)                                                   | Working memory                                                              | Final prompt                                 | Creates the final prompt using the working memory as input for the language model's response generation process                             |
| generateResponse(finalPrompt)                                                        | Final prompt                                                                | Generated response                           | Generates a response based on the final prompt, ensuring it is coherent, contextually appropriate, and adheres to the desired output format |
