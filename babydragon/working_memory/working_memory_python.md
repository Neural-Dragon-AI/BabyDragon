# Working Memory Concept for Python Code Generation

The working memory class for generating Python code is a specialized implementation of the general working memory concept in large language models (LLMs). It is designed to provide the LLM with a contextual and coherent framework for generating Python code based on user inputs, incorporating various components of the working memory, such as short-term memory, episodic memory, associative memory, goal definition, self-prompting, and output format definition.

In the context of the general LLM working memory class framework, this Python code generation working memory class aims to enhance the LLM's ability to understand, process, and generate code that adheres to user-defined goals and constraints. By leveraging the different memory components, the working memory class efficiently processes user input, retrieves relevant information, and iteratively refines the generated code until it aligns with the user's objectives.

 The working memory class includes the following agents:

1. Short-term Memory: GitMemory
2. Episodic Memory: CommitIndex
3. Associative Memory: External Code Repositories
4. Goal Definition: Meta-code
5. Self-prompting: Draft Script Iteration
6. Output Format Definition: Python Code

## Python Code Generation Workflow

1. User specifies the desired class or method objective in natural language.

2. GitMemory loads the user's current codebase into a vector storage, parsed with the libcst library.

3. A meta-code definition of the goal is generated.

4. A query into GitMemory and the associative memory (containing other libraries and resources) is used to create the context for the task.

5. A draft of the class or method is prepared.

## CommitIndex for Draft Script

1. The episodic memory of the draft script is handled like a git code modification.

2. Each writing operation corresponds to a commit, with a time-ordered vector index to handle each iteration, a commit name, and a message to describe the reason.

3. The git message contains a save of the state of the working memory that pushed it, so it can be replicated.

4. Both the target class and the meta-code definition of the class are stored in the CommitIndex structure for easy rollbacks and memory modifications.

## Short-term and Contextual Memory

1. The short-term memory of the agent is composed of the GitMemory of the repo hosting the code generation target.

2. The contextual memory is composed of other GitHub repositories and coding resources that the model has access to.

## HighLevel Specification of the WorkingMemory Loop
```
function codeGenerationIteration(userInput, gitMemory, commitIndex, contextualMemory):
    metaCode = generateMetaCode(userInput)

    # Concurrent processing - Step 1
    parallel {
        gitMemoryContext = loadGitMemory(gitMemory)
        commitContext = loadCommitContext(commitIndex)
        contextResources = loadContextualResources(contextualMemory)
    }

    draftCode = getCurrentDraft(commitContext)

    # Concurrent processing - Step 2
    parallel {
        modifications = generateModifications(draftCode, metaCode, contextResources)
        sourceCode = extractSourceCode(modifications, contextResources)
    }

    updatedDraft = applyModifications(draftCode, modifications)

    # Concurrent processing - Step 3
    parallel {
        validationResult = validateDraft(updatedDraft, sourceCode)
        comparisonResult = compareDraftWithObjective(updatedDraft, metaCode)
    }

    if validationResult and comparisonResult:
        storeDraftToCommitIndex(commitIndex, updatedDraft)
    else:
        updatedDraft = rollback(commitIndex)

    return updatedDraft
```

| Method                      | Input                                                                 | Output                                       | Function                                                               |
|-----------------------------|----------------------------------------------------------------------|----------------------------------------------|-----------------------------------------------------------------------|
| generateMetaCode(userInput) | Raw user input (text)                                                | Meta-code representation                    | Transforms the user input into a meta-code representation             |
| loadGitMemory(gitMemory)    | GitMemory                                                            | GitMemory context                            | Retrieves the GitMemory context for the code generation task          |
| loadCommitContext(commitIndex) | CommitIndex                                                         | Commit context                               | Retrieves the commit context for the current draft code               |
| loadContextualResources(contextualMemory) | ContextualMemory                                                  | Context resources                            | Loads contextual resources required for the code generation task      |
| getCurrentDraft(commitContext) | Commit context                                                     | Draft code                                   | Gets the current draft code from the commit context                   |
| generateModifications(draftCode, metaCode, contextResources) | Draft code, meta-code, context resources | Code modifications                           | Generates modifications to the draft code based on the meta-code and context resources |
| extractSourceCode(modifications, contextResources) | Code modifications, context resources    | Source code                                  | Extracts the source code from the modifications and context resources |
| applyModifications(draftCode, modifications) | Draft code, code modifications           | Updated draft code                           | Applies the modifications to the draft code                            |
| validateDraft(updatedDraft, sourceCode) | Updated draft code, source code          | Validation result                            | Validates the updated draft code and checks for consistency           |
| compareDraftWithObjective(updatedDraft, metaCode) | Updated draft code, meta-code           | Comparison result                            | Compares the updated draft code with the user's goal based on the meta-code |
| storeDraftToCommitIndex(commitIndex, updatedDraft) | CommitIndex, updated draft code         | Updated CommitIndex                          | Stores the updated draft code in the commit index                      |
| rollback(commitIndex)      | CommitIndex                                                          | Rolled back draft code                       | Rolls back the draft code to the previous state in the commit index   |

## HighLevel Specification of the WorkingMemory Loop with more details
```
function codeGenerationIteration(userInput, gitMemory, commitIndex, contextualMemory):
    metaCode = generateMetaCode(userInput)

    # Concurrent processing - Step 1
    parallel {
        gitMemoryContext = loadGitMemory(gitMemory)
        commitContext = loadCommitContext(commitIndex)
        contextResources = loadContextualResources(contextualMemory)
    }

    draftCode = getCurrentDraft(commitContext)

    # Concurrent processing - Step 2
    parallel {
        modifications = generateModifications(draftCode, metaCode, contextResources)
        sourceCode = extractSourceCode(modifications, contextResources)
    }

    updatedDraft = applyModifications(draftCode, modifications)

    # Concurrent processing - Step 3
    parallel {
        validationResult = validateDraft(updatedDraft, sourceCode)
        comparisonResult = compareDraftWithObjective(updatedDraft, metaCode)
    }

    if validationResult and comparisonResult:
        storeDraftToCommitIndex(commitIndex, updatedDraft)

        # Concurrent processing - Step 4
        parallel {
            testResults = runTests(updatedDraft)
            performanceMetrics = analyzePerformance(updatedDraft)
        }

        # Concurrent processing - Step 5
        parallel {
            documentation = generateDocumentation(updatedDraft)
            codeReviewFeedback = requestPeerReview(updatedDraft)
        }

        # If tests pass and performance metrics are acceptable, proceed
        if testResults and performanceMetrics:
            updatedDraft = incorporateFeedback(updatedDraft, codeReviewFeedback)
            finalizeDocumentation(documentation)
            storeFinalDraft(commitIndex, updatedDraft)
        else:
            updatedDraft = rollback(commitIndex)

    else:
        updatedDraft = rollback(commitIndex)

    return updatedDraft
```