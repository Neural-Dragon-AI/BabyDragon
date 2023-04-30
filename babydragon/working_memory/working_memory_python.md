# Working Memory Concept for Python Code Generation

The working memory class for generating Python code is a specialized
implementation of the general working memory concept in large language models
(LLMs). It is designed to provide the LLM with a contextual and coherent
framework for generating Python code based on user inputs, incorporating various
components of the working memory, such as short-term memory, episodic memory,
associative memory, goal definition, self-prompting, and output format
definition.

In the context of the general LLM working memory class framework, this Python
code generation working memory class aims to enhance the LLM's ability to
understand, process, and generate code that adheres to user-defined goals and
constraints. By leveraging the different memory components, the working memory
class efficiently processes user input, retrieves relevant information, and
iteratively refines the generated code until it aligns with the user's
objectives.

The working memory class includes the following agents:

1. Short-term Memory: GitMemory - Stores the current state of the user's
   codebase.
2. Episodic Memory: CommitIndex - Manages the history of draft script
   iterations, mimicking git code modifications.
3. Associative Memory: External Code Repositories - Contains external libraries
   and resources for providing context during code generation.
4. Goal Definition: Meta-code - Represents the user's goal in a structured
   format that guides the code generation process.
5. Self-prompting: Draft Script Iteration - Enables the model to iteratively
   refine the draft script based on user input, context, and goal.
6. Output Format Definition: Python Code - Ensures the generated code conforms
   to Python syntax and style guidelines.

## Python Code Generation Workflow

1. User provides the desired class or method objective in natural language.
2. GitMemory imports the user's current codebase into a vector storage,
   utilizing the libcst library for parsing.
3. A meta-code representation of the user's goal is generated.
4. A query leveraging GitMemory and the associative memory (which includes
   external libraries and resources) establishes the task context.
5. An initial draft of the class or method is created based on the context.
6. Modifications to the draft code are generated, considering the meta-code and
   context resources.
7. The updated draft code is validated for consistency and compared with the
   user's goal based on the meta-code.
8. If the updated draft passes validation and meets the user's goal, it is
   stored in the commit index. Otherwise, the draft is rolled back to the
   previous state.

## CommitIndex for Draft Script

1. The episodic memory for the draft script is managed similarly to a git code
   modification.
2. Each writing operation is treated as a commit, with a time-ordered vector
   index for tracking iterations, a commit name, and a message explaining the
   rationale.
3. The git message stores the state of the working memory that triggered the
   commit, enabling replication.
4. Both the target class and the meta-code representation are maintained within
   the CommitIndex structure, simplifying rollbacks and memory modifications.

## Short-term and Contextual Memory

1. The short-term memory of the agent comprises the GitMemory of the repository
   containing the code generation target.
2. Contextual memory consists of additional GitHub repositories and coding
   resources accessible to the model.

## Additional Workflow Steps

1. If the updated draft is stored in the commit index, tests are run, and
   performance metrics are analyzed.
2. Documentation is generated, and a peer review is requested for the updated
   draft.
3. If tests pass and performance metrics meet expectations, the updated draft
   incorporates feedback from the peer review, documentation is finalized, and
   the final draft is stored in the commit index.
4. If tests fail or performance metrics are unsatisfactory, the draft is rolled
   back to the previous state.

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

| Method                                                       | Input                                    | Output                   | Function                                                                               |
| ------------------------------------------------------------ | ---------------------------------------- | ------------------------ | -------------------------------------------------------------------------------------- |
| generateMetaCode(userInput)                                  | Raw user input (text)                    | Meta-code representation | Transforms the user input into a meta-code representation                              |
| loadGitMemory(gitMemory)                                     | GitMemory                                | GitMemory context        | Retrieves the GitMemory context for the code generation task                           |
| loadCommitContext(commitIndex)                               | CommitIndex                              | Commit context           | Retrieves the commit context for the current draft code                                |
| loadContextualResources(contextualMemory)                    | ContextualMemory                         | Context resources        | Loads contextual resources required for the code generation task                       |
| getCurrentDraft(commitContext)                               | Commit context                           | Draft code               | Gets the current draft code from the commit context                                    |
| generateModifications(draftCode, metaCode, contextResources) | Draft code, meta-code, context resources | Code modifications       | Generates modifications to the draft code based on the meta-code and context resources |
| extractSourceCode(modifications, contextResources)           | Code modifications, context resources    | Source code              | Extracts the source code from the modifications and context resources                  |
| applyModifications(draftCode, modifications)                 | Draft code, code modifications           | Updated draft code       | Applies the modifications to the draft code                                            |
| validateDraft(updatedDraft, sourceCode)                      | Updated draft code, source code          | Validation result        | Validates the updated draft code and checks for consistency                            |
| compareDraftWithObjective(updatedDraft, metaCode)            | Updated draft code, meta-code            | Comparison result        | Compares the updated draft code with the user's goal based on the meta-code            |
| storeDraftToCommitIndex(commitIndex, updatedDraft)           | CommitIndex, updated draft code          | Updated CommitIndex      | Stores the updated draft code in the commit index                                      |
| rollback(commitIndex)                                        | CommitIndex                              | Rolled back draft code   | Rolls back the draft code to the previous state in the commit index                    |

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
