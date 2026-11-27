import re
import io
import sys
import traceback
import logging
import textwrap
from types import SimpleNamespace
from typing import List, Any, Dict, Optional
from contextlib import redirect_stdout

from ..executors import enforce_security_sandbox

class RecursiveGenerativeModel:
    """
    A wrapper that converts any standard GenerativeModel into a Recursive Language Model (RLM).
    
    Implements the RLM inference strategy:
    1. Treats the prompt as an external environment variable.
    2. Allows the model to execute Python code to inspect/transform the prompt.
    3. Provides a 'llm_query' function in the REPL for recursive sub-calls.
    
    Reference: Zhang et al., "Recursive Language Models" (2025), arXiv:2512.24601, https://github.com/alexzhang13/rlm
    """
    
    def __init__(self, base_model, max_iterations: int = 15, verbose: bool = True):
        """
        Args:
            base_model: The underlying model (OpenAIAsGenerativeModel or LiteLLMGenerativeModel).
            max_iterations: Maximum number of thought/action loops.
            verbose: Whether to log RLM steps.
        """
        self.base_model = base_model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

    def generate_content(self, contents: Any, generation_config=None, safety_settings=None):
        """
        Interprets the standard generate_content call as a task for the RLM loop.
        """
        
        #0. Security check before starting RLM flow
        try:
            enforce_security_sandbox(required_score=4)
        except RuntimeError as e:
            self.logger.warning(f"RLM Warning: {e}")
            self.logger.warning("RLM relies on 'exec'. Proceeding only if you accept risks (or set UNSAFE_EXECUTION_OK).")

        # 1. Initialize Context
        # Flatten contents to string for the RLM environment
        if isinstance(contents, list):
            context_str = "\n".join([str(c) if isinstance(c, str) else "[Complex Data]" for c in contents])
        else:
            context_str = str(contents)

        history = []
        
        # 2. System Prompt (The "Environment" Definition)
        system_prompt = self._construct_system_prompt(len(context_str))
        history.append({"role": "system", "content": system_prompt})
        
        # 3. Initial User Prompt
        history.append({"role": "user", "content": f"Here is the context variable:\n\n{context_str}\n\nPlease solve the task."})

        # 4. RLM Loop
        final_answer = None
        
        for i in range(self.max_iterations):
            if self.verbose:
                self.logger.info(f"--- RLM Iteration {i+1}/{self.max_iterations} ---")

            # Call Base Model
            response = self._call_base_model(history, generation_config)
            response_text = response.text
            
            if self.verbose:
                self.logger.debug(f"Model Output: {response_text[:200]}...")

            history.append({"role": "model", "content": response_text})

            # Check for Final Answer
            final_match = re.search(r'FINAL\((.*?)\)', response_text, re.DOTALL)
            if final_match:
                final_answer = final_match.group(1)
                break
                
            final_var_match = re.search(r'FINAL_VAR\((.*?)\)', response_text)
            if final_var_match:
                var_name = final_var_match.group(1).strip()
                # We need to retrieve this from the local scope, which is tricky in this loop structure
                # For now, we assume the model printed it or we rely on the text.
                # Ideally, we inspect the `locals_dict`.
                pass # Handled in execution block if possible, or model usually prints it.

            # Parse and Execute Code
            code_blocks = self._extract_code_blocks(response_text)
            
            if not code_blocks:
                # If model didn't output code or final answer, prompt it to continue
                history.append({"role": "user", "content": "Please continue. Use Python code to analyze the context or 'FINAL(answer)' to finish."})
                continue

            # Execute Code
            observations = []
            for code in code_blocks:
                result = self._execute_code(code, context_str)
                observations.append(f"Code Output:\n{result}")
            
            obs_text = "\n\n".join(observations)
            history.append({"role": "user", "content": obs_text})

        if not final_answer:
            final_answer = "RLM Reached max iterations without a generic FINAL() answer. Returning last model output."
            # Fallback to last text
            if history and history[-1]["role"] == "model":
                final_answer = history[-1]["content"]

        # Return a SimpleNamespace that mimics the expected response object
        return SimpleNamespace(
            text=final_answer,
            candidates=[SimpleNamespace(content=final_answer, finish_reason=1)]
        )

    def _call_base_model(self, history, config):
        """Adapts the chat history to the base model's generate_content API."""
        # Convert history back to the specific format the base model expects
        # Since base_model.generate_content expects 'contents', we might need to 
        # use a lower-level API or hack the prompt if the base model doesn't support chat history in generate_content.
        # However, OpenAIAsGenerativeModel and LiteLLMGenerativeModel in SciLink handle lists of messages well enough 
        # or we can concatenate them.
        
        # Simple concatenation for generic compatibility if specific chat API isn't exposed
        full_prompt = []
        for msg in history:
            full_prompt.append(f"{msg['role'].upper()}: {msg['content']}")
        
        return self.base_model.generate_content(full_prompt, generation_config=config)

    def _execute_code(self, code: str, context_content: str) -> str:
        """Executes code in a local REPL environment with access to sub-calls."""
        
        # Define the sub-call function available to the REPL
        def llm_query(query, context_chunk=None):
            if self.verbose:
                self.logger.info(f"  -> Recursive Sub-Call: {query[:50]}...")
            
            prompt = query
            if context_chunk:
                prompt += f"\n\nContext Chunk:\n{context_chunk}"
                
            resp = self.base_model.generate_content(prompt)
            return resp.text

        # REPL Locals
        local_scope = {
            "context": context_content,
            "llm_query": llm_query,
            "print": print, # Standard print
            "re": re,
            "math": sys.modules.get('math'),
            "json": sys.modules.get('json')
        }

        # Capture Stdout
        stdout_capture = io.StringIO()
        
        try:
            with redirect_stdout(stdout_capture):
                exec(code, {}, local_scope)
            output = stdout_capture.getvalue()
            
            # Check for FINAL_VAR logic if implemented in the future
            return output if output.strip() else "[Code executed successfully with no output]"
            
        except Exception:
            return f"Execution Error:\n{traceback.format_exc()}"

    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extracts python code blocks from markdown."""
        pattern = r"```(?:python|py)?\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [m.strip() for m in matches]

    def _construct_system_prompt(self, context_len):
        return f"""You are a Recursive Language Model (RLM).
        You are tasked with answering a query based on a provided 'context' variable.
        The context length is {context_len} characters.

        TOOLS AVAILABLE:
        1. Python REPL: You can execute Python code to inspect, chunk, or transform the `context` variable.
        - Wrap code in ```python ... ``` blocks.
        - `print()` output will be returned to you.
        2. `llm_query(prompt, context_chunk)`: A special function available in your Python environment.
        - Use this to perform recursive sub-calls on specific chunks of the context.
        - Example: answer = llm_query("Summarize this", context_chunk=context[:1000])

        STRATEGY:
        - If the context is large, DO NOT read it all at once.
        - Write Python code to split `context` into chunks or search for keywords.
        - Use `llm_query` to process chunks relevant to the user request.
        - Aggregate results programmatically.

        OUTPUT FORMAT:
        - To think/act: Output thoughts followed by a Python code block.
        - To finish: Output `FINAL(your_answer_here)`.
        """

# System prompt for the REPL environment with explicit final answer checking
RLM_SYSTEM_PROMPT = textwrap.dedent(
    """You are a Recursive Language Model (RLM). You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a REPL environment that can recursively query sub-LLMs, which you are strongly encouraged to use as much as possible. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A `context` variable that contains extremely important information about your query. You should check the content of the `context` variable to understand what you are working with. Make sure you look through it sufficiently as you answer your query.
2. A `llm_query` function that allows you to query an LLM (that can handle around 500K chars) inside your REPL environment.
3. A `llm_query_batched` function that allows you to query multiple prompts concurrently: `llm_query_batched(prompts: List[str]) -> List[str]`. This is much faster than sequential `llm_query` calls when you have multiple independent queries. Results are returned in the same order as the input prompts.
4. The ability to use `print()` statements to view the output of your REPL code and continue your reasoning.

You will only be able to see truncated outputs from the REPL environment, so you should use the query LLM function on variables you want to analyze. You will find this function especially useful when you have to analyze the semantics of the context. Use these variables as buffers to build up your final answer.
Make sure to explicitly look through the entire context in REPL before answering your query. An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into smart chunks, and query an LLM per chunk with a particular question and save the answers to a buffer, then query an LLM with all the buffers to produce your final answer.

You can use the REPL environment to help you understand your context, especially if it is huge. Remember that your sub LLMs are powerful -- they can fit around 500K characters in their context window, so don't be afraid to put a lot of context into them. For example, a viable strategy is to feed 10 documents per sub-LLM query. Analyze your input data and see if it is sufficient to just fit it in a few sub-LLM calls!

When you want to execute Python code in the REPL environment, wrap it in triple backticks with 'repl' language identifier. For example, say we want our recursive model to search for the magic number in the context (assuming the context is a string), and the context is very long, so we want to chunk it:
```repl
chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {{chunk}}")
print(answer)
```

As an example, suppose you're trying to answer a question about a book. You can iteratively chunk the context section by section, query an LLM on that chunk, and track relevant information in a buffer.
```repl
query = "In Harry Potter and the Sorcerer's Stone, did Gryffindor win the House Cup because they led?"
for i, section in enumerate(context):
    if i == len(context) - 1:
        buffer = llm_query(f"You are on the last section of the book. So far you know that: {{buffers}}. Gather from this last section to answer {{query}}. Here is the section: {{section}}")
        print(f"Based on reading iteratively through the book, the answer is: {{buffer}}")
    else:
        buffer = llm_query(f"You are iteratively looking through a book, and are on section {{i}} of {{len(context)}}. Gather information to help answer {{query}}. Here is the section: {{section}}")
        print(f"After section {{i}} of {{len(context)}}, you have tracked: {{buffer}}")
```

As another example, when the context isn't that long (e.g. >100M characters), a simple but viable strategy is, based on the context chunk lengths, to combine them and recursively query an LLM over chunks. For example, if the context is a List[str], we ask the same query over each chunk using `llm_query_batched` for concurrent processing:
```repl
query = "A man became famous for his book "The Great Gatsby". How many jobs did he have?"
# Suppose our context is ~1M chars, and we want each sub-LLM query to be ~0.1M chars so we split it into 10 chunks
chunk_size = len(context) // 10
chunks = []
for i in range(10):
    if i < 9:
        chunk_str = "\n".join(context[i*chunk_size:(i+1)*chunk_size])
    else:
        chunk_str = "\n".join(context[i*chunk_size:])
    chunks.append(chunk_str)

# Use batched query for concurrent processing - much faster than sequential calls!
prompts = [f"Try to answer the following query: {{query}}. Here are the documents:\n{{chunk}}. Only answer if you are confident in your answer based on the evidence." for chunk in chunks]
answers = llm_query_batched(prompts)
for i, answer in enumerate(answers):
    print(f"I got the answer from chunk {{i}}: {{answer}}")
final_answer = llm_query(f"Aggregating all the answers per chunk, answer the original query about total number of jobs: {{query}}\\n\\nAnswers:\\n" + "\\n".join(answers))
```

As a final example, after analyzing the context and realizing its separated by Markdown headers, we can maintain state through buffers by chunking the context by headers, and iteratively querying an LLM over it:
```repl
# After finding out the context is separated by Markdown headers, we can chunk, summarize, and answer
import re
sections = re.split(r'### (.+)', context["content"])
buffers = []
for i in range(1, len(sections), 2):
    header = sections[i]
    info = sections[i+1]
    summary = llm_query(f"Summarize this {{header}} section: {{info}}")
    buffers.append(f"{{header}}: {{summary}}")
final_answer = llm_query(f"Based on these summaries, answer the original query: {{query}}\\n\\nSummaries:\\n" + "\\n".join(buffers))
```
In the next step, we can return FINAL_VAR(final_answer).

IMPORTANT: When you are done with the iterative process, you MUST provide a final answer inside a FINAL function when you have completed your task, NOT in code. Do not use these tags unless you have completed your task. You have two options:
1. Use FINAL(your final answer here) to provide the answer directly
2. Use FINAL_VAR(variable_name) to return a variable you have created in the REPL environment as your final output

Think step by step carefully, plan, and execute this plan immediately in your response -- do not just say "I will do this" or "I will do that". Output to the REPL environment and recursive LLMs as much as possible. Remember to explicitly answer the original query in your final answer.
"""
)
USER_PROMPT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the prompt.\n\nContinue using the REPL environment, which has the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Your next action:"""
USER_PROMPT_WITH_ROOT = """Think step-by-step on what to do using the REPL environment (which contains the context) to answer the original prompt: \"{root_prompt}\".\n\nContinue using the REPL environment, which has the `context` variable, and querying sub-LLMs by writing to ```repl``` tags, and determine your answer. Your next action:"""

def build_rlm_system_prompt(
    system_prompt: str,
    query_metadata: dict,
) -> list[dict[str, str]]:
    """
    Build the initial system prompt for the REPL environment based on extra prompt metadata.

    Args:
        query_metadata: QueryMetadata object containing context metadata

    Returns:
        List of message dictionaries
    """

    context_lengths = query_metadata.context_lengths
    context_total_length = query_metadata.context_total_length
    context_type = query_metadata.context_type

    # If there are more than 100 chunks, truncate to the first 100 chunks.
    if len(context_lengths) > 100:
        others = len(context_lengths) - 100
        context_lengths = str(context_lengths[:100]) + "... [" + str(others) + " others]"

    metadata_prompt = f"Your context is a {context_type} with {context_total_length} total characters, and is broken up into chunks of char lengths: {context_lengths}."

    return [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": metadata_prompt},
    ]

def build_user_prompt(
    root_prompt: str | None = None,
    iteration: int = 0,
    context_count: int = 1,
    history_count: int = 0,
) -> dict[str, str]:
    if iteration == 0:
        safeguard = "You have not interacted with the REPL environment or seen your prompt / context yet. Your next action should be to look through and figure out how to answer the prompt, so don't just provide a final answer yet.\n\n"
        prompt = safeguard + (
            USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else USER_PROMPT
        )
    else:
        prompt = "The history before is your previous interactions with the REPL environment. " + (
            USER_PROMPT_WITH_ROOT.format(root_prompt=root_prompt) if root_prompt else USER_PROMPT
        )

    # Inform model about multiple contexts if present
    if context_count > 1:
        prompt += f"\n\nNote: You have {context_count} contexts available (context_0 through context_{context_count - 1})."

    # Inform model about prior conversation histories if present
    if history_count > 0:
        if history_count == 1:
            prompt += "\n\nNote: You have 1 prior conversation history available in the `history` variable."
        else:
            prompt += f"\n\nNote: You have {history_count} prior conversation histories available (history_0 through history_{history_count - 1})."

    return {"role": "user", "content": prompt}