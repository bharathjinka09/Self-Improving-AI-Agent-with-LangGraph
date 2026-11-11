import operator
from typing import Annotated, List, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    task: str
    code: str
    test_result: str
    reflection: str
    iteration: int
    messages: Annotated[List[BaseMessage], operator.add]


def build_self_improving_graph(model_name: str = "gpt-4o", max_iterations: int = 3):
    llm = ChatOpenAI(model=model_name, temperature=0.2)

    def generate_node(state: AgentState) -> dict:
        task = state["task"]
        iteration = state.get("iteration", 0)
        reflection = state.get("reflection", "")

        prompt = f"Task: {task}\n"
        if reflection:
            prompt += (
                f"\nPrevious attempt failed with the following feedback/reflection:\n"
                f"{reflection}\n"
                "Write corrected Python code. Return ONLY valid executable Python within ```python ... ``` blocks."
            )
        else:
            prompt += "Write clean, robust Python code to solve the task. Include built-in assertions to test it."

        response = llm.invoke([
            SystemMessage(content="You are an expert Python developer tasked with writing self-contained code."),
            HumanMessage(content=prompt)
        ])

        return {
            "code": response.content,
            "iteration": iteration + 1,
            "messages": [response]
        }

    def evaluate_node(state: AgentState) -> dict:
        code_block = state.get("code", "")
        if "```python" in code_block:
            raw_code = code_block.split("```python")[1].split("```")[0].strip()
        else:
            raw_code = code_block.strip()

        try:
            exec_globals = {}
            exec(raw_code, exec_globals)
            test_result = "SUCCESS"
        except Exception as exc:
            test_result = f"FAILED: {type(exc).__name__}: {str(exc)}"

        return {"test_result": test_result}

    def reflect_node(state: AgentState) -> dict:
        reflection_prompt = f"""The following code failed execution:
```python
{state['code']}
```
Execution error:
{state['test_result']}

Analyze the root cause of this failure and write concrete, actionable guidance on how to fix it in the next attempt."""

        response = llm.invoke([
            SystemMessage(content="You are a senior code reviewer."),
            HumanMessage(content=reflection_prompt)
        ])

        return {"reflection": response.content}

    def should_continue(state: AgentState) -> Literal["reflect", "complete"]:
        if state.get("test_result") == "SUCCESS":
            return "complete"
        if state.get("iteration", 0) >= max_iterations:
            return "complete"
        return "reflect"

    workflow = StateGraph(AgentState)
    workflow.add_node("generator", generate_node)
    workflow.add_node("evaluator", evaluate_node)
    workflow.add_node("reflector", reflect_node)

    workflow.set_entry_point("generator")
    workflow.add_edge("generator", "evaluator")
    workflow.add_conditional_edges(
        "evaluator",
        should_continue,
        {
            "reflect": "reflector",
            "complete": END
        }
    )
    workflow.add_edge("reflector", "generator")

    return workflow.compile()
