import os
from src.agent import build_self_improving_graph


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Note: Set OPENAI_API_KEY environment variable before running.")

    app = build_self_improving_graph()

    task = (
        "Write a Python function `flatten_nested(lst)` that flattens a deeply "
        "nested list of arbitrary depth into a flat list, and write assertions "
        "testing standard lists, empty lists, and deeply nested lists."
    )

    print(f"Executing self-improving workflow for task:\n{task}\n")
    initial_state = {
        "task": task,
        "code": "",
        "test_result": "",
        "reflection": "",
        "iteration": 0,
        "messages": []
    }

    final_state = app.invoke(initial_state)

    print("=" * 60)
    print(f"Iterations: {final_state.get('iteration')}")
    print(f"Final Test Result: {final_state.get('test_result')}")
    print("=" * 60)
    print("Final Code Output:")
    print(final_state.get("code"))


if __name__ == "__main__":
    main()
