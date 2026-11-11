# Self-Improving AI Agent with LangGraph

An implementation of a self-improving code generation agent built using LangGraph, LangChain, and OpenAI. The agent employs an **Actor-Critic-Reflection** cyclical workflow: it writes Python code, evaluates/executes it against test conditions, reflects on runtime errors or assertion failures, and iteratively refines its solution until completion or reaching an iteration ceiling.

## Project Structure

```text
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── __init__.py
│   ├── agent.py         # LangGraph workflow, state, nodes, and routing logic
│   └── main.py          # CLI runner and entrypoint
├── tests/
│   ├── __init__.py
│   └── test_agent.py    # Unit tests for graph execution
├── .gitignore
├── LICENSE
├── requirements.txt
├── setup.py
└── pyproject.toml
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bharathjinka09/Self-Improving-AI-Agent-with-LangGraph.git
   cd Self-Improving-AI-Agent-with-LangGraph
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

## Usage

Run the agent with a sample coding task:

```bash
python src/main.py
```

## Running Tests

```bash
pytest tests/
```

## License

MIT License. See [LICENSE](LICENSE) for details.
