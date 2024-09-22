# langchain-mcts

<p align="center"
  <a href="https://github.com/billyvinning/langchain-mcts/actions?query=workflows:Unit Tests" target="_blank">
    <img src="https://github.com/billyvinning/langchain-mcts/actions/workflows/test.yaml/badge.svg" alt="GitHub Actions Tests Workflow.">
  </a>
  <a href="https://www.python.org" target="_blank">
    <img src="https://img.shields.io/badge/Python-3.10 | 3.11 | 3.12-blue.svg?style=flat&logo=python&logoColor=white" alt="Supported Python versions.">
  </a>
  <a href="https://github.com/pre-commit/pre-commit" target="_blank">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white" alt="pre-commit">
  </a>
  <a href="http://mypy-lang.org/" target="_blank">
    <img src="http://www.mypy-lang.org/static/mypy_badge.svg" alt="Checked with mypy">
  </a>
  <a href="https://github.com/astral-sh/ruff" target="_blank">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://interrogate.readthedocs.io/en/latest/" target="_blank">
    <img src=".github/interrogate-badge.svg" alt="Interrogate">
  </a>
  <a href="https://coverage.readthedocs.io/en/latest/" target="_blank">
    <img src=".github/coverage-badge.svg" alt="Coverage">
  </a>
</p>


An implementation of Monte Carlo Tree Search with Self-Refine (MCTS-SR) in LangGraph, reproducing MathBlackBox.

## Installation

Install directly from GitHub with `pip` via:

```bash
pip install git+https://github.com/billyvinning/langchain-mcts
```

Alternatively, install from a local clone with `pip` via:

```bash
git clone git@github.com:billyvinning/langchain-mcts.git
cd langchain-mcts
pip install .
```


## Usage


Firstly, initialise your favourite LLM; this example uses Gemini 1.5 Flash from Google's Vertex AI.

```python
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(model_name="gemini-1.5-flash")
```


## License

This project is licensed under the MIT license, see `LICENSE.md` for more information.
