# Best-of-N Inference-Aware Fine-Tuning

This repository implements a Best-of-N (BoN) strategy for inference-aware fine-tuning of large language models. The system supports multiple leading LLM providers and includes comprehensive testing and benchmarking capabilities.

The development of this GitHub Repository was inspired by the "Inference-Aware Fine-Tuning for Best-of-N
Sampling in Large Language Models" Paper. To read the entire paper. visit https://arxiv.org/pdf/2412.15287

## Features

- **Best-of-N Strategy**: Generate multiple responses and select the best one based on configurable criteria
- **Multi-Provider Support**:
  - OpenAI (GPT-4, GPT-3.5)
  - Anthropic (Claude)
  - Google (Gemini)
  - Mistral
  - Groq
- **Rate Limiting**: Token bucket algorithm for API request rate control
- **Error Handling**: Comprehensive error handling with retry mechanism
- **Response Evaluation**: Sophisticated response evaluation based on multiple criteria

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/bon-inference-ft.git
cd bon-inference-ft
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

The main component is the `BestOfNAgent` class which implements the Best-of-N strategy:

```python
from bon_agent import BestOfNAgent

agent = BestOfNAgent(
    provider='openai',
    model='gpt-4',
    n_samples=3,
    temperature=0.8
)

responses = agent.generate_responses(prompt)
best_response = agent.select_best(responses)
```

### Example Use Cases

The repository includes several example use cases demonstrating different applications:

1. **Code Generation**

```python
# Generate a binary search tree implementation
prompt = """Write a Python implementation of a binary search tree with methods:
- insert: Insert a new value
- delete: Delete a value
- search: Search for a value
- inorder: Return inorder traversal"""

responses = agent.generate_responses(prompt)
best_implementation = agent.select_best(responses)
```

Example output:

```python
class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

class BST:
    def __init__(self):
        self.root = None

    def insert(self, key):
        """Insert a new value"""
        if self.root is None:
            self.root = Node(key)
        else:
            self._insert(self.root, key)

    # ... (see example.py for full implementation)
```

2. **Creative Writing**

```python
# Generate a story about AI consciousness
prompt = """Write a short story about an AI that becomes conscious.
Focus on emotional journey and philosophical implications."""

responses = agent.generate_responses(prompt)
best_story = agent.select_best(responses)
```

3. **Technical Explanations**

```python
# Explain quantum entanglement at different levels
base_prompt = "Explain quantum entanglement"
audience_levels = [
    "to a 5-year-old",
    "to a high school student",
    "to a college physics major",
    "to a quantum physicist"
]

for audience in audience_levels:
    prompt = f"{base_prompt} {audience}"
    responses = agent.generate_responses(prompt)
    best_explanation = agent.select_best(responses)
```

## Architecture

### Components

1. **BestOfNAgent** (`bon_agent.py`)

   - Main class implementing the Best-of-N strategy
   - Handles provider-specific API calls
   - Implements response selection logic

2. **Utilities** (`utils.py`)

   - Rate limiting implementation
   - Retry mechanism with exponential backoff
   - Error handling and validation

3. **Examples** (`example.py`)
   - Demonstrates different use cases
   - Shows how to configure and use the agent

### Response Selection

The agent uses multiple criteria to select the best response:

- **Length**: Prefers responses of moderate length
- **Diversity**: Measures uniqueness compared to other responses
- **Quality**: Basic metrics like sentence structure and formatting

## Configuration

Configuration is done through environment variables in `.env`:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_key

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_key

# Google
GOOGLE_API_KEY=your_google_key

# Mistral
MISTRAL_API_KEY=your_mistral_key

# Groq
GROQ_API_KEY=your_groq_key
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT models
- Anthropic for Claude
- Google for Gemini
- Mistral AI
- Groq
