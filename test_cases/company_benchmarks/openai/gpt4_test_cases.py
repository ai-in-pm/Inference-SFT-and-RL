import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from bon_agent import BonAgent, BonConfig
from model_interfaces import OpenAIInterface
import json
from datetime import datetime

class GPT4TestCases:
    def __init__(self, api_key):
        self.config = BonConfig(
            n_samples=5,
            temperature=0.8
        )
        self.model = OpenAIInterface(api_key=api_key)
        self.agent = BonAgent(self.config, self.model)
        
    def run_test_case(self, test_name, prompt, expected_criteria=None):
        """Run a single test case and record results"""
        results = {
            "test_name": test_name,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "n_samples": self.config.n_samples,
                "temperature": self.config.temperature
            },
            "prompt": prompt,
            "responses": [],
            "best_response": None,
            "metrics": {}
        }
        
        # Generate responses
        responses = self.agent.generate_responses(prompt)
        results["responses"] = responses
        
        # Get best response
        best_response = self.agent.select_best_response(prompt)
        results["best_response"] = best_response
        
        return results

def main():
    # Test cases
    test_cases = [
        {
            "name": "code_generation",
            "prompt": "Write a Python function to calculate the Fibonacci sequence",
            "criteria": ["correctness", "efficiency", "readability"]
        },
        {
            "name": "creative_writing",
            "prompt": "Write a short story about artificial intelligence in exactly 50 words",
            "criteria": ["creativity", "coherence", "word_count"]
        },
        {
            "name": "logical_reasoning",
            "prompt": "Solve this logical puzzle: If all A are B, and some B are C, what can we conclude about A and C?",
            "criteria": ["logical_validity", "clarity", "completeness"]
        }
    ]
    
    # Initialize test runner
    api_key = os.getenv("OPENAI_API_KEY")
    test_runner = GPT4TestCases(api_key)
    
    # Run tests and save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = []
    for test_case in test_cases:
        result = test_runner.run_test_case(
            test_case["name"],
            test_case["prompt"],
            test_case["criteria"]
        )
        all_results.append(result)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"test_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
