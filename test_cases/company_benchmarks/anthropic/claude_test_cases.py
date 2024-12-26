import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from bon_agent import BonAgent, BonConfig
from model_interfaces import AnthropicInterface
import json
from datetime import datetime

class ClaudeTestCases:
    def __init__(self, api_key):
        self.config = BonConfig(
            n_samples=5,
            temperature=0.8
        )
        self.model = AnthropicInterface(api_key=api_key)
        self.agent = BonAgent(self.config, self.model)
    
    def run_test_suite(self):
        test_cases = [
            {
                "category": "code_generation",
                "tests": [
                    {
                        "name": "algorithm_implementation",
                        "prompt": "Implement a binary search algorithm in Python with detailed comments"
                    },
                    {
                        "name": "code_optimization",
                        "prompt": "Optimize this code snippet for performance: def fibonacci(n): return fibonacci(n-1) + fibonacci(n-2) if n > 1 else n"
                    }
                ]
            },
            {
                "category": "reasoning",
                "tests": [
                    {
                        "name": "mathematical_reasoning",
                        "prompt": "Solve this probability problem: If you roll two fair dice, what's the probability of getting a sum of 7?"
                    },
                    {
                        "name": "logical_deduction",
                        "prompt": "If all cats have tails, and Fluffy is a cat, what can we conclude? Explain your reasoning step by step."
                    }
                ]
            },
            {
                "category": "creative_writing",
                "tests": [
                    {
                        "name": "story_generation",
                        "prompt": "Write a 100-word story about an AI that learns to dream"
                    },
                    {
                        "name": "poetry",
                        "prompt": "Write a sonnet about the relationship between humans and artificial intelligence"
                    }
                ]
            }
        ]
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "model": "claude-3-sonnet",
            "config": self.config.__dict__,
            "results": {}
        }
        
        for category in test_cases:
            category_name = category["category"]
            results["results"][category_name] = {}
            
            for test in category["tests"]:
                print(f"Running test: {test['name']}")
                
                # Generate multiple responses
                responses = self.agent.generate_responses(test["prompt"])
                
                # Select best response
                best_response = self.agent.select_best_response(test["prompt"])
                
                results["results"][category_name][test["name"]] = {
                    "prompt": test["prompt"],
                    "responses": responses,
                    "best_response": best_response,
                    "metrics": {
                        "avg_length": sum(len(r) for r in responses) / len(responses),
                        "response_diversity": len(set(responses)) / len(responses),
                        "best_response_length": len(best_response)
                    }
                }
        
        return results

def main():
    # Initialize test runner
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment variables")
        return
    
    test_runner = ClaudeTestCases(api_key)
    
    # Run tests
    print("Running Claude test suite...")
    results = test_runner.run_test_suite()
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"claude_results_{timestamp}.json")
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
