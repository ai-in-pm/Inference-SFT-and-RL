import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from bon_agent import BonAgent, BonConfig
from model_interfaces import OpenAIInterface, AnthropicInterface, GeminiInterface, MistralInterface, GroqInterface
import time
import json
from datetime import datetime
import numpy as np

class BenchmarkSuite:
    def __init__(self):
        self.config = BonConfig(
            n_samples=10,
            temperature=0.8
        )
        self.interfaces = {}
        
    def add_model(self, name, interface):
        """Add a model interface to benchmark"""
        self.interfaces[name] = BonAgent(self.config, interface)
    
    def run_latency_test(self, prompt, num_trials=5):
        """Measure response generation latency"""
        results = {}
        
        for name, agent in self.interfaces.items():
            latencies = []
            for _ in range(num_trials):
                start_time = time.time()
                _ = agent.generate_responses(prompt)
                latency = time.time() - start_time
                latencies.append(latency)
            
            results[name] = {
                "mean_latency": np.mean(latencies),
                "std_latency": np.std(latencies),
                "min_latency": np.min(latencies),
                "max_latency": np.max(latencies)
            }
        
        return results
    
    def run_quality_test(self, prompt):
        """Compare response quality across models"""
        results = {}
        
        for name, agent in self.interfaces.items():
            responses = agent.generate_responses(prompt)
            best_response = agent.select_best_response(prompt)
            
            results[name] = {
                "num_responses": len(responses),
                "avg_length": np.mean([len(r) for r in responses]),
                "best_response": best_response,
                "best_response_length": len(best_response)
            }
        
        return results
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        test_cases = [
            {
                "name": "code_generation",
                "prompt": "Write a Python function to calculate the factorial of a number"
            },
            {
                "name": "creative_writing",
                "prompt": "Write a short story about a robot learning to feel emotions"
            },
            {
                "name": "analytical_reasoning",
                "prompt": "Explain the concept of quantum entanglement to a high school student"
            }
        ]
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.__dict__,
            "latency_tests": {},
            "quality_tests": {}
        }
        
        for test_case in test_cases:
            print(f"Running benchmark for: {test_case['name']}")
            
            # Run latency test
            results["latency_tests"][test_case["name"]] = self.run_latency_test(
                test_case["prompt"]
            )
            
            # Run quality test
            results["quality_tests"][test_case["name"]] = self.run_quality_test(
                test_case["prompt"]
            )
        
        return results

def main():
    # Initialize benchmark suite
    suite = BenchmarkSuite()
    
    # Add available models
    if os.getenv("OPENAI_API_KEY"):
        suite.add_model("gpt4", OpenAIInterface(os.getenv("OPENAI_API_KEY")))
    
    if os.getenv("ANTHROPIC_API_KEY"):
        suite.add_model("claude", AnthropicInterface(os.getenv("ANTHROPIC_API_KEY")))
    
    if os.getenv("GOOGLE_API_KEY"):
        suite.add_model("gemini", GeminiInterface(os.getenv("GOOGLE_API_KEY")))
    
    if os.getenv("MISTRAL_API_KEY"):
        suite.add_model("mistral", MistralInterface(os.getenv("MISTRAL_API_KEY")))
    
    if os.getenv("GROQ_API_KEY"):
        suite.add_model("groq", GroqInterface(os.getenv("GROQ_API_KEY")))
    
    # Run benchmarks
    results = suite.run_full_benchmark()
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"benchmark_results_{timestamp}.json")
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Benchmark results saved to {results_file}")

if __name__ == "__main__":
    main()
