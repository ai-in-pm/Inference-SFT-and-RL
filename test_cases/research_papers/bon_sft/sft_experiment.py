import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import torch
from bon_agent import BonAgent, BonConfig
from model_interfaces import OpenAIInterface
import json
from datetime import datetime

class SFTExperiment:
    def __init__(self, api_key):
        self.config = BonConfig(
            n_samples=32,
            temperature=0.8,
            learning_rate=1e-4,
            batch_size=16,
            max_epochs=100
        )
        self.model = OpenAIInterface(api_key=api_key)
        self.agent = BonAgent(self.config, self.model)
        
    def create_training_data(self):
        """Create synthetic training data for SFT"""
        training_data = []
        prompts = [
            "Write a function to sort a list",
            "Explain quantum computing",
            "Write a haiku about nature"
        ]
        
        for prompt in prompts:
            # Generate multiple responses
            responses = self.agent.generate_responses(prompt)
            
            # Simulate human labels (in practice, these would be human-rated)
            # Here we're using length as a simple proxy for quality
            scores = [len(response) for response in responses]
            labels = torch.tensor(scores) / max(scores)  # Normalize to [0,1]
            
            training_data.append({
                "prompt": prompt,
                "responses": responses,
                "labels": labels.tolist()
            })
        
        return training_data
    
    def train_sft(self, training_data):
        """Train the agent using supervised fine-tuning"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.__dict__,
            "training_metrics": [],
            "final_evaluation": None
        }
        
        # Training loop
        for epoch in range(self.config.max_epochs):
            epoch_loss = 0
            for data in training_data:
                batch = {
                    "prompts": [data["prompt"]] * len(data["responses"]),
                    "responses": data["responses"],
                    "labels": torch.tensor(data["labels"])
                }
                
                metrics = self.agent.train_step(batch)
                epoch_loss += metrics["loss"]
            
            avg_epoch_loss = epoch_loss / len(training_data)
            results["training_metrics"].append({
                "epoch": epoch,
                "loss": avg_epoch_loss
            })
        
        return results

def main():
    # Initialize experiment
    api_key = os.getenv("OPENAI_API_KEY")
    experiment = SFTExperiment(api_key)
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Run experiment
    print("Creating training data...")
    training_data = experiment.create_training_data()
    
    print("Training model...")
    results = experiment.train_sft(training_data)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"sft_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
