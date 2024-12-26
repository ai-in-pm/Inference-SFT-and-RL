import torch
from typing import List, Dict, Any, Tuple
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np
from bon_agent import BonAgent, BonConfig, BonEnvironment

class BonTrainer:
    """Trainer class for Best-of-N agent"""
    
    def __init__(self, agent: BonAgent, config: Dict[str, Any]):
        self.agent = agent
        self.config = config
        self.device = config['device']
        
    def train_sft(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """Train using supervised fine-tuning"""
        wandb.init(project="bon-sft", config=self.config)
        
        for epoch in range(self.config['max_epochs']):
            # Training loop
            self.agent.policy.train()
            train_losses = []
            
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                metrics = self.agent.train_step(batch)
                train_losses.append(metrics['loss'])
            
            # Validation
            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                wandb.log({
                    'epoch': epoch,
                    'train_loss': np.mean(train_losses),
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy']
                })
            else:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': np.mean(train_losses)
                })
    
    def train_rl(self, env: BonEnvironment, n_episodes: int):
        """Train using reinforcement learning"""
        wandb.init(project="bon-rl", config=self.config)
        
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Generate responses and select action
                responses = self.agent.generate_responses(state['prompt'])
                features = self.agent._extract_features(state['prompt'], responses)
                scores = self.agent.score_responses(responses, features)
                action_idx = torch.argmax(scores).item()
                action = responses[action_idx]
                
                # Take step in environment
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                
                # Update policy
                advantage = reward - scores.mean().item()  # Simple baseline
                loss = -scores[action_idx] * advantage
                
                self.agent.optimizer.zero_grad()
                loss.backward()
                self.agent.optimizer.step()
                
                state = next_state
            
            wandb.log({
                'episode': episode,
                'episode_reward': episode_reward
            })
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the agent"""
        self.agent.policy.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Generate and score responses
                responses = [self.agent.generate_responses(prompt) for prompt in batch['prompts']]
                features = self.agent._extract_features(batch['prompts'], responses)
                scores = self.agent.score_responses(responses, features)
                
                # Compute metrics
                predictions = torch.argmax(scores, dim=1)
                correct += (predictions == batch['labels']).sum().item()
                total += len(batch['labels'])
                
                if 'labels' in batch:
                    loss = torch.nn.functional.cross_entropy(scores, batch['labels'])
                    total_loss += loss.item()
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total
        }

def compute_pass_at_k(n_samples: List[int], n_correct: int, n: int) -> float:
    """Compute pass@k metric"""
    if n_correct == 0:
        return 0.0
    
    def n_choose_k(n: int, k: int) -> float:
        if k > n:
            return 0.0
        return float(np.prod(range(n - k + 1, n + 1))) / float(np.prod(range(1, k + 1)))
    
    def compute_prob(n: int, c: int, k: int) -> float:
        return 1.0 - float(n_choose_k(n - c, k)) / float(n_choose_k(n, k))
    
    return compute_prob(n_samples, n_correct, n)
