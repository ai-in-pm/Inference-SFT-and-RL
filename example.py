import logging
import os

from dotenv import load_dotenv

from bon_agent import BestOfNAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def code_generation_example():
    """Example of using BoN for code generation."""
    try:
        agent = BestOfNAgent(
            provider='openai',
            model='gpt-4',
            n_samples=3,
            temperature=0.8
        )
        
        prompt = """Write a Python implementation of a binary search tree with the following methods:
        - insert: Insert a new value
        - delete: Delete a value
        - search: Search for a value
        - inorder: Return inorder traversal
        Make it efficient and include docstrings."""
        
        logger.info("Generating code samples...")
        responses = agent.generate_responses(prompt)
        
        logger.info("Evaluating responses...")
        scores = agent.evaluate_responses(responses)
        
        best_response = agent.select_best(responses)
        logger.info("Selected best implementation:\n%s", best_response)
        
        return best_response
    
    except ValueError as e:
        logger.error("Configuration error: %s", str(e))
        return None
    except Exception as e:
        logger.error("Error in code generation: %s", str(e))
        return None

def creative_writing_example():
    """Example of using BoN for creative writing."""
    try:
        agent = BestOfNAgent(
            provider='openai',
            model='gpt-4',
            n_samples=5,
            temperature=1.0
        )
        
        prompt = """Write a short story (max 500 words) about an AI that becomes conscious. 
        Focus on the emotional journey and philosophical implications. 
        Make it thought-provoking but not cliché."""
        
        logger.info("Generating story variations...")
        responses = agent.generate_responses(prompt)
        
        logger.info("Evaluating responses...")
        scores = agent.evaluate_responses(responses, {
            'length': 0.2,
            'diversity': 0.4,
            'quality': 0.4
        })
        
        best_response = agent.select_best(responses)
        logger.info("Selected best story:\n%s", best_response)
        
        return best_response
    
    except ValueError as e:
        logger.error("Configuration error: %s", str(e))
        return None
    except Exception as e:
        logger.error("Error in creative writing: %s", str(e))
        return None

def technical_explanation_example():
    """Example of using BoN for technical explanations at different levels."""
    try:
        agent = BestOfNAgent(
            provider='openai',
            model='gpt-4',
            n_samples=4,
            temperature=0.7
        )
        
        base_prompt = "Explain quantum entanglement"
        audience_levels = [
            "to a 5-year-old",
            "to a high school student",
            "to a college physics major",
            "to a quantum physicist"
        ]
        
        all_explanations = {}
        
        for audience in audience_levels:
            prompt = f"{base_prompt} {audience}"
            logger.info("Generating explanation %s...", audience)
            
            responses = agent.generate_responses(prompt)
            best_response = agent.select_best(responses)
            all_explanations[audience] = best_response
        
        return all_explanations
    
    except ValueError as e:
        logger.error("Configuration error: %s", str(e))
        return None
    except Exception as e:
        logger.error("Error in technical explanation: %s", str(e))
        return None

def main():
    """Run all example cases."""
    try:
        results = {}
        
        # Code Generation
        logger.info("Running code generation example...")
        code = code_generation_example()
        results['code_generation'] = code
        
        # Creative Writing
        logger.info("\nRunning creative writing example...")
        story = creative_writing_example()
        results['creative_writing'] = story
        
        # Technical Explanation
        logger.info("\nRunning technical explanation example...")
        explanations = technical_explanation_example()
        results['technical_explanation'] = explanations
        
        # Log results
        logger.info("\nResults Summary:")
        for example, result in results.items():
            if result:
                logger.info(f"{example}: Success")
            else:
                logger.warning(f"{example}: Failed")
        
    except Exception as e:
        logger.error("Error running examples: %s", str(e))

if __name__ == "__main__":
    main()
