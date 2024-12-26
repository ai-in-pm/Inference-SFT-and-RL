import os

def create_directory_structure():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_cases_dir = os.path.join(base_dir, 'test_cases')
    
    # Main directories
    directories = {
        'company_benchmarks': [
            'openai',
            'anthropic',
            'google',
            'mistral',
            'groq'
        ],
        'research_papers': [
            'bon_sft',
            'bon_rl',
            'comparative_studies'
        ],
        'development': [
            'performance_metrics',
            'model_comparisons',
            'integration_tests'
        ],
        'status_tracking': [
            'ongoing',
            'completed',
            'planned'
        ]
    }
    
    # Create main test_cases directory
    os.makedirs(test_cases_dir, exist_ok=True)
    
    # Create subdirectories
    for main_dir, subdirs in directories.items():
        main_path = os.path.join(test_cases_dir, main_dir)
        os.makedirs(main_path, exist_ok=True)
        
        # Create README for each main directory
        with open(os.path.join(main_path, 'README.md'), 'w') as f:
            f.write(f"# {main_dir.replace('_', ' ').title()}\n\n")
            f.write("## Overview\n")
            f.write(f"This directory contains {main_dir.replace('_', ' ')} related test cases and documentation.\n\n")
            f.write("## Subdirectories\n")
            for subdir in subdirs:
                f.write(f"- `{subdir}`: {subdir.replace('_', ' ').title()}\n")
        
        # Create subdirectories
        for subdir in subdirs:
            subdir_path = os.path.join(main_path, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            
            # Create README for each subdirectory
            with open(os.path.join(subdir_path, 'README.md'), 'w') as f:
                f.write(f"# {subdir.replace('_', ' ').title()}\n\n")
                f.write("## Status\n")
                f.write("- [ ] In Progress\n")
                f.write("- [ ] Completed\n\n")
                f.write("## Test Cases\n")
                f.write("*Add test cases here*\n\n")
                f.write("## Results\n")
                f.write("*Document results here*\n")

if __name__ == "__main__":
    create_directory_structure()
