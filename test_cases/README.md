# Test Cases Overview

## Directory Structure and Test Cases

### 1. Company Benchmarks (`company_benchmarks/`)
- **OpenAI Tests** (`openai/`)
  - `gpt4_test_cases.py`: Basic test suite for GPT-4 implementation
  - Tests code generation, creative writing, and logical reasoning capabilities

### 2. Research Papers (`research_papers/`)
- **BoN-SFT** (`bon_sft/`)
  - `sft_experiment.py`: Supervised Fine-Tuning implementation
  - Includes synthetic training data generation
  - Tracks training metrics and model performance

### 3. Development (`development/`)
- **Performance Metrics** (`performance_metrics/`)
  - `benchmark_suite.py`: Comprehensive benchmarking tool
  - Measures latency and response quality across different models
  - Supports all implemented LLM providers

## Test Tracking

### Templates
- Located in `status_tracking/ongoing/test_tracking_template.md`
- Includes sections for:
  - Test information and metadata
  - Detailed test description and methodology
  - Results and metrics tracking
  - Analysis and next steps

### How to Use

1. **Running Company-Specific Tests**
   ```bash
   cd test_cases/company_benchmarks/openai
   python gpt4_test_cases.py
   ```

2. **Running Research Experiments**
   ```bash
   cd test_cases/research_papers/bon_sft
   python sft_experiment.py
   ```

3. **Running Performance Benchmarks**
   ```bash
   cd test_cases/development/performance_metrics
   python benchmark_suite.py
   ```

### Results
All test results are automatically saved in JSON format in their respective `results/` directories with timestamps for easy tracking and comparison.

## Adding New Tests

1. Copy the relevant template from existing test cases
2. Create a new directory under the appropriate category
3. Implement your test cases following the established patterns
4. Use the test tracking template to document progress

## Status Tracking

- Use the `status_tracking/` directory to maintain:
  - Ongoing tests
  - Completed tests
  - Planned future tests

## Next Steps

1. Implement additional test cases for specific use cases
2. Add automated result visualization
3. Create comparative analysis tools
4. Implement continuous integration testing
