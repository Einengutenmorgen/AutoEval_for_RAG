# RAG Evaluation Dataset

A comprehensive framework for creating and maintaining evaluation datasets for Retrieval-Augmented Generation (RAG) systems.

## Project Overview

This project aims to create a robust evaluation framework for RAG systems by:
- Developing diverse question sets that test different aspects of retrieval and generation
- Implementing comprehensive metrics for retrieval accuracy, context relevance, and answer quality
- Providing tools for both automated and human evaluation
- Supporting iterative refinement of evaluation datasets

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- 100GB+ storage space for document corpus

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-organization/rag-evaluation-dataset.git
cd rag-evaluation-dataset
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

4. Configure your data sources:
```bash
cp config/data_config.example.yaml config/data_config.yaml
# Edit config/data_config.yaml with your settings
```

## Project Structure

- `config/`: Configuration files for data processing and evaluation
- `data/`: Raw and processed document corpus
- `question_sets/`: Question-answer pairs organized by type and complexity
- `evaluation/`: Evaluation metrics, rubrics, and results
- `scripts/`: Utility scripts for data processing and evaluation
- `notebooks/`: Exploratory analyses and demonstrations
- `docs/`: Project documentation

## Usage

### Data Processing

1. Configure your data sources in `config/data_config.yaml`
2. Run the data ingestion process:
```bash
python scripts/data_processing/ingest.py
```

3. Process the documents:
```bash
python scripts/data_processing/process.py
```

4. Create data splits:
```bash
python scripts/data_processing/split.py
```

### Question Generation

1. Create seed questions:
```bash
python scripts/question_generation/create_seeds.py
```

2. Generate additional questions:
```bash
python scripts/question_generation/generate.py
```

3. Validate questions:
```bash
python scripts/question_generation/validate.py
```

### Evaluation

1. Configure evaluation settings in `config/evaluation_config.yaml`
2. Run the evaluation:
```bash
python scripts/evaluation/run.py --system-name your_system_name
```

3. Generate reports:
```bash
python scripts/evaluation/report.py --results-file results.json
```

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Contributors and reviewers
- Organizations providing data sources
- Open-source libraries that made this project possible