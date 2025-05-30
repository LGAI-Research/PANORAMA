# PANORAMA Benchmarks

This directory contains three benchmark tasks derived from the PANORAMA dataset, each representing a key step in the patent examination process. These benchmarks are designed to evaluate language models' capabilities in different aspects of patent analysis.

## Overview of Benchmark Tasks

The PANORAMA dataset captures the end-to-end examination workflow and the underlying reasons for patent applications. Based on real-world patent examination procedures, we divide this workflow into three benchmark tasks that replicate the main steps taken by examiners:

### 1. Prior-Art Retrieval for Patent Claims (PAR4PC)

**Task Description:** Select the document(s) from a pool of candidate prior-art documents that must be consulted to determine whether a target claim should be rejected.

**Directory:** [`par4pc/`](./par4pc/)

### 2. Paragraph Identification for Patent Claims (PI4PC)

**Task Description:** Given a claim and a prior-art document, identify the paragraph number within the document that should be compared with the claim when assessing patentability.

**Directory:** [`pi4pc/`](./pi4pc/)

### 3. Novelty and Non-Obviousness Classification for Patent Claims (NOC4PC)

**Task Description:** Given a claim and the cited prior-art documents with the relevant paragraphs, determine whether the claim is novel and non-obvious in relation to that prior art.

**Directory:** [`noc4pc/`](./noc4pc/)

## Using the Benchmarks

Each benchmark directory contains the following structure:

```
benchmarks/
├── par4pc/
│   ├── data/           # Sample data for testing
│   ├── inference.py    # Scripts for generating inference results
│   └── evaluation.py   # Scripts for evaluating the inference results
├── pi4pc/
│   ├── data/           # Sample data for testing
│   ├── inference.py    # Scripts for generating inference results
│   └── evaluation.py   # Scripts for evaluating the inference results
└── noc4pc/
    ├── data/           # Sample data for testing
    ├── inference.py    # Scripts for generating inference results
    └── evaluation.py   # Scripts for evaluating the inference results
```

### Data Format

The benchmark tasks use data from the PANORAMA dataset available on Hugging Face:

- [DxD-Lab/PANORAMA-NOC4PC-Bench](https://huggingface.co/datasets/DxD-Lab/PANORAMA-NOC4PC-Bench)
- [DxD-Lab/PANORAMA-PAR4PC-Bench](https://huggingface.co/datasets/DxD-Lab/PANORAMA-PAR4PC-Bench)
- [DxD-Lab/PANORAMA-PI4PC-Bench](https://huggingface.co/datasets/DxD-Lab/PANORAMA-PI4PC-Bench)

### Inference and Evaluation Process

For all benchmarks, the general process is:

1. Run the inference script to generate predictions
2. Run the evaluation script to evaluate the predictions against the ground truth

### Benchmark-Specific Commands

#### PAR4PC:

**Inference:**

```bash
python benchmarks/par4pc/inference.py --provider [provider] --model [model] --prompt_mode [mode]
```

**Evaluation:**

```bash
python benchmarks/par4pc/evaluation.py <path_to_inference_results.csv>
```

#### PI4PC:

**Inference:**

```bash
python benchmarks/pi4pc/inference.py --provider [provider] --model [model] --prompt_mode [mode]
```

**Evaluation:**

```bash
python benchmarks/pi4pc/evaluation.py <path_to_inference_results.csv>
```

#### NOC4PC:

**Inference:**

```bash
python benchmarks/noc4pc/inference.py --provider [provider] --model [model] --prompt_mode [mode]
```

**Evaluation:**

```bash
python benchmarks/noc4pc/evaluation.py <path_to_inference_results.csv>
```

### Dependencies

All benchmarks require the dependencies specified in the project's `requirements.txt` file. Additionally, the NOC4PC benchmark requires the BLEURT library for evaluation:

```bash
# Install BLEURT from GitHub
pip install git+https://github.com/google-research/bleurt.git

# If you encounter issues, you may need to clone and install manually:
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
```
