# PANORAMA

# PANORAMA: A Dataset and Benchmark Tasks Capturing the Evaluation Trails and Rationales in Patent Examination

PANORAMA is a dataset of U.S. patent examination records designed to capture the complete patent examination process. It includes original patent applications, cited prior art references, rejection documents, and approval notices along with examiners' detailed rationales. From this dataset, we've developed three benchmark tasks that test different aspects of the patent examination workflow: Prior Art Retrieval for Patent Claims (PAR4PC), Patentability Identification for Patent Claims (PI4PC), and Novelty and Obviousness Characterization for Patent Claims (NOC4PC).

## 🛠️ Setup

### Dependencies

To run the PANORAMA data processing and benchmark scripts, you'll need Python 3.10 and the following dependencies:

```bash
python -m venv panorama_env
source panorama_env/bin/activate  # On Windows use: panorama_env\Scripts\activate
pip install -r requirements.txt
```

For evaluating NOC4PC benchmark results, you'll need to install the BLEURT package. Please refer to [google-research/bleurt](https://github.com/google-research/bleurt) for installation instructions and downloading the required pre-trained checkpoints.

### Directory Structure

The PANORAMA project is organized as follows:

```
panorama/
├── panorama_generator.py             # Main script for dataset generation
├── convert2bench_noc4pc.py           # Script to convert data for NOC4PC benchmark
├── convert2bench_par4pc.py           # Script to convert data for PAR4PC benchmark
├── convert2bench_pi4pc.py            # Script to convert data for PI4PC benchmark
├── run_panorama_pipeline.sh          # Shell script to run the complete pipeline
├── run_panorama_pipeline_without_record_generator.sh
├── record_generator/                 # Scripts for generating patent records
├── spec_parser/                      # Scripts for parsing patent specifications
└── ctnf_parser/                      # Scripts for parsing CTNF documents
```

## 🔄 Dataset Generation Pipeline

The PANORAMA dataset is generated through a multi-step pipeline that processes raw patent data into structured formats suitable for benchmarking tasks.

### Full Pipeline

To run the complete PANORAMA pipeline:

```bash
./run_panorama_pipeline.sh
```

This script executes all steps of the pipeline, including record generation, CTNF parsing, specification parsing, and benchmark conversion.

> **⚠️ Important Note:** Currently, there is an issue with the `patent_client` library due to migration errors. The USPTO Open Data Portal (https://data.uspto.gov/home) has recently undergone significant updates that changed their API request structure. As a result, `record_generator.py` may not function properly.

To address this issue, we've added a script that uses sample records instead:

```bash
./run_panorama_pipeline_without_record_generator.sh
```

Sample patent records are available in the `/data/record` directory. You can use these files for testing.


### Individual Pipeline Components

#### 1. Panorama Generation

The core dataset generation is handled by `panorama_generator.py`:

```bash
python panorama_generator.py --base_data_dir [path/to/data]
```

#### 2. Benchmark Conversion

Convert the PANORAMA dataset to specific benchmark formats:

**NOC4PC** (Novelty and Obviousness Characterization):

```bash
python panorama/convert2bench_noc4pc.py
```

**PAR4PC** (Prior Art Retrieval):

```bash
python panorama/convert2bench_par4pc.py
```

**PI4PC** (Patentability Identification):

```bash
python panorama/convert2bench_pi4pc.py
```

## 📜 License

This project is licensed under the CC-BY-NC-4.0.
