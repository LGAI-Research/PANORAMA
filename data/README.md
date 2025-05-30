# PANORAMA Data Samples

This directory contains sample data for the PANORAMA project. The data includes various types of documents and related information used in the patent examination process.

## Directory Structure

```
data/
├── record/            # Patent application record files (rec_*.json)
├── parsed_CTNF/       # Parsed CTNF (patent rejection notice) files (pC_*.json)
├── parsed_SPEC/       # Parsed patent specification files
├── panorama/          # Integrated data for the PANORAMA project
└── benchmark/         # Data for benchmark testing
```

## Notes

- The data in this directory is **sample data**. While the content and format are identical to the data generated through actual code execution, folder names, file names, and storage locations may differ.

- Naming conventions for files in each folder:
  - record/: `rec_r[number]_[application_number].json` (e.g., rec_r00123_12345678.json)
  - parsed_CTNF/: `pC_r[number]*[application_number].json` (e.g., pC_r00123_12345678.json)
  - parsed_SPEC/: `spec_txt*[patent_id]\_parsed.json` (e.g., spec_txt_11058770_parsed.json)
  - panorama/: `panorama_r[record_number]_[application_number].json` (e.g., panorama_r00001_15091542.json)
  - benchmark/: See benchmark/README.md for detailed naming conventions for each benchmark type

## Requirements

- While maintaining the content and format of the original files, the paths and names referenced in the code may need appropriate adjustments.
