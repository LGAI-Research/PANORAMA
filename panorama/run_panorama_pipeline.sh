#!/bin/bash

set -e

echo "Starting record_generator.py..."
python panorama/record_generator/record_generator.py

echo "Finished record_generator.py."

echo "Starting CTNF_parser.py..."
python panorama/ctnf_parser/CTNF_parser.py

echo "Finished CTNF_parser.py."

echo "Starting CTNF_validation_and_add_PN.py..."
python panorama/ctnf_parser/CTNF_validation_and_add_PN.py

echo "Finished CTNF_validation_and_add_PN.py."

echo "Starting panorama_generator.py..."
python panorama/panorama_generator.py

echo "Finished panorama_generator.py."

echo "All scripts executed successfully." 