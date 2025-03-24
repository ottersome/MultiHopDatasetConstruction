#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script extracts entries with 3 and 4 hops from the MQuAKE-CF.json dataset
and saves them to a new JSON file.

Usage:
    python extract_multi_hop_data.py

Output:
    data/mquake_multi_hop.json - JSON file containing only entries with 3 or 4 hops
"""

import json
import os
from tqdm import tqdm

def main():
    # Input and output file paths
    input_file = "data/MQuAKE-CF.json"
    output_file = "data/mquake_4-3_multi_hop.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load the MQuAKE dataset
    print(f"Loading MQuAKE dataset from {input_file}...")
    with open(input_file, 'r') as f:
        mquake_data = json.load(f)
    
    print(f"Loaded {len(mquake_data)} entries. Extracting entries with 3 and 4 hops...")
    
    # List to hold entries with 3 or 4 hops
    multi_hop_entries = []
    
    # Extract entries with 3 or 4 hops
    for entry in tqdm(mquake_data):
        # Count the number of hops in this entry (number of triples)
        if "orig" in entry and "triples_labeled" in entry["orig"]:
            hops = len(entry["orig"]["triples_labeled"])
            
            # Check if this entry has 3 or 4 hops
            if hops in [3, 4]:
                # Extract the important information
                extracted_entry = {
                    "case_id": entry.get("case_id"),
                    "questions": entry.get("questions", []),
                    "answer": entry.get("answer"),
                    "num_hops": hops,
                    "triples_labeled": entry["orig"]["triples_labeled"]
                }
                
                multi_hop_entries.append(extracted_entry)
    
    # Save the extracted entries to a JSON file
    print(f"Extracted {len(multi_hop_entries)} entries with 3 or 4 hops.")
    
    if multi_hop_entries:
        with open(output_file, 'w') as f:
            json.dump(multi_hop_entries, f, indent=2)
        print(f"Saved extracted entries to {output_file}")
    else:
        print("No entries with 3 or 4 hops found in the dataset.")

if __name__ == "__main__":
    main() 