#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 20:38:04 2024

@author: ottersome

Summary: 
This script expands the MQuAKE triplet dataset by querying WikiData to create a larger
knowledge graph dataset comparable to FB15k in size. It handles entity extraction from MQuAKE,
expands entity relationships through WikiData API calls, and processes the resulting triplets
to create a high-quality dataset for knowledge graph embedding tasks.

Key functionalities:
1. **Entity & Relation Extraction**: Extracts entities and relations from MQuAKE dataset
2. **Neighborhood Expansion**: Expands the entity set through N-hop neighbors
3. **WikiData Querying**: Uses WikiData API to retrieve all relationships for entities
4. **Triplet Processing**: Handles filtering, cleaning, and deduplication
5. **Dataset Creation**: Splits into train/test/validation for knowledge graph embedding

This pipeline can be run end-to-end or in stages:
- **Entity Extraction**: Extract initial entities from MQuAKE
- **Entity Expansion**: Expand the entity set through WikiData neighbors
- **Triplet Generation**: Query WikiData for all relationships within the entity set
- **Dataset Creation**: Create the final processed dataset

Usage:
    python mquake_triplet_process.py --mode extract_entities --mquake_path [path_to_mquake]
    python mquake_triplet_process.py --mode expand_entities --entity_file [path_to_entities]
    python mquake_triplet_process.py --mode generate_triplets --entity_file [path_to_expanded_entities]
    python mquake_triplet_process.py --mode create_dataset --triplet_file [path_to_raw_triplets]
    python mquake_triplet_process.py --mode full_pipeline --mquake_path [path_to_mquake]
"""

import argparse
import csv
import json
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from numpy.random import f
import pandas as pd
import requests
from tqdm import tqdm
from wikidata.client import Client

from utils.basic import load_to_set, save_triplets, str2bool
from utils.logging import create_logger
from utils.process_triplets import (
    clean_triplet_relations,
    extract_triplet_sets,
    filter_triplets_by_entities,
    split_triplets,
)
from utils.wikidata_v2 import fetch_entity_triplet, process_entity_triplets, retry_fetch


def parse_args():
    """Parse command line arguments for the MQuAKE triplet expansion pipeline."""
    parser = argparse.ArgumentParser(description="Expand MQuAKE triplets by querying WikiData")

    # Pipeline mode selection
    parser.add_argument('--mode', type=str, 
                        choices=['extract_entities', 'expand_entities', 'generate_triplets', 'process_triplets', 'create_dataset', 'full_pipeline'],
                        default='full_pipeline', help='Operating mode for the pipeline')

    # MQuAKE input paths
    parser.add_argument('--mquake_path', type=str, default='./data/MQuAKE-CF.json',
                        help='Path to the MQuAKE dataset file')

    # Entity/relation paths
    parser.add_argument('--og_entity_output', type=str, default='./data/mquake/entities.txt',
                        help='Path to save the extracted entities')
    parser.add_argument('--og_relation_output', type=str, default='./data/mquake/relations.txt',
                        help='Path to save the extracted relations')
    parser.add_argument('--cf_entity_output', type=str, default='./data/mquake/cf_entities.txt',
                        help='Path to save the extracted entities')
    parser.add_argument('--cf_relation_output', type=str, default='./data/mquake/cf_relations.txt',
                        help='Path to save the extracted relations')
    parser.add_argument('--rr_entity_output', type=str, default='./data/mquake/rr_entities.txt',
                        help='Path to save the extracted entities from counterfactual set')
    parser.add_argument('--rr_relation_output', type=str, default='./data/mquake/rr_relations.txt',
                        help='Path to save the extracted relations from counterfactual set')

    parser.add_argument('--expanded_triplet_output', type=str, default='./data/mquake/expanded_triplets.csv',
                        help='Path to save the expanded entity set')

    # Triplet processing parameters
    parser.add_argument('--raw_triplet_output', type=str, default='./data/mquake/raw_triplets.txt',
                        help='Path to save raw triplets from WikiData querying')
    parser.add_argument('--processed_triplet_output', type=str, default='./data/mquake/processed_triplets.txt',
                        help='Path to save processed triplets')

    # Dataset creation parameters
    parser.add_argument('--train_output', type=str, default='./data/mquake/train.txt',
                        help='Path to save the training triplets')
    parser.add_argument('--test_output', type=str, default='./data/mquake/test.txt',
                        help='Path to save the test triplets')
    parser.add_argument('--valid_output', type=str, default='./data/mquake/valid.txt',
                        help='Path to save the validation triplets')

    # Entity expansion parameters
    parser.add_argument('--expansion_hops', type=int, default=1,
                        help='Number of hops to expand from core entities')
    parser.add_argument('--target_entity_count', type=int, default=15000,
                        help='Target number of entities to include in the dataset')
    parser.add_argument('--entity_batch_size', type=int, default=1000,
                        help='Number of entities to process in each batch for expansion')

    # Triplet generation parameters
    parser.add_argument('--max_workers', type=int, default=20,
                        help='Maximum number of concurrent workers for WikiData API queries')
    parser.add_argument('--max_retries', type=int, default=3,
                        help='Maximum number of retries for failed WikiData queries')
    parser.add_argument('--timeout', type=int, default=5,
                        help='Timeout in seconds for WikiData queries')
    parser.add_argument('--target_triplet_count', type=int, default=200000,
                        help='Target number of triplets for the final dataset')

    # Dataset split parameters
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Proportion of triplets to use for training')
    parser.add_argument('--test_valid_ratio', type=float, default=0.5,
                        help='Proportion of non-training triplets to use for testing vs validation')

    # Additional options
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()


def extract_mquake_entities(
    mquake_path: str,
) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    """
    Extract WikiData entity and relation IDs from MQuAKE questions and answers.
    
    Args:
        mquake_path: Path to the MQuAKE dataset file (standard JSON)
    
    Returns:
        Tuple containing:
        - Set of entity IDs (Q-prefixed) extracted from MQuAKE
        - Set of relation IDs (P-prefixed) extracted from MQuAKE
        - Set of entity IDs (Q-prefixed) extracted from MQuAKE counterfactual set
        - Set of relation IDs (P-prefixed) extracted from MQuAKE counterfactual set
    """
    entities = set()  # For Q-prefixed entity IDs
    relations = set()  # For P-prefixed relation IDs

    rr_entities = set()
    rr_relations = set()

    # Process the main MQuAKE file
    print(f"Extracting entities and relations from MQuAKE: {mquake_path}")

    with open(mquake_path, 'r') as f:
        # Load the entire JSON file (array of examples)
        mquake_data = json.load(f)

    # Extract entity IDs from requested_rewrite
    for i, example in tqdm(enumerate(mquake_data), desc="Extracting data"):
        # TEST: Assumption to test if there are rewrites and orig triplets on each examplet
        assert (
            "requested_rewrite" in example
            and "orig" in example
            and "triples" in example["orig"]
        ), "Not every sample provided contains request_rewrite and original triplets"

        for h,r,t in example["orig"]["triples"]:
            entities.add(h)
            relations.add(r)
            entities.add(t)

        #DEBUG:
        for h,r,t in example["orig"]["triples_labeled"]:
            logger.debug(f"Head: {h}, Realtion: {r}, Tail: {t}")

        for rewrite in example["requested_rewrite"]:
            # Extract relation ID (Wikidata Property)
            if "relation_id" in rewrite and rewrite["relation_id"].startswith("P"):
                rr_relations.add(rewrite["relation_id"])

            # Extract target entity IDs
            if "target_new" in rewrite and "id" in rewrite["target_new"] and rewrite["target_new"]["id"].startswith("Q"):
                rr_entities.add(rewrite["target_new"]["id"])

    # Process test set if provided (for additional entities)
    print(f"Extracted {len(entities)} unique entities (Q-prefixed)")
    print(f"Extracted {len(relations)} unique relations (P-prefixed)")

    print(f"Extracted {len(rr_entities)} unique entities (Q-prefixed) from counterfactual set")
    print(f"Extracted {len(rr_relations)} unique relations (P-prefixed) from counterfactual set")

    return entities, relations, rr_entities, rr_relations

def _batch_entity_set_expansion(
    batch: list[str],
    max_workers: int,
    client: Client,
    max_retries: int,
    timeout: int,
    use_qualifiers_for_expansion: bool,
) -> tuple[set[tuple[str, str, str]], dict]:
    """
    Batch-wise entity expansion

    Returns (2)
    ---------
    - newfound_triplets: Set of new triplets
    - qualifier_dictionary: Dictionary of qualifier triplets
    """

    newfound_triplets: set[tuple[str, str, str]] = set()
    qualifier_dictionary = dict()
    # Use ThreadPoolExecutor to fetch neighbors concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        # Submit tasks to fetch neighbors for each entity in the batch
        futures = {
            executor.submit(
                retry_fetch,
                fetch_entity_triplet,
                entity,
                client,
                "expanded" if use_qualifiers_for_expansion else "separate",
                max_retries=max_retries,
                timeout=timeout,
            ): entity
            for entity in batch
        }
        
        # Process results as they complete
        for future in as_completed(futures):
            entity = futures[future]
            try:
                triplets, qualifier_triplet_dict = future.result(timeout=timeout)

                newfound_triplets.update(triplets)
                qualifier_dictionary.update(qualifier_triplet_dict)

            except Exception as e:
                print(f"Error processing entity {entity}: \n\t{e}")
                # Unrecoverable error
                exit(-1)

    return newfound_triplets, qualifier_dictionary

def expand_triplet_set(
    entity_set: Set[str],
    target_size: int,
    expansion_hops: int,
    batch_size: int,
    max_workers: int, 
    max_retries: int,
    timeout: int,
) -> tuple[set[tuple[str, str, str]], dict[tuple,Any]]:
    """
    Expand entity and relation set by finding neighbors of existing entities through WikiData.
    And then create new triplets from the expanded entity set.
    
    Args:
        entity_set: Initial set of entities to expand from
        target_size: Target number of entities in the expanded set
        expansion_hops: Number of hops to expand (careful with values > 1)
        batch_size: Number of entities to process in each batch
        max_workers: Maximum number of concurrent workers for WikiData API queries
        max_retries: Maximum number of retries for failed WikiData queries
        timeout: Timeout in seconds for WikiData queries
    
    Returns:
        Expanded set of entities
    """
    logger.info(f"Target size: {target_size}, Expansion hops: {expansion_hops}")
    
    # Initialize the expanded entity set with the initial entities
    expanded_triplets: set[tuple[str,str,str]] = set()
    qualifier_dictionary = dict()
    entities_to_process = list(entity_set)
    processed_entities = set()
    
    client = Client()
    
    num_ogEntities_processed = 0
    
    # Process in hops
    for hop in range(expansion_hops):
        if len(expanded_triplets) >= target_size:
            break
            
        print(f"Beginning hop {hop+1} with {len(entities_to_process)} entities to process")
        
        # Track neighbors found in this hop
        new_neighbors = set()
        # Process in batches
        batch_num = 0
        _exit = False
        for batch_start in range(0, len(entities_to_process), batch_size):

            batch_end = min(batch_start + batch_size, len(entities_to_process))
            batch = entities_to_process[batch_start:batch_end]
            batch = [entity for entity in batch if entity not in processed_entities]

            newfound_triplets, _qualifier_dictionary = _batch_entity_set_expansion(
                batch,
                max_workers,
                client,
                max_retries,
                timeout,
                use_qualifiers_for_expansion=False,
            )
            logger.debug(f"At {batch_num} we have process {len(newfound_triplets)} triplets")
            num_ogEntities_processed += batch_size
            logger.debug(f"At {batch_num} we have process {num_ogEntities_processed}")

            expanded_triplets.update(newfound_triplets)
            qualifier_dictionary.update(_qualifier_dictionary)

            # Use tails in newfound_triplets to expand the entity set
            # DEBUG: Commenting this for a sec because I want to let it finish to see how it behaves
            # for _, _, t in newfound_triplets:
            #     new_neighbors.add(t)

            #TOREM: Mostly to test and not explode during testing
            batch_num += 1
            if batch_num >= 1:
                _exit = True
                break

        if _exit: 
            break
        
        # If no new neighbors were found, we can't expand further
        if not new_neighbors:
            print("No new neighbors found, stopping expansion")
            break
            
        # Prepare for next hop if needed
        entities_to_process = list(new_neighbors - processed_entities)
        print(f"Hop {hop+1} complete. Found {len(new_neighbors)} new neighbors.")
        print(f"Total entities now: {len(expanded_triplets)}")
    
    print(f"Entity expansion complete. Final count: {len(expanded_triplets)}")
    
    return expanded_triplets, qualifier_dictionary


def generate_triplets_from_entities(
    entity_file: str,
    output_file: str,
    max_workers: int = 20,
    max_retries: int = 3,
    timeout: int = 5,
    target_triplet_count: int = 200000,
) -> None:
    """
    Generate triplets by querying WikiData for relationships between entities.
    
    Args:
        entity_file: Path to the file containing entities
        output_file: Path to save the generated triplets
        max_workers: Maximum number of concurrent workers for WikiData API queries
        max_retries: Maximum number of retries for failed WikiData queries
        timeout: Timeout in seconds for WikiData queries
        target_triplet_count: Target number of triplets to generate
    """
    print(f"Generating triplets from entities in {entity_file}")
    print(f"Target triplet count: {target_triplet_count}")
    
    # Use the existing process_entity_triplets function from utils.wikidata_v2
    process_entity_triplets(
        entity_file,
        output_file,
        max_workers=max_workers,
        max_retries=max_retries,
        timeout=timeout,
    )
    
    # Count the generated triplets
    triplet_count = 0
    with open(output_file, 'r') as f:
        for _ in f:
            triplet_count += 1
    
    print(f"Generated {triplet_count} triplets")
    
    if triplet_count < target_triplet_count:
        print(f"Warning: Generated triplet count ({triplet_count}) is less than target ({target_triplet_count})")
        print("Consider expanding the entity set or increasing the expansion hops")


def process_triplets(
    triplet_file: str,
    output_file: str,
    entity_file: Optional[str] = None,
    relation_file: Optional[str] = None,
    handle_inverses: bool = True,
) -> None:
    """
    Process raw triplets: deduplicate, handle inverse relations, and filter.
    
    Args:
        triplet_file: Path to the file containing raw triplets
        output_file: Path to save the processed triplets
        entity_file: Optional path to save the filtered entities
        relation_file: Optional path to save the filtered relations
        handle_inverses: Whether to identify and handle inverse relations
    """
    print(f"Processing triplets from {triplet_file}")
    
    # Load raw triplets
    raw_triplets = []
    with open(triplet_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                raw_triplets.append(tuple(parts))
    
    print(f"Loaded {len(raw_triplets)} raw triplets")
    
    # Count entity and relation frequencies
    entity_freq = defaultdict(int)
    relation_freq = defaultdict(int)
    
    for head, rel, tail in raw_triplets:
        entity_freq[head] += 1
        entity_freq[tail] += 1
        relation_freq[rel] += 1
    
    # Remove duplicates
    unique_triplets = list(set(raw_triplets))
    
    print(f"Removed {len(raw_triplets) - len(unique_triplets)} duplicate triplets")
    
    # Handle inverse relations if requested
    if handle_inverses:
        # Identify potential inverse relations
        relation_pairs = defaultdict(lambda: defaultdict(int))
        
        # First pass: collect relation pair patterns
        for head, rel, tail in unique_triplets:
            # For each triplet, record the entity pair and relation
            relation_pairs[(head, tail)][rel] += 1
            
        # Identify likely inverse relations
        inverse_relations = {}
        potential_inverse_pairs = []
        
        # Look for relations that frequently occur with reversed entity pairs
        for (head1, tail1), rels1 in relation_pairs.items():
            for rel1 in rels1:
                # Check the reverse entity pair
                if (tail1, head1) in relation_pairs:
                    # For each relation in the reverse pair
                    for rel2 in relation_pairs[(tail1, head1)]:
                        if rel1 != rel2:  # Different relations
                            # Record the potential inverse pair
                            potential_inverse_pairs.append((rel1, rel2, rels1[rel1]))
        
        # Sort by frequency and select top pairs
        potential_inverse_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Map relations to their most frequent inverse
        for rel1, rel2, freq in potential_inverse_pairs:
            if rel1 not in inverse_relations and rel2 not in inverse_relations.values():
                inverse_relations[rel1] = rel2
        
        print(f"Identified {len(inverse_relations)} potential inverse relation pairs")
        
        # Process triplets to handle inverses
        processed_triplets = []
        seen_triplets = set()
        seen_inverse_patterns = set()
        
        for head, rel, tail in unique_triplets:
            triplet_key = (head, rel, tail)
            
            # Skip if we've seen this exact triplet
            if triplet_key in seen_triplets:
                continue
                
            # Check if this is an inverse of one we've seen
            if rel in inverse_relations and (tail, inverse_relations[rel], head) in seen_triplets:
                continue
                
            # Check if we've seen the inverse relation pattern
            inverse_pattern = (tail, head)
            if rel in inverse_relations and inverse_pattern in seen_inverse_patterns:
                continue
                
            # Add to processed triplets and tracking sets
            processed_triplets.append(triplet_key)
            seen_triplets.add(triplet_key)
            
            if rel in inverse_relations:
                seen_inverse_patterns.add((head, tail))
    else:
        # Simply use the unique triplets
        processed_triplets = unique_triplets
    
    print(f"Final processed triplet count: {len(processed_triplets)}")
    
    # Save processed triplets
    with open(output_file, 'w') as f:
        for head, rel, tail in processed_triplets:
            f.write(f"{head} {rel} {tail}\n")
    
    # Extract and save final entities and relations if requested
    if entity_file or relation_file:
        final_entities = set()
        final_relations = set()
        
        for head, rel, tail in processed_triplets:
            final_entities.add(head)
            final_entities.add(tail)
            final_relations.add(rel)
        
        if entity_file:
            with open(entity_file, 'w') as f:
                for entity in sorted(final_entities):
                    f.write(f"{entity}\n")
                    
        if relation_file:
            with open(relation_file, 'w') as f:
                for relation in sorted(final_relations):
                    f.write(f"{relation}\n")
        
        print(f"Final entity count: {len(final_entities)}")
        print(f"Final relation count: {len(final_relations)}")


def create_dataset_splits(
    processed_triplet_file: str,
    train_file: str,
    test_file: str,
    valid_file: str,
    train_ratio: float = 0.8,
    test_valid_ratio: float = 0.5,
    seed: int = 42,
) -> None:
    """
    Split processed triplets into train/test/validation sets for model training.
    
    Args:
        processed_triplet_file: Path to the processed triplets file
        train_file: Path to save the training triplets
        test_file: Path to save the test triplets
        valid_file: Path to save the validation triplets
        train_ratio: Proportion of triplets to use for training
        test_valid_ratio: Proportion of non-training triplets to use for testing vs validation
        seed: Random seed for reproducibility
    """
    print(f"Creating dataset splits from {processed_triplet_file}")
    print(f"Train ratio: {train_ratio}, Test-valid ratio: {test_valid_ratio}")
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Load processed triplets
    triplets = []
    with open(processed_triplet_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                triplets.append(tuple(parts))
    
    # Count total triplets
    total_triplets = len(triplets)
    
    print(f"Loaded {total_triplets} triplets")
    
    # Track entity and relation statistics
    entity_to_triplets = defaultdict(list)
    relation_to_triplets = defaultdict(list)
    
    for i, (head, rel, tail) in enumerate(triplets):
        entity_to_triplets[head].append(i)
        entity_to_triplets[tail].append(i)
        relation_to_triplets[rel].append(i)
    
    # Determine split sizes
    train_size = int(total_triplets * train_ratio)
    test_valid_size = total_triplets - train_size
    test_size = int(test_valid_size * test_valid_ratio)
    valid_size = test_valid_size - test_size
    
    # Initialize sets for tracking
    train_indices = set()
    test_indices = set()
    valid_indices = set()
    
    # First, ensure each relation has examples in each split
    min_examples_per_split = 3
    
    for rel, triplet_indices in relation_to_triplets.items():
        if len(triplet_indices) < min_examples_per_split * 3:
            # Too few examples, just put all in training
            train_indices.update(triplet_indices)
        else:
            # Shuffle and allocate
            shuffled = triplet_indices.copy()
            random.shuffle(shuffled)
            
            # Add some to each split
            valid_indices.update(shuffled[:min_examples_per_split])
            test_indices.update(shuffled[min_examples_per_split:min_examples_per_split*2])
            train_indices.update(shuffled[min_examples_per_split*2:])
    
    # Fill remaining slots
    remaining_indices = set(range(total_triplets)) - train_indices - test_indices - valid_indices
    remaining_list = list(remaining_indices)
    random.shuffle(remaining_list)
    
    # Calculate how many more we need for each split
    train_needed = train_size - len(train_indices)
    test_needed = test_size - len(test_indices)
    valid_needed = valid_size - len(valid_indices)
    
    # Add remaining to appropriate splits
    if train_needed > 0:
        train_indices.update(remaining_list[:train_needed])
        remaining_list = remaining_list[train_needed:]
    
    if test_needed > 0 and remaining_list:
        test_indices.update(remaining_list[:test_needed])
        remaining_list = remaining_list[test_needed:]
    
    if valid_needed > 0 and remaining_list:
        valid_indices.update(remaining_list[:valid_needed])
    
    # Prepare final triplet lists
    train_triplets = [triplets[i] for i in train_indices]
    test_triplets = [triplets[i] for i in test_indices]
    valid_triplets = [triplets[i] for i in valid_indices]
    
    # Save splits to files
    with open(train_file, 'w') as f:
        for head, rel, tail in train_triplets:
            f.write(f"{head} {rel} {tail}\n")
    
    with open(test_file, 'w') as f:
        for head, rel, tail in test_triplets:
            f.write(f"{head} {rel} {tail}\n")
    
    with open(valid_file, 'w') as f:
        for head, rel, tail in valid_triplets:
            f.write(f"{head} {rel} {tail}\n")

    print(f"Dataset split complete:")
    print(f"- Training: {len(train_triplets)} triplets ({len(train_triplets)/total_triplets*100:.1f}%)")
    print(f"- Testing: {len(test_triplets)} triplets ({len(test_triplets)/total_triplets*100:.1f}%)")
    print(f"- Validation: {len(valid_triplets)} triplets ({len(valid_triplets)/total_triplets*100:.1f}%)")


def main():
    """Main function to execute the MQuAKE triplet expansion pipeline."""
    args = parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    paths_to_create = [
        args.og_entity_output,
        args.og_relation_output,
        args.cf_entity_output,
        args.cf_relation_output,
        args.expanded_triplet_output,
        args.raw_triplet_output,
        args.processed_triplet_output,
        args.train_output,
        args.test_output,
        args.valid_output,
    ]

    # Create output directories if they don't exist
    for path in paths_to_create:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    # Execute pipeline based on mode
    if args.mode in ['extract_entities', 'full_pipeline']:
        # Extract entities from MQuAKE data
        og_entities, og_relations, cf_entities, cf_relations = extract_mquake_entities(
            args.mquake_path, 
        )
        paths_to_save = [
            args.og_entity_output,
            args.og_relation_output,
            args.cf_entity_output,
            args.cf_relation_output,
        ]
        entities_to_save = [og_entities, og_relations, cf_entities, cf_relations]

        for path, ents in zip(paths_to_save, entities_to_save):
            with open(path, 'w') as f:
                for entity in sorted(ents):
                    f.write(f"{entity}\n")
            print(f"Saved {len(ents)} RDFs to {path}")
        print("Note: At the moment the counter factual entities and relations have not use. It is in case they are needed in the future.")

    if args.mode in ['expand_entities', 'full_pipeline']:
        # Load MQuAKE entities if not already extracted in this run
        mquake_entities = set()
        with open(args.og_entity_output, 'r') as f:
            mquake_entities = {line.strip() for line in f}

        # Expand entity set
        logger.info(f"Expanding the original entity set from {len(mquake_entities)} initial entities...")
        expanded_triplets, qualifier_dict = expand_triplet_set(
            mquake_entities,
            target_size=args.target_entity_count,
            expansion_hops=args.expansion_hops,
            batch_size=args.entity_batch_size,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
            timeout=args.timeout,
        )

        # Save expanded entities to file using pandas
        columns = ["head", "rel", "tail", "qualifiers"]
        data = []
        for new_trip in expanded_triplets:
            if new_trip in qualifier_dict:
                data.append((*new_trip, str(qualifier_dict[new_trip])))
            else:
                data.append((*new_trip, ""))
        
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(args.expanded_triplet_output, index=False)

        print(f"Saved {len(expanded_triplets)} expanded entities to {args.expanded_triplet_output}")

    exit()

    # TOREM: Potentially redundant
    # if args.mode in ['generate_triplets', 'full_pipeline']:
    #     # Generate triplets by querying WikiData
    #     generate_triplets_from_entities(
    #         args.expanded_entity_output if args.mode == 'full_pipeline' else args.og_entity_output,
    #         args.raw_triplet_output,
    #         max_workers=args.max_workers,
    #         max_retries=args.max_retries,
    #         timeout=args.timeout,
    #         target_triplet_count=args.target_triplet_count,
    #     )

    if args.mode in ['process_triplets', 'full_pipeline']:
        # Process triplets: deduplicate, handle inverses, etc.
        process_triplets(
            args.raw_triplet_output,
            args.processed_triplet_output,
            entity_file=args.og_entity_output,
            relation_file=args.og_relation_output,
            handle_inverses=True,
        )

    if args.mode in ['create_dataset', 'full_pipeline']:
        # Create dataset splits
        create_dataset_splits(
            args.processed_triplet_output,
            args.train_output,
            args.test_output,
            args.valid_output,
            train_ratio=args.train_ratio,
            test_valid_ratio=args.test_valid_ratio,
            seed=args.seed,
        )

    if args.mode == 'full_pipeline':
        print("\nFull pipeline completed successfully!")
        print(f"- Initial MQuAKE entities: {args.og_entity_output}")
        print(f"- Expanded entity set: {args.expanded_entity_output}")
        print(f"- Relations: {args.og_relation_output}")
        print(f"- Raw triplets: {args.raw_triplet_output}")
        print(f"- Processed triplets: {args.processed_triplet_output}")
        print(f"- Training set: {args.train_output}")
        print(f"- Test set: {args.test_output}")
        print(f"- Validation set: {args.valid_output}")


if __name__ == "__main__":
    logger = create_logger("mquake_triplet_process")
    main() 
