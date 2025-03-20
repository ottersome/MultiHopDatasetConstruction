# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 23:22:45 2024

@author: Eduin Hernandez

Summary: Filters Jeopardy questions to retain only those compatible with the designated
    dataset by verifying the presence of entities in both the questions and the target dataset. 
    The script reads the processed Jeopardy questions, valid triplets, and node data,
    and outputs the filtered questions containing entities that align with the target dataset.
"""
import argparse

import pandas as pd
from tqdm import tqdm

from utils.basic import load_pandas, load_triplets, extract_literals


def parse_args() -> argparse.Namespace:
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="")
    
    # Input
    parser.add_argument('--jeopardy-path', type=str, default='./data/jeopardy_processed_bert.csv',
                        help='Path to the CSV file containing processed Jeopardy questions with associated QIDs.')
    parser.add_argument('--triplets-path', type=str, default='./data/triplets_fb_wiki_v2.txt',
                        help='Path to the text file containing valid triplets of entities used for filtering.')
    parser.add_argument('--nodes-data-path', type=str, default='./data/node_data_fb_wiki.csv',
                        help='Path to the CSV file containing node data, including the respective entity name for each QID.')
    
    # Output
    parser.add_argument('--jeopardy-output-path', type=str, default='./data/jeopardy_fb_wiki_alt.csv',
                        help='Path to save the filtered Jeopardy questions containing entities that match the target dataset.')
    
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()
    
    #--------------------------------------------------------------------------
    
    jeopardy_df = load_pandas(args.jeopardy_path)
    jeopardy_df_num_rows = jeopardy_df.shape[0]
    triplets = load_triplets(args.triplets_path)
    node_data = load_pandas(args.nodes_data_path)
    
    nodes = set(triplets['head'].tolist()) | set(triplets['tail'].tolist())

    questions_qids = jeopardy_df['Question-Qid']
    answers_qids = jeopardy_df['Answer-Qid']

    assert isinstance(questions_qids, pd.Series), "Question-Qid column is not a pandas Series"
    assert isinstance(answers_qids, pd.Series), "Answer-Qid column is not a pandas Series"
    
    jeopardy_questions = extract_literals(questions_qids)
    jeopardy_answers = extract_literals(answers_qids)

    assert isinstance(jeopardy_questions, pd.Series), "Later on jeopardy_questions is expected to be a series, and thus needs to be checked against being a list"
    assert isinstance(jeopardy_answers, pd.Series), "Later on jeopardy_answers is expected to be a series, and thus needs to be checked against being a list"
    
    # Create a new DataFrame to store filtered rows
    filtered_rows = []
    
    for j, (i0, row) in tqdm(enumerate(jeopardy_df.iterrows()), total=jeopardy_df_num_rows, desc="Iterating through rows of unfiltered jeopardy dataset"):

        q_rdfs = set(jeopardy_questions.iloc[i0])
        a_rdfs = set(jeopardy_answers.iloc[i0])

        if not(a_rdfs.issubset(nodes)):
            continue
        
        intersection = q_rdfs & nodes
        # if not(intersection) and len(intersection) < 2:
        if len(intersection) < 2:
            continue
        
        # Update q_rdf with the intersection list
        row_copy = row.copy()
        row_copy['Question-Qid'] = list(intersection)
        
        row_copy['Question-Entities'] = str([e0 for e0 in node_data[node_data['RDF'].isin(intersection)]['Title']])
        
        filtered_rows.append(row_copy)
        
    # Create a new DataFrame with filtered rows
    filtered_df = pd.DataFrame(filtered_rows)

    # Save or use the filtered DataFrame as needed
    filtered_df.to_csv(args.jeopardy_output_path, index=False)
