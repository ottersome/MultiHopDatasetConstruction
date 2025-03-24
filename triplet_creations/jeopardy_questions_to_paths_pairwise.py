# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:24:18 2024

@author: Eduin Hernandez

Summary:
This script processes Jeopardy questions to extract meaningful paths between entities
 using the Freebase-Wikidata hybrid graph stored in Neo4j. It uses embeddings, ANN for
 nearest neighbor search, and threading to efficiently extract relationships and paths
 between nodes in the graph.

"""

import argparse
from tqdm import tqdm

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from typing import List, Optional, Tuple, Any

from utils.configs import global_configs
from utils.basic import load_pandas, overload_parse_defaults_with_yaml
from utils.basic import extract_literals, random_dataframes, str2bool
from utils.verify_triplets import sort_path_by_node_match, filter_tuples_by_node, visualize_path
from utils.openai_api import OpenAIHandler
from utils.fb_wiki_graph import FbWikiGraph, Path
from utils.fb_wiki_ann import FbWikiANN

def parse_args() -> argparse.Namespace:
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process Jeopardy questions to extract entity paths using Neo4j graph database.")

    # Saved configs for posterity
    # Example config file is: ./configs/jeopardy_questions_to_paths/jeopardy_fbwikiv4.yaml
    parser.add_argument('--saved_config', type=str, help='Path to a preconfigured save of arguments in a YAML config file')
    
    # Input Data from the CherryPicked
    parser.add_argument('--jeopardy-data-path', type=str, default='./data/jeopardy_cherrypicked.csv',
                        help='Path to the CSV file containing jeopardy questions')
    parser.add_argument('--node-data-path', type=str, default='./data/node_data_cherrypicked.csv',
                        help='Path to the CSV file containing entity data.')
    parser.add_argument('--relation-data-path', type=str, default='./data/relation_data_subgraph.csv',
                        help='Path to the CSV file containing relationship data')
    parser.add_argument('--relation-embeddings-path', type=str, default='./data/relationship_embeddings_gpt_subgraph_full.csv',
                        help='Path to the CSV file containing the relationships embeddings.')
    parser.add_argument('--database', type=str, default='neo4j',
                        help='Name of the Neo4j database to use.')

    # General Parameters
    parser.add_argument('--max-relevant-relations', type=int, default=None, #25 is the ideal value
                        help='How many relevant relations to extract through nearest neighbors.')
    parser.add_argument('--max-questions', type=int, default=None,
                        help='Max number of jeopardy questions to use. For all, use None.')

    # Neo4j
    parser.add_argument('--config-path', type=str, default='./configs/config_neo4j.ini',
                        help='Path to the configuration file for Neo4j connection.')
    # parser.add_argument('--database', type=str, default='subgraph',
    #                     help='Name of the Neo4j database to use.')
    parser.add_argument('--min-hops', type=int, default=1,
                        help='Minimum number of hops to consider in the path.')
    parser.add_argument('--max-hops', type=int, default=4,
                        help='Maximum number of hops to consider in the path.')
    parser.add_argument('--num-workers', type=int, default=10,
                        help='Number of workers to use for path extractions.')
    parser.add_argument('--path_return_limit', type=int, default=1,
                        help='Maximum number of paths to return for each node pair.')
    
    # ANN Parameters
    parser.add_argument('--ann-exact-computation', type=str2bool, default='True',
                        help='Flag to use exact computation for the search or an approximation.')
    parser.add_argument('--ann-nlist', type=int, default=32,
                        help='Specifies how many partitions (Voronoi cells) we’d like our ANN index to have. Used only on the approximate search.')

    # LLM models
    parser.add_argument('--embedding-model', type=str, default='text-embedding-3-small',
                        help='Model name to be used for embedding calculations (e.g., "text-embedding-3-small"). Must be a key in pricing_embeddings.')
    parser.add_argument('--encoding-model', type=str, default='cl100k_base',
                        help='Encoding name used by the model to tokenize text for embeddings.')

    # Output
    parser.add_argument('--save-results', type=str2bool, default='True', 
                        help='Flag for whether to save the results or not.')
    parser.add_argument('--result-output-path', type=str, default='./data/jeopardy_cherrypicked_path.csv',
                        help='Path to the CSV file containing jeopardy questions')

    # Statistics
    parser.add_argument('--verbose-debug', type=str2bool, default='True',
                        help='Flag to enable detailed logging for debugging purposes.')
    parser.add_argument('--verbose', type=str2bool, default='True',
                        help='Flag to enable output of summary statistics at the end of processing.')

    args = parser.parse_args()

    if args.saved_config:
        print(
            f"\033[1;32mConfiguration loaded from {args.saved_config}."
            "\nThe config will override any CLI arguments.\033[0m"
        )
        args = overload_parse_defaults_with_yaml(args.saved_config, args)
        # Show me dump for sanity check
    else:
        print("\033[1;32mUsing default configuration\033[0m")

    return args


def extract_path(
    g: FbWikiGraph,
    x: str,
    y: str,
    min_hops: Optional[int],
    max_hops: Optional[int],
    limit: Optional[int],
    rels: Optional[list[str]],
    non_inform: List[str],
) -> list[Path]:
    """
    Extracts paths between two nodes in the graph.

    Args:
        g (FbWikiGraph): The graph object.
        x (str): The RDF identifier of the start node.
        y (str): The RDF identifier of the end node.
        args (argparse.Namespace): Parsed arguments.
        rels (List[str], optional): List of relationship types to consider. Defaults to None.
        non_inform (List[str], optional): List of non-informative relationships to filter out. Defaults to [].

    Returns:
        List[Tuple[List[Any], List[Any]]]: A list of paths between the nodes.
    """
    paths = g.find_path(x, y, 
                    min_hops=min_hops,
                    max_hops=max_hops,
                    relationship_types=rels,
                    noninformative_types=non_inform,
                    limit=limit,
                    rdf_only=True,
                    can_cycle=False
                    )
    return paths


def find_paths_pairwise(
    g: FbWikiGraph,
    nodes: list[str],
    allowed_relTypes: list[str],
    non_inform_relTypes: list[str],
    timeout: Optional[int],
    num_paths_return_limit: Optional[int],
    max_hops: Optional[int],
    min_hops: Optional[int],
    num_workers: int,
) -> list[Path]:
    """
    Find paths connecting all nodes in a set where each node connects to at least one other.
    
    Args:
        g: Neo4j graph object
        nodes: List of node IDs including question nodes and answer node
        args: Arguments for path extraction
        allowed_rel_types: Allowed relationship types
        non_inform: Non-informative relationship types to filter out
    Returns:
        List of path tuples
    """
    
    all_paths = []
    connected_nodes = set()  # Nodes that already have at least one connection
    unconnected_nodes = set(nodes)  # Nodes still needing a connection
    
    while unconnected_nodes:
        # Take the first unconnected node
        current_node = next(iter(unconnected_nodes))
        
        # Define potential target nodes (all nodes except self)
        potential_targets = [n for n in nodes if n != current_node]
        
        # Find a path to ANY other node
        found_path_inner = False
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Create futures for path searches
            future_to_target = {}
            for target in potential_targets:
                future = executor.submit(
                    extract_path,
                    g,
                    current_node,
                    target,
                    min_hops,
                    max_hops,
                    num_paths_return_limit,
                    allowed_relTypes,
                    non_inform_relTypes,
                )
                future_to_target[future] = target
            
            # Process results as they complete with timeout
            for future in as_completed(future_to_target):
                target = future_to_target[future]
                try:
                    paths: list[Path] = future.result(timeout=timeout)
                    if paths:
                        all_paths.extend(paths[0]) # Just grab the first path. We dont care otherwise
                        connected_nodes.add(current_node)
                        connected_nodes.add(target)
                        found_path_inner = True
                        break
                except TimeoutError:
                    print(f"Path search from {current_node} to {target} timed out after {timeout}s")
                    continue
        
        # Remove this node from unconnected, whether a path was found or not
        unconnected_nodes.remove(current_node)
        
        # If no path found, we still mark it as processed but print a warning
        if not found_path_inner:
            # Unteneable question
            return []
    
    return all_paths

def main():
    args = parse_args()
    
    configs = global_configs(args.config_path)
    neo4j_parameters = configs['Neo4j']
    
    #--------------------------------------------------------------------------
    'Instantiating Models and Loading Data'
    
    # Question, Embedding, and ANN models
    if args.max_relevant_relations is not None:
        embedding_gpt = OpenAIHandler(model=args.embedding_model, encoding=args.encoding_model)
        ann = FbWikiANN(
                data_path = args.relation_data_path,
                embedding_path = args.relation_embeddings_path, 
                exact_computation = args.ann_exact_computation,
                nlist=args.ann_nlist
                )
    else:
        ann = None
    
    g = FbWikiGraph(neo4j_parameters['uri'], neo4j_parameters['user'],
                    neo4j_parameters['password'], database = args.database)
    
    # Data
    jeopardy_df = load_pandas(args.jeopardy_data_path)
    node_data_df = load_pandas(args.node_data_path)
    relation_df = load_pandas(args.relation_data_path)
    
    node_data_df.set_index('RDF', inplace=True)
    relation_df.set_index('Property', inplace=True)
    
    if 'Unnamed: 0' in jeopardy_df.columns: jeopardy_df.drop(columns=['Unnamed: 0'], inplace=True)
    if 'Unnamed: 0' in node_data_df.columns: node_data_df.drop(columns=['Unnamed: 0'], inplace=True)
    if 'Unnamed: 0' in relation_df.columns: relation_df.drop(columns=['Unnamed: 0'], inplace=True)
    
    if args.max_questions and args.max_questions < len(jeopardy_df):
        jeopardy_df = random_dataframes(jeopardy_df, args.max_questions)
    
    noninformative_pids = ['P31', 'P279', 'P518', 'P1343']
    
    final_results = {}

    jeopardy_df['Path'] = [[] for _ in range(len(jeopardy_df))]
    jeopardy_df['has_path'] = False
    for i0, row in tqdm(jeopardy_df.iterrows(), total=len(jeopardy_df), desc="Processing Jeopardy Questions"):
        question = 'Category: ' + row['Category'] + ' Question: ' + row['Question']
        answers = list(extract_literals(row['Answer_RDF'])[0])
        q_ids = list(set(extract_literals(row['Question_RDF'])[0]))
        
        log_output = []
        
        if args.verbose_debug:
            log_output.append(f'\n==================\nSample {i0+1}')
            log_output.append(f"{question}")
            log_output.append(f"Answer: {node_data_df.loc[answers[0]]['Title']}")
            log_output.append(f"Entities: {node_data_df.loc[q_ids]['Title'].tolist()}")
        
        if ann is not None:
            embeddings = np.array(embedding_gpt.get_embedding(question))[None,:]
            _, indices = ann.search(embeddings, args.max_relevant_relations)
            allowed_relTypes = list(set(ann.index2data(indices, 'Property', max_indices=args.max_relevant_relations)[0]))
        else:
            allowed_relTypes = None

        paths = find_paths_pairwise(
            g,
            q_ids,
            answers[0],
            allowed_relTypes,
            noninformative_pids,
            timeout=args.path_search_timeout,
        )
        # question nodes and answer node
            
        sorted_tuples, _, _ = sort_path_by_node_match(paths, q_ids)
        
        visual_path = 'NO PATH FOUND'
        if paths:
            final_results[i0] = sorted_tuples[0]
            jeopardy_df.at[i0, 'Path'] = list(sorted_tuples[0])
            jeopardy_df.at[i0, 'has_path'] = True
            visual_path = visualize_path(sorted_tuples[0], node_data_df, relation_df)
        else:
            final_results[i0] = []
         
        if args.verbose_debug:
            log_output.append(f"{visual_path}")
            # log_output.append(f"Rels: {relation_df.loc[p_ids]['Title'].tolist()}")
            # log_output.append(f"\nDistances: {d}")
            
            # Write all log outputs at the end of the iteration
            for line in log_output:
                tqdm.write(line)

if __name__ == '__main__':
    main()


if args.save_results:
    jeopardy_df.to_csv(args.result_output_path, index=False)

if args.verbose:
    jeparday_path_df = jeopardy_df[jeopardy_df['has_path'] == True]
    for j0, row in jeparday_path_df.iterrows():
        question = 'Category: ' + row['Category'] + ' Question: ' + row['Question']
        answers = list(extract_literals(row['Answer_RDF'])[0])
        q_ids = list(set(extract_literals(row['Question_RDF'])[0]))
        entities = node_data_df.loc[q_ids]['Title'].tolist()
        
        path = row['Path']
        visual_path = visualize_path(path, node_data_df, relation_df)
        
        print(f'\n==================\nSample {j0+1}')
        print(f"{question}")
        print(f"Answer: {node_data_df.loc[answers[0]]['Title']}")
        print(f"Entities: {entities}")
        print(f"{visual_path}")
    
    print('\n==================')
    print(f"Jeopardy Questions with Paths: {sum(jeopardy_df['has_path'])}/{len(jeopardy_df)}")
