import json
import pandas as pd
import argparse

def get_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mquake_file", default="./data/MQuAKE-CF.json")
    ap.add_argument("--triplet_file", default="./data/mquake/expanded_triplets_wo_qualifiers.csv")

    return ap.parse_args()

def main():
    args = get_args()

    # Load the json file
    with open(args.mquake_file, "r") as f:
        mquake_data = json.load(f)

    # Load the triplets
    triplets = pd.read_csv(args.triplet_file)

    # Get Counts for entities and relations
    entity_counts = triplets["head"].value_counts()
    relation_counts = triplets["relation"].value_counts()

    # Get counts for questions
    question_counts = triplets["head"].value_counts()

    # Create a set of entities
    entities = set(triplets["head"])
    entities.update(triplets["tail"])
    entities.update(triplets["relation"])

    # Create a set of relations
    relations = set(triplets["relation"])

    # Create a set of questions
    questions = set()
    for question in mquake_data["questions"]:
        questions.add(question["question"])


if __name__ == "__main__":
    main()
