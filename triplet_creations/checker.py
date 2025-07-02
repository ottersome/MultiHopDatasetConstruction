"""
This file ensure that all entities in the origina mquake dataset are present in the pruned and expanded one.
"""

import pandas as pd

from utils.basic import load_triplets
from utils.mquake import extract_mquake_entities

og_entities, og_relations, _, _  = extract_mquake_entities("./data/MQuAKE-CF.json")

mquake_entities_expNPrun_set = set(pd.read_csv("./data/mquake/expandedNpruned_entities.txt", sep="\t").iloc[:, 0].values.tolist())
mquake_relations_expNPrun_set = set(pd.read_csv("./data/mquake/expandedNPruned_relations.txt", sep="\t").iloc[:, 0].values.tolist())

print(f"Length of og_entities : {len(og_entities)} and length of expNprun: {len(mquake_entities_expNPrun_set)}")
print(f"Length of og_relations : {len(og_relations)} and length of expNprun: {len(mquake_relations_expNPrun_set)}")

# Check which og_entities are not in its pruned version:
if not set(og_entities).issubset(mquake_entities_expNPrun_set):
    print("Not all og_entities are in the pruned version")
    # The difference is:
    og_entities_notpruned = og_entities - mquake_entities_expNPrun_set
    print(f"Amount of og_entites not in final pruned version: {len(og_entities_notpruned)}")
    with open("./debug/og_entities_notpruned.txt", "w") as f:
        for entity in og_entities_notpruned:
            f.write(f"{entity}\n")
if not set(og_relations).issubset(mquake_relations_expNPrun_set):
    print("Not all og_relations are in the pruned version")
    # The difference is:
    og_relations_notpruned = og_relations - mquake_relations_expNPrun_set
    print(f"Amount of og_relations not in final pruned version: {len(og_relations_notpruned)}")
    with open("./debug/og_relations_notpruned.txt", "w") as f:
        for relation in og_relations_notpruned:
            f.write(f"{relation}\n")

