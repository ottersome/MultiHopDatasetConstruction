"""
This script checks the inverse relations in the MQuAKE dataset.
"""
import pandas as pd

# Overall paths
relations_path = "./data/mquake/expanded_relations.txt"
rel_hierarchy_path = "./data/relationships_hierarchy.txt"

with open(relations_path, 'r') as f:
    # Each line is a relation
    relations = [line.strip()  for line in f.readlines()]
relations_df = pd.DataFrame(relations, columns=["relation"])
print(f"relations head: \n{relations_df.head()}")

# Load Relationship Hierarchy Mapping
rel_hierarchy_df = pd.read_csv(
    rel_hierarchy_path,
    sep="\t",
    header=None,
    names=["head", "relation", "tail"],
)
rel_inverses = rel_hierarchy_df[rel_hierarchy_df["relation"] == "P1696"]
print(f"rel_inverses (count: {len(rel_inverses)})  head():\n"
      "-----------------------\n"
      f"{rel_inverses.head()}\n"
      "-----------------------\n"
      )

# Count how many of the relations have hierarchies
merged_df = pd.merge(relations_df, rel_inverses, left_on="relation", right_on="head", how="inner").drop("relation_x", axis=1)
print(f"merged_df (left) (count: {len(merged_df)}) : \n{merged_df.head()}")

# Other side now
merged_df = pd.merge(relations_df, rel_inverses, left_on="relation", right_on="tail", how="inner").drop("relation_x", axis=1)
print(f"merged_df (right) (count: {len(merged_df)}) : \n{merged_df.head()}")
