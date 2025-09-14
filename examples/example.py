import pandas as pd
from matchmaker.data import Interactions

print("Importing data...")
df = pd.read_csv("examples/data/swipes.csv")

print("Building swipes...")
swipes = [
    (row.decidermemberid, row.othermemberid)
    for row in df.itertuples(index=False)
    if row.like == 1
]

i = Interactions(swipes)

R, indexer = i.build_matrix()
print(R)
print(indexer)