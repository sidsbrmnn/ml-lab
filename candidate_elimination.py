import numpy as np
import pandas as pd

data = pd.DataFrame(pd.read_csv("data/weather.csv"))
concepts = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])

specific_h = concepts[0].copy()
general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(concepts))]

for i, h in enumerate(concepts):
    if target[i] == "Yes":
        for x in range(len(specific_h)):
            if h[x] != specific_h[x]:
                specific_h[x] = "?"
                general_h[x][x] = "?"
    else:
        for x in range(len(specific_h)):
            if h[x] != specific_h[x]:
                general_h[x][x] = specific_h[x]
            else:
                general_h[x][x] = "?"

indices = [
    i for i, h in enumerate(general_h) if h == ["?" for _ in range(len(specific_h))]
]
for i in indices:
    general_h.pop(i)

print("Final S:", specific_h)
print("Final G:", general_h)
