import numpy as np
import pandas as pd

data = pd.DataFrame(pd.read_csv("data/weather.csv"))
concepts = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])

specific_h = concepts[0].copy()

for i, h in enumerate(concepts):
    if target[i] == "Yes":
        for x in range(len(specific_h)):
            if h[x] != specific_h[x]:
                specific_h[x] = "?"

print("Specific H:", specific_h)

for _, h in concepts:
    d = [False] * len(specific_h)
    indices = np.where(specific_h == "?")[0]

    for i, val in enumerate(specific_h):
        d[i] = True if i in indices else val == h[i]

    print(h, d)
