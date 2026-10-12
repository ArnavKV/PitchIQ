
import pandas as pd

players = {
    "Player": ["Player A", "Player B", "Player C"],
    "Goals": [20, 15, 12],
    "Assists": [8, 11, 6]
}

df = pd.DataFrame(players)

print(df)
