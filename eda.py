import os
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("emissions.csv")
    result = defaultdict(list)
    for _, row in df.iterrows():
        project_name = row["project_name"]
        emissions = row["emissions"]
        if pd.isna(emissions):
            continue
        result[project_name].append(emissions)
    for key, value in result.items():
        sns.histplot(value, bins=100)
        plt.savefig(os.path.join("figures", f"{key}.png"))
        print(key, np.mean(value), np.std(value), len(value))


if __name__ == "__main__":
    main()