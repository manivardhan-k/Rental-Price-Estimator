import random
import os

# https://www.kaggle.com/datasets/austinreese/usa-housing-listings

filename = "csvs/housing.csv"

def sample_csv(path, k, has_header=True):
    sample = []
    with open(path, 'r', encoding='utf-8') as f:
        header = next(f) if has_header else None
        for i, line in enumerate(f, start=1):
            if i <= k:
                sample.append(line)
            else:
                j = random.randint(1, i)
                if j <= k:
                    sample[j-1] = line
    if header:
        sample.insert(0, header)
    return sample

sample_lines = sample_csv(filename, 50000)
# with open("csvs/housing_sample50k.csv", "w", encoding="utf-8") as f:
#     f.writelines(sample_lines)