import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


def main() -> None:
    dataset = sklearn.datasets.load_diabetes()
    
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

    df['target'] = dataset.target

    print(df.columns)
    print(df.head())


if __name__ == "__main__":
    main()
