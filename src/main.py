from DecisionTrees import DecisionTree
import pandas as pd
import numpy as np

if __name__ == "__main__":
    train = pd.read_csv("./data/train_preprocessed.csv")
    test = pd.read_csv("./data/test_preprocessed.csv")

    model = DecisionTree()
    model.fit(data=train, target="Survived")

    pred: np.ndarray =  model.predict(data=test)