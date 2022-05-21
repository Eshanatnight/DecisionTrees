import numpy as np
import pandas as pd

class DecisionTree:
    """
    A simple decision tree class
    """

    def __init__(self, max_depth=6, depth=1):
        self.left = None
        self.right = None
        self.depth = depth
        self.max_depth = max_depth 

    def fit(self, data, target):
        """
        a fit function
        """
        self.data = data
        self.target = target
        self.independent = self.data.columns.tolist()
        self.independent.remove(target)
    
    def predict(self, data):
        return np.array([self.__flow_data_thru_tree(row) for row in data.values ])

    def __flow_data_thru_tree(self, row):
        return self.data[self.target].value_counts().apply(lambda n: x/len(self.data)).tolist()

    def __calculate_impurity_score(self, data):
        if data is None or data.empty:
            return 0 
        
        p_i, _ = data.value_counts().apply(lambda x: x/len(data)).tolist()
        return p_i * (1 - p_i) * 2


if __name__ == "__main__":
    train = pd.read_csv("./data/train_preprocessed.csv")
    test = pd.read_csv("./data/test_preprocessed.csv")

    model = DecisionTree()
    model.fit(data=test, target="Survived")

    pred = model.predict(test)

    print(pred)