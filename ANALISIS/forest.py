### I M P O R T S ###
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import tree

import matplotlib.pyplot as plt
from colorama import Fore


### C L A S S ###

class RandomForestModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.random_forest = RandomForestRegressor(n_estimators=100, random_state=42)

    def load_data(self):
        df = dd.read_csv(self.data_path)
        X = df[['Month']]
        y = df['Passenger Count']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        self.X_train = X_train.compute()
        self.X_test = X_test.compute()
        self.y_train = y_train.compute()
        self.y_test = y_test.compute()

    def describe(self):
        df = dd.read_csv(self.data_path)
        description = df[['Month', 'Passenger Count']].describe().compute()
        print(description)

    def train(self):
        self.random_forest.fit(self.X_train, self.y_train)

    def predict(self):
        y_pred = self.random_forest.predict(self.X_test)
        return y_pred

    def calculate_metrics(self, y_pred):
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(Fore.GREEN + "Mean Squared Error:"+Fore.WHITE, mse)
        print(Fore.GREEN + "Mean Absolute Error:"+Fore.WHITE, mae)
        print(Fore.GREEN + "R^2 Score:"+Fore.WHITE, r2)
        return mse, mae, r2

    def visualize_tree(self, tree_index=0):
        plt.figure(figsize=(10, 10))
        tree.plot_tree(self.random_forest.estimators_[tree_index], feature_names=['Month'], filled=True)
        plt.show()

    def get_average_depth(self):
        total_depth = 0
        for estimator in self.random_forest.estimators_:
            total_depth += estimator.tree_.max_depth
        average_depth = total_depth / len(self.random_forest.estimators_)
        print(Fore.GREEN + "Average Tree Depth:"+Fore.WHITE, average_depth)
        return average_depth

    def show_all(self):
        self.load_data()
        self.describe()
        self.train()
        y_pred = self.predict()
        self.calculate_metrics(y_pred)
        self.get_average_depth()
        self.visualize_tree(tree_index=0)


# Crear una instancia del modelo
model = RandomForestModel('/Users/mariagarcia/Documents/BigData_datos_aeropuerto/dataset_limpiado.csv')

