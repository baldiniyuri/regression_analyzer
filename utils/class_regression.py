import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score



class SalesRegressionAnalyzer():

    def __init__(self, filename) -> None:
        self.csv_file = pd.read_csv(filename, sep = ',')
        self.lm = LinearRegression()

    def csv_head(self):
        return self.csv_file.head()


    def csv_info(self):
        self.csv_file.info()
    

    def csv_describe(self):
        self.csv_file.describe()
    

    def get_columns_items(self):
        return self.csv_head().columns.tolist()
    

    def outliers_verify(self):
        fig, ax = plt.subplots()
        xticks = self.get_columns_items()
        ax.set_xticklabels(xticks)
        plt.show()
    

    def box_plot_graphic(self, columns):
        fig, ax = plt.subplots()
        self.csv_file.boxplot(column=columns, ax=ax)
        ax.set_xticklabels(columns)
        plt.show()
    

    def variables_verifications(self, y_vars="sales"):
        sns.pairplot(self.csv_file, x_vars=self.get_columns_items(), y_vars=y_vars)
        plt.figure()
        sns.heatmap(self.csv_file.corr(), annot=True)
        plt.show()

    
    def regression_module(self, param='sales'):
        self.csv_file.columns
        x = self.csv_file[self.get_columns_items()]
        y = self.csv_file[param]
        return x, y


    def regression_splitter(self):
        x, y = self.regression_module()

        X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size = 0.7, random_state = 20)
        return X_train, X_test, Y_train, Y_test


    def linear_regression(self):
        X_train, X_test, Y_train, Y_test = self.regression_splitter()
        
        self.lm.fit(X_train, Y_train)

        return X_train, X_test, Y_train, Y_test


    def score(self):
        _, X_test, _, Y_test = self.linear_regression()
        Y_pred = self.lm.predict(X_test)
        regression = r2_score(Y_test, Y_pred)
        return regression, Y_pred, Y_test


    def print_regression(self):
        X_train, X_test, Y_train, Y_test = self.regression_splitter()
        print(X_train.shape)
        print(X_test.shape)
        print(Y_train.shape)
        print(Y_test.shape)

        r = self.score()
        print(r)


    def sum_columns(self):
        sums = {}
        for column_name in self.csv_file.columns:
            column_sum = self.csv_file[column_name].sum()
            sums[column_name] = column_sum
        return sums
    

    def display_graphic(self):
        _, Y_pred, Y_test = self.score()
    
        c = [i for i in range(1, len(Y_test) + 1)]
        plt.figure(figsize=(8, 5))
        plt.plot(c, Y_test, color='blue')
        plt.plot(c, Y_pred, color='red')
        plt.legend(['Test data', 'Predicted data'], bbox_to_anchor=(1.05, 0.6))
        plt.xlabel('Index')
        plt.ylabel('Generated Sales')

        plt.show()



def init_analysis():
    print("Enter the CSV path:")
    path = str(input())
    regression = SalesRegressionAnalyzer(path)
    column_sums = regression.sum_columns()
    entry = [list(column_sums.values())]
    regression.linear_regression()
    predictions = regression.lm.predict(entry)
    regression.print_regression()
    regression.display_graphic()
    regression.box_plot_graphic(['youtube', 'facebook', 'newspaper'])
    regression.variables_verifications()
    print("Predicted sales:", predictions[0])
    