import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report
from joblib import dump

class ModelPicker:
    """
    A class for selecting, fitting, and saving machine learning models for regression or classification problems.

    Methods:
    - __init__(): Initializes ModelPicker class attributes.
    - user_input(): Asks the user to specify the type of problem (regression or classification), and input file path.
    - data_preprocessing(): Performs data preprocessing tasks such as handling missing values and encoding string columns.
    - check_y_type(): Checks if the target variable type is compatible with the selected problem type.
    - handle_string_columns(): Encodes string columns if requested by the user.
    - create_models(): Initializes machine learning models and their respective hyperparameter grids based on the selected problem type.
    - fit_models(): Fits each model using grid search with cross-validation, evaluates performance metrics, and selects the best model.
    - save_model(): Saves the best model to a file using joblib.
    - run(): Executes the entire model selection, fitting, and saving process.
    """
    def __init__(self):
        self.user_type = ''
        self.df = None
        self.y_column = ''
        self.final_model = None
        self.models = {}
        self.param_grid = {}


    def user_input(self):
        """
        Asks the user to specify the type of problem (regression or classification), and input file path.
        """
        print("Are you solving a REGRESSION or CLASSIFICATION problem?")
        print("R = REGRESSION / C = CLASSIFICATION")
        self.user_type = input("R/C: ").upper()
        
        if self.user_type not in ["R", "C"]:
            print("Wrong input. Please choose between 'R' or 'C'.")
            return False
        
        print("You have selected a " + ("REGRESSION" if self.user_type == "R" else "CLASSIFICATION") + " problem.")
        
        file_path = input("Please provide the file path or filename: ")
        self.df = pd.read_csv(file_path)
        
        print("-" * 35)
        print(self.df.info())
        print("-" * 35)
        print("\nColumns in your dataset:\n", self.df.columns)
        print("-" * 35)
        
        self.y_column = input("What is your target variable (y): ")


    def data_preprocessing(self):
        """
        Performs data preprocessing tasks such as handling missing values and encoding string columns.
        """
        if not self.check_y_type():
            return False
        
        if self.df.isnull().any().any():
            print("Your data contains missing values. Please clean it before proceeding.")
            return False
        
        string_columns = [col for col in self.df.columns if self.df[col].dtype == 'object']
        if string_columns:
            print("You have STRING columns in your data: ", string_columns)
            user_answer = input("Would you like to encode them? Y/N: ").upper()
            if user_answer == "Y":
                self.handle_string_columns(string_columns)
            else:
                return False
        return True


    def check_y_type(self):
        """
        Checks if the target variable type is compatible with the selected problem type.
        """
        y_type = self.df[self.y_column].dtype
        if ((self.user_type == "C" and y_type in ['object', 'int64', 'float64']) or
            (self.user_type == "R" and y_type in ['int64', 'float64'])):
            print(f"Your target variable (y) is a {self.user_type} target.")
            return True
        else:
            print("There is an issue with the data type of your target variable (y).")
            return False


    def handle_string_columns(self, string_columns):
        """
        Encodes string columns if requested by the user.

        Args:
        - string_columns: A list of string column names.
        """
        for col in string_columns:
            self.df[col] = pd.factorize(self.df[col])[0]
        print("String columns have been encoded.")


    def create_models(self):
        """
        Initializes machine learning models and their respective hyperparameter grids based on the selected problem type.
        """
        if self.user_type == "R":
            self.models = {
                'Linear Regression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'Elastic Net': ElasticNet(),
                'Support Vector Regressor': SVR()
            }
            self.param_grid = {
                'Linear Regression': {},
                'Lasso': {'alpha': [0.01, 0.1, 1, 10]},
                'Ridge': {'alpha': [0.01, 0.1, 1, 10]},
                'Elastic Net': {'alpha': [0.01, 0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9]},
                'Support Vector Regressor': {'C': [0.1, 1, 10, 100], 'epsilon': [0.01, 0.1, 1], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
            }
        elif self.user_type == "C":
            self.models = {
                'Logistic Regression': LogisticRegression(),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Support Vector Classifier': SVC()
            }
            self.param_grid = {
                'Logistic Regression': {},
                'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7]},
                'Support Vector Classifier': {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
            }


    def fit_models(self):
        """
        Fits each model using grid search with cross-validation, evaluates performance metrics, and selects the best model.
        """
        X = self.df.drop(self.y_column, axis=1)
        y = self.df[self.y_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        best_score = float('inf')
        best_model = None
        
        for name, model in self.models.items():
            grid = GridSearchCV(model, self.param_grid[name], cv=5)
            grid.fit(X_train_scaled, y_train)
            y_pred = grid.predict(X_test_scaled)
            if self.user_type == "R":
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                print(f"Model: {name}, Mean Absolute Error: {mae}, R-squared (R2) Score: {r2}")
            elif self.user_type == "C":
                conf_matrix = confusion_matrix(y_test, y_pred)
                class_report = classification_report(y_test, y_pred)
                print(f"Model: {name}, Confusion Matrix:\n{conf_matrix}")
                print("Classification Report:\n", class_report)
            
            score = mean_squared_error(y_test, y_pred)
            if score < best_score:
                best_score = score
                best_model = grid.best_estimator_
        
        self.final_model = best_model
        print(f"The best model is {type(self.final_model).__name__} with MSE: {best_score}")


    def save_model(self):
        """
        Saves the best model to a file using joblib.
        """
        file_name = input("Enter a name for the saved model file: ")
        dump(self.final_model, f"{file_name}.joblib")
        print(f"The model has been saved as {file_name}.joblib")


    def run(self):
        """
        Executes the entire model selection, fitting, and saving process.
        """
        self.user_input()
        if self.data_preprocessing():
            self.create_models()
            self.fit_models()
            self.save_model()


if __name__ == "__main__":
    picker = ModelPicker()
    picker.run()