import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class FileSelector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()

    def select_file(self):
        file_path = filedialog.askopenfilename(title="Select a .txt file", filetypes=[("Text files", "*.txt")])
        if file_path:
            print(f"Selected file: {file_path}")
            return file_path
        else:
            print("No file selected.")
            return None


class DataPreprocessor:
    def __init__(self, input_filename):
        self.input_filename = input_filename

    def remove_prefix(self, output_filename, target_string):
        try:
            with open(self.input_filename, 'r') as input_file, open(output_filename, 'w') as output_file:
                for line in input_file:
                    target_index = line.find(target_string)
                    if target_index != -1:
                        output_file.write(line[target_index + len(target_string):])
                    else:
                        output_file.write("\n")
            print(f"Data has been successfully written to {output_filename}")
        except FileNotFoundError:
            print(f"The file {self.input_filename} was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def extract_month_data(self, output_filename, month):
        try:
            with open(self.input_filename, 'r') as input_file, open(output_filename, 'w') as output_file:
                for line in input_file:
                    date_index = 0
                    for i in range(len(line)):
                        if line[i:i+8].isdigit():
                            date_index = i
                            break
                    date = line[date_index:date_index+8]
                    line_month = date[4:6]
                    if line_month == month:
                        output_file.write(line)
            print(f"{self.get_month_name(month)} data has been successfully written to {output_filename}")
        except FileNotFoundError:
            print(f"The file {self.input_filename} was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_month_name(self, month):
        month_names = {
            "01": "January",
            "02": "February",
            "03": "March",
            "04": "April",
            "05": "May",
            "06": "June",
            "07": "July",
            "08": "August",
            "09": "September",
            "10": "October",
            "11": "November",
            "12": "December"
        }
        return month_names.get(month, "Unknown")

    def clean_data(self, output_filename):
        with open(self.input_filename, 'r') as file:
            lines = file.readlines()

        cleaned_lines = []
        for line in lines:
            line = ''.join(char for char in line if not char.isalpha())
            values = line.strip().split()
            cleaned_values = [value for value in values if value != '9999']
            cleaned_line = ' '.join(cleaned_values)
            cleaned_line = ' '.join(cleaned_line.split())
            cleaned_lines.append(cleaned_line)

        with open(output_filename, 'w') as file:
            file.writelines([line + '\n' for line in cleaned_lines])


class DataAnalyzer:
    def __init__(self, input_filename):
        self.input_filename = input_filename

    def load_data(self):
        data = pd.read_csv(self.input_filename, sep=' ', header=None, names=['Date', 'PRCP', 'SNWD', 'TAVG', 'TMAX', 'TMIN'])
        data.replace(-9999, np.nan, inplace=True)
        data['Year'] = data['Date'].apply(lambda x: int(str(x)[:4]))
        data['Month'] = data['Date'].apply(lambda x: int(str(x)[4:6]))
        data['Day'] = data['Date'].apply(lambda x: int(str(x)[6:8]))
        return data

    def split_data(self, data):
        target_vars = ['PRCP', 'SNWD', 'TAVG', 'TMAX', 'TMIN']
        feature_vars = ['Year', 'Month', 'Day']
        X = data[feature_vars]
        y_PRCP = data['PRCP']
        y_SNWD = data['SNWD']
        y_TAVG = data['TAVG']
        y_TMAX = data['TMAX']
        y_TMIN = data['TMIN']

        X_PRCP = X[y_PRCP.notna()]
        y_PRCP = y_PRCP[y_PRCP.notna()]
        X_SNWD = X[y_SNWD.notna()]
        y_SNWD = y_SNWD[y_SNWD.notna()]
        X_TAVG = X[y_TAVG.notna()]
        y_TAVG = y_TAVG[y_TAVG.notna()]
        X_TMAX = X[y_TMAX.notna()]
        y_TMAX = y_TMAX[y_TMAX.notna()]
        X_TMIN = X[y_TMIN.notna()]
        y_TMIN = y_TMIN[y_TMIN.notna()]

        X_train_PRCP, X_test_PRCP, y_train_PRCP, y_test_PRCP = train_test_split(X_PRCP, y_PRCP, test_size=0.2, random_state=42)
        X_train_SNWD, X_test_SNWD, y_train_SNWD, y_test_SNWD = train_test_split(X_SNWD, y_SNWD, test_size=0.2, random_state=42)
        X_train_TAVG, X_test_TAVG, y_train_TAVG, y_test_TAVG = train_test_split(X_TAVG, y_TAVG, test_size=0.2, random_state=42)
        X_train_TMAX, X_test_TMAX, y_train_TMAX, y_test_TMAX = train_test_split(X_TMAX, y_TMAX, test_size=0.2, random_state=42)
        X_train_TMIN, X_test_TMIN, y_train_TMIN, y_test_TMIN = train_test_split(X_TMIN, y_TMIN, test_size=0.2, random_state=42)

        return X_train_PRCP, X_test_PRCP, y_train_PRCP, y_test_PRCP, X_train_SNWD, X_test_SNWD, y_train_SNWD, y_test_SNWD, X_train_TAVG, X_test_TAVG, y_train_TAVG, y_test_TAVG, X_train_TMAX, X_test_TMAX, y_train_TMAX, y_test_TMAX, X_train_TMIN, X_test_TMIN, y_train_TMIN, y_test_TMIN

    def train_model(self, X_train, y_train, model_type):
        if model_type == '1':
            model = LinearRegression()
        elif model_type == '2':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == '3':
            model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.7, random_state=42)
        else:
            print("Invalid choice. Defaulting to Linear Regression.")
            model = LinearRegression()

        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"MSE: {mse}")

    def make_prediction(self, model, input_df):
        prediction = model.predict(input_df)
        return prediction


class WeatherPredictor:
    def __init__(self):
        self.file_selector = FileSelector()
        self.data_preprocessor = None
        self.data_analyzer = None

    def run(self):
        input_filename = self.file_selector.select_file()
        self.data_preprocessor = DataPreprocessor(input_filename)
        output_filename = "TempRemovedPrefix.txt"
        self.data_preprocessor.remove_prefix(output_filename, "TEKIRDAG TU")
        input_filename = output_filename
        output_filename = "TempRemovedPrefixMonth.txt"
        month = input("Enter the month to extract data for (1-12): ")
        self.data_preprocessor.extract_month_data(output_filename, month)
        input_filename = output_filename
        output_filename = "CleanedWeatherData.txt"
        self.data_preprocessor.clean_data(output_filename)
        self.data_analyzer = DataAnalyzer(output_filename)
        data = self.data_analyzer.load_data()
        X_train_PRCP, X_test_PRCP, y_train_PRCP, y_test_PRCP, X_train_SNWD, X_test_SNWD, y_train_SNWD, y_test_SNWD, X_train_TAVG, X_test_TAVG, y_train_TAVG, y_test_TAVG, X_train_TMAX, X_test_TMAX, y_train_TMAX, y_test_TMAX, X_train_TMIN, X_test_TMIN, y_train_TMIN, y_test_TMIN = self.data_analyzer.split_data(data)

        model_type = input("Choose a model: 1. Linear Regression, 2. Random Forest Regressor, 3. Gradient Boosting Regressor: ")
        models = []
        
        print("Checking for -9999 values in training data:")
        print("X_train_PRCP:", X_train_PRCP.eq(-9999).any().any())
        print("y_train_PRCP:", y_train_PRCP.eq(-9999).any())
        print("X_train_SNWD:", X_train_SNWD.eq(-9999).any().any())
        print("y_train_SNWD:", y_train_SNWD.eq(-9999).any())
        print("X_train_TAVG:", X_train_TAVG.eq(-9999).any().any())
        print("y_train_TAVG:", y_train_TAVG.eq(-9999).any())
        print("X_train_TMAX:", X_train_TMAX.eq(-9999).any().any())
        print("y_train_TMAX:", y_train_TMAX.eq(-9999).any())
        print("X_train_TMIN:", X_train_TMIN.eq(-9999).any().any())
        print("y_train_TMIN:", y_train_TMIN.eq(-9999).any())

        for X_train, y_train, X_test, y_test in zip([X_train_PRCP, X_train_SNWD, X_train_TAVG, X_train_TMAX, X_train_TMIN], [y_train_PRCP, y_train_SNWD, y_train_TAVG, y_train_TMAX, y_train_TMIN], [X_test_PRCP, X_test_SNWD, X_test_TAVG, X_test_TMAX, X_test_TMIN], [y_test_PRCP, y_test_SNWD, y_test_TAVG, y_test_TMAX, y_test_TMIN]):
            model = self.data_analyzer.train_model(X_train, y_train, model_type)
            self.data_analyzer.evaluate_model(model, X_test, y_test)
            models.append(model)

        year = 2025
        month = 4
        for day in range(1, 31):
            input_df = pd.DataFrame({'Year': [year], 'Month': [month], 'Day': [day]})
            predictions = []
            for i, model in enumerate(models):
                prediction = self.data_analyzer.make_prediction(model, input_df)
                predictions.append(prediction[0])

            print(f"Date: {day:02d}{month:02d}{year}, Predicted weather:")
            print(f"PRCP: {predictions[0]:.2f}")
            print(f"SNWD: {predictions[1]:.2f}")
            print(f"TAVG: {predictions[2]:.2f}")
            print(f"TMAX: {predictions[3]:.2f}")
            print(f"TMIN: {predictions[4]:.2f}")
            print()


if __name__ == "__main__":
    predictor = WeatherPredictor()
    predictor.run()

