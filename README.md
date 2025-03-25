Introduction
This project uses different machine learning models (Linear Regression, Random Forest Regressor, Gradient Boosting Regressor) to predict weather patterns. The data used for this project is obtained from the National Centers for Environmental Information.

Data Description
The data is stored in a file and contains the following columns:

STATION
STATION_NAME
DATE
PRCP (Precipitation)
SNWD (Snow Depth)
TAVG (Average Temperature)
TMAX (Maximum Temperature)
TMIN (Minimum Temperature)
Data Preprocessing
The data is preprocessed to remove any unnecessary columns and values. The remove_prefix function is used to clear out the data until the date column. Additionally, the code cleans out any unknown values represented by "9999" and words that represent wind direction.

Model Training
The preprocessed data is then used to train different machine learning models. The DataAnalyzer class is used to train the models and allows for fine-tuning of hyperparameters such as learning_rate and n_estimators.

Example Use Case
1- Download weather data and python code.
2- Run it choose the weather1963to2024.txt.
3- Select month to extract from the dataset.
4- Choose a model to predict weather patterns for the next year.
