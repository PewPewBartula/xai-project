import pandas as pd 
# Make sure all columns aree displayed
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

class DataLoader():

    # Data file path
    Path="healthcare-dataset-stroke-data.csv"

    def __init__(self):
        self.data = None

    def load_dataset(self):
        self.data = pd.read_csv(self.Path)

    def preprocess_data(self):
        # Encode columns with string values
        string_cols = ["gender",
                            "ever_married",
                            "work_type",
                            "Residence_type",
                            "smoking_status"]
        
        encoded_cols = pd.get_dummies(self.data[string_cols], 
                                prefix=string_cols)

        # Update data with new columns
        self.data = pd.concat([encoded_cols, self.data], axis=1)
        
        # Delete old columns
        self.data.drop(string_cols, axis=1, inplace=True)

        # Fill missing values of BMI with 0
        self.data.bmi = self.data.bmi.fillna(0)
        
        # Drop id as it is not needed
        self.data.drop(["id"], axis=1, inplace=True)

    def get_splited_data(self):
        # X is only input data, so everything without last column
        X = self.data.iloc[:,:-1]
        # Y is only output, so only last column
        y = self.data.iloc[:,-1]
        # Return splited data - for model testing and training  
        return train_test_split(X, y, test_size=0.2, random_state=252734)
    
    def oversample_data(self, X_train, y_train):
        oversampler = RandomOverSampler(random_state=252734)
        # Convert to numpy and oversample given data
        x_np, y_np = oversampler.fit_resample(X_train.to_numpy() , y_train.to_numpy())
        # Convert back to pandas
        x_over = pd.DataFrame(x_np, columns=X_train.columns)
        y_over = pd.Series(y_np, name=y_train.name)
        # Return oversampled data
        return x_over, y_over
    