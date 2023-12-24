import pandas as pd 
# Make sure all columns aree displayed
pd.set_option('display.max_columns', None)

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

    
data_loader = DataLoader()
data_loader.load_dataset()
data = data_loader.data
# print(data)

data_loader.preprocess_data()
data = data_loader.data
print(data)




