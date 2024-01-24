from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from lime import lime_tabular

# Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()

# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_splited_data()

# Oversample the train data
X_train, y_train = data_loader.oversample_data(X_train, y_train)

# Use blackbox model and train it
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"F1 Score {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy {accuracy_score(y_test, y_pred)}")

# Create explainable LIME model
explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['healthy', 'ill'],
    mode='classification'
)

for x in range(6):
    # Explain given instance
    exp = explainer.explain_instance(
        data_row=X_test.iloc[x], 
        predict_fn=rf.predict_proba
    )

    # Export result to a file
    exp.save_to_file(f'lime'+str(x)+'.html')