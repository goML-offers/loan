import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
import numpy as np


def preprocess_data(df):
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    df['Married'] = label_encoder.fit_transform(df['Married'])
    df['Self_Employed'] = label_encoder.fit_transform(df['Self_Employed'])
    df['Property_Area'] = label_encoder.fit_transform(df['Property_Area'])
    if 'Loan_Status'  in df.columns:
        df['Loan_Status'] = label_encoder.fit_transform(df['Loan_Status'])
    df['Education'] = label_encoder.fit_transform(df['Education'])
    df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce')

    mean_value = df['LoanAmount'].mean()
    df['LoanAmount'].fillna(mean_value, inplace=True)

    mean_value = df['Dependents'].mean()
    df['Dependents'].fillna(mean_value, inplace=True)

    mean_value = df['Loan_Amount_Term'].mean()
    df['Loan_Amount_Term'].fillna(mean_value, inplace=True)

    mean_value = df['Credit_History'].mean()
    df['Credit_History'].fillna(mean_value, inplace=True)

 

    return df

 

def train_loan_approval_model(path):
    # Load and preprocess the data
    df = pd.read_csv(path)
    df = preprocess_data(df)

    # Define features (X) and labels (y)
    X = df[['Gender', 'Married', 'Dependents', 'Education',
           'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
           'Loan_Amount_Term', 'Credit_History', 'Property_Area']]
    y = df['Loan_Status']

    # Initialize RandomForestClassifier
    rf_classifier = RandomForestClassifier(random_state=42)

    # Define hyperparameters for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }

    # Perform Grid Search Cross Validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    grid_search = GridSearchCV(rf_classifier, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)


    # Get the best estimator
    best_estimator = grid_search.best_estimator_

 

    # Predict labels for the test set
    y_pred = best_estimator.predict(X_test)

 

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # multiply by 10 with two decimal points
    accuracy = np.round(accuracy*100, 2)
    
    # print(f"Accuracy: {accuracy}")

 

    # Train RandomForestClassifier on the entire dataset
    rf_classifier.fit(X, y)

    

    # Save the model to a pickle file
    # with open('process_pred.pkl', 'wb') as model_file:
    #     pickle.dump(rf_classifier, model_file)
    
    precision = precision_score(y_test, y_pred, average='binary', pos_label=1)
    precision = np.round(precision*100, 2)
    recall = recall_score(y_test, y_pred, average='binary', pos_label=1)
    recall = np.round(recall*100, 2)
    
    
  
    return {"accuracy": accuracy,"precision": precision,"recall": recall}

 

def predict_loan_approval(input_path):
    input_data = pd.read_csv(input_path)
    with open('/api/process_pred_new.pkl', 'rb') as model_file:
        rf_classifier = pickle.load(model_file)
    
    predictions  = rf_classifier.predict(input_data)
    # # Predict using the loaded model
    
    print("Predictions",predictions,type(predictions))

    input_data['predictions'] = predictions
    
    return str(input_data.to_csv())

