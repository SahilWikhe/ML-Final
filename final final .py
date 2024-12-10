import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.svm import LinearSVC  # Faster linear SVM implementation
from sklearn.calibration import CalibratedClassifierCV  # For probability estimates
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import os

def load_data():
    """
    Load the Adult Census Income dataset
    """
    print("Loading data...")
    X, y = fetch_openml("adult", version=1, as_frame=True, return_X_y=True)
    return X, y

def explore_data(X, y):
    """
    Perform initial data exploration
    """
    print("\nData Exploration:")
    print("-----------------")
    print(f"Dataset shape: {X.shape}")
    print("\nFeature names:", X.columns.tolist())
    print("\nMissing values:\n", X.isnull().sum())
    
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    print("\nNumerical Features Statistics:")
    print(X[numerical_columns].describe())

def preprocess_data(X, y):
    """
    Preprocess the data including handling missing values and encoding categorical variables
    """
    print("\nPreprocessing data...")
    
    numerical_columns = ['age', 'fnlwgt', 'education-num', 'capitalgain', 'capitalloss', 'hoursperweek']
    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 
                         'relationship', 'race', 'sex', 'native-country']
    
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_columns),
            ('cat', categorical_pipeline, categorical_columns)
        ])

    X_processed = preprocessor.fit_transform(X)
    
    onehot_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_columns)
    feature_names = numerical_columns + list(onehot_features)
    
    X_processed = pd.DataFrame(X_processed, columns=feature_names)
    
    label_encoder_y = LabelEncoder()
    y_processed = label_encoder_y.fit_transform(y)
    
    return X_processed, y_processed, preprocessor

def evaluate_model(model, X, y, dataset_name):
    """
    Evaluate model on given dataset and return accuracy
    """
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"{dataset_name} Accuracy: {accuracy:.4f}")
    return accuracy

def subsample_data(X, y, max_samples=10000):
    """
    Subsample the data if it's too large to speed up training
    """
    if X.shape[0] > max_samples:
        print(f"\nSubsampling data from {X.shape[0]} to {max_samples} samples for faster training...")
        indices = np.random.choice(X.shape[0], max_samples, replace=False)
        return X.iloc[indices], y[indices]
    return X, y

def train_models(X, y):
    """
    Train and evaluate multiple models with train/validation/test splits
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Second split: separate train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    
    print(f"\nData split sizes:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create a faster SVM implementation
    base_svm = LinearSVC(
        dual='auto',  # Choose the faster solver automatically
        C=1.0,
        max_iter=1000,
        random_state=42
    )
    
    # Wrap with CalibratedClassifierCV for probability estimates
    # Using fewer CV folds (2) for speed
    svm_with_proba = CalibratedClassifierCV(base_svm, cv=2, n_jobs=-1)
    
    models = {
        'SVM': svm_with_proba,
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate on all sets
        print(f"\n{name} Performance:")
        train_accuracy = evaluate_model(model, X_train, y_train, "Training")
        val_accuracy = evaluate_model(model, X_val, y_val, "Validation")
        test_accuracy = evaluate_model(model, X_test, y_test, "Test")
        
        # Make predictions for test set
        y_pred = model.predict(X_test)
        
        # Store results
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'accuracies': {
                'train': train_accuracy,
                'validation': val_accuracy,
                'test': test_accuracy
            }
        }
        
        print(f"\n{name} Detailed Results:")
        print("Classification Report:")
        print(results[name]['classification_report'])
        
        # For SVM, print coefficients information
        if name == 'SVM':
            try:
                # Access the base estimator directly
                base_estimator = model.calibrated_classifiers_[0].estimator
                print("\nModel Coefficients Summary:")
                coef_summary = pd.DataFrame({
                    'feature': X.columns,
                    'coefficient': np.abs(base_estimator.coef_[0])
                }).sort_values('coefficient', ascending=False)
                print("Top 10 Most Important Features based on coefficient magnitude:")
                print(coef_summary.head(10))
            except Exception as e:
                print("Note: Could not access model coefficients due to:", str(e))
    
    return results, X_train, X_val, X_test, y_train, y_val, y_test

def plot_results(results, X, y):
    """
    Create visualizations for model results including accuracy comparisons
    """
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Plot confusion matrices
    for name, result in results.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'plots/confusion_matrix_{name.lower().replace(" ", "_")}.png')
        plt.close()
    
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    accuracies = {
        'Model': [],
        'Dataset': [],
        'Accuracy': []
    }
    
    for model_name, result in results.items():
        for dataset, accuracy in result['accuracies'].items():
            accuracies['Model'].append(model_name)
            accuracies['Dataset'].append(dataset.capitalize())
            accuracies['Accuracy'].append(accuracy)
    
    accuracy_df = pd.DataFrame(accuracies)
    sns.barplot(data=accuracy_df, x='Model', y='Accuracy', hue='Dataset')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/accuracy_comparison.png')
    plt.close()
    
    # For SVM, plot coefficient importance
    if 'SVM' in results:
        try:
            svm_model = results['SVM']['model'].calibrated_classifiers_[0].estimator
            coef_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': np.abs(svm_model.coef_[0])
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 6))
            sns.barplot(data=coef_importance.head(10), x='importance', y='feature')
            plt.title('Top 10 Feature Importance - Linear SVM')
            plt.xlabel('Absolute Coefficient Value')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig('plots/svm_coefficient_importance.png')
            plt.close()
        except Exception as e:
            print("Note: Could not create SVM coefficient plot due to:", str(e))

def main():
    """
    Main function to run the entire pipeline
    """
    # Load data
    X, y = load_data()
    
    # Explore data
    explore_data(X, y)
    
    # Preprocess data
    X_processed, y_processed, preprocessor = preprocess_data(X, y)
    
    X_subsampled, y_subsampled = subsample_data(X_processed, y_processed)
    
    # Train and evaluate models
    results, X_train, X_val, X_test, y_train, y_val, y_test = train_models(X_subsampled, y_subsampled)
    
    # Plot results
    plot_results(results, X_processed, y_processed)
    
    print("\nPipeline completed! Check the 'plots' directory for visualizations.")
    
    return results, X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    results, X_train, X_val, X_test, y_train, y_val, y_test = main()