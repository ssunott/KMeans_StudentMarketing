from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_model(model, X_test, y_test):
    """Function to predict and evaluate model"""
    
    # Predict price on the testing set
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_pred, y_test)
    r2 = r2_score(y_test, y_pred)

    return mae, r2