import numpy as np

class DataUtils:
    @staticmethod
    def load_data(file_path):
        # Fonction pour charger les données à partir d'un fichier
        data = np.load(file_path)
        return data

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        # Évaluation du modèle
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Loss: {loss}, Accuracy: {accuracy}")
