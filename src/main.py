from data_preprocessing import LogProcessor
from model import ThreatDetectionModel
from utils import DataUtils

def main():
    # Prétraitement des données
    log_processor = LogProcessor('data/logs/')
    logs = log_processor.load_logs()
    processed_logs = log_processor.preprocess_logs()
    word2vec_model = log_processor.train_word2vec_model(processed_logs)
    word2vec_model.save("word2vec_model.model")

    X_train, y_train = DataUtils.load_data('data/train_data.npy')
    X_test, y_test = DataUtils.load_data('data/test_data.npy')

    # Modèle de détection des menaces
    detection_model = ThreatDetectionModel((10, 100))  # Exemple de forme d'entrée
    detection_model.train(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Évaluation du modèle
    loss, accuracy = detection_model.evaluate(X_test, y_test)
    print(f"Final Loss: {loss}, Final Accuracy: {accuracy}")

if __name__ == "__main__":
    main()
