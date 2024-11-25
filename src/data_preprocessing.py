import pandas as pd
from gensim.models import Word2Vec
import os

class LogProcessor:
    def __init__(self, log_directory):
        self.log_directory = log_directory
        self.logs = []

    def load_logs(self):
        for filename in os.listdir(self.log_directory):
            if filename.endswith('.log'):
                with open(os.path.join(self.log_directory, filename), 'r') as file:
                    self.logs.append(file.readlines())
        return self.logs

    def preprocess_logs(self):
        processed_logs = []
        for log in self.logs:
            tokens = [line.strip().split() for line in log]
            processed_logs.extend(tokens)
        return processed_logs

    def train_word2vec_model(self, processed_logs):
        model = Word2Vec(sentences=processed_logs, vector_size=100, window=5, min_count=1, workers=4)
        return model

if __name__ == "__main__":
    log_processor = LogProcessor('data/logs/')
    logs = log_processor.load_logs()
    processed_logs = log_processor.preprocess_logs()
    word2vec_model = log_processor.train_word2vec_model(processed_logs)
    word2vec_model.save("word2vec_model.model")
