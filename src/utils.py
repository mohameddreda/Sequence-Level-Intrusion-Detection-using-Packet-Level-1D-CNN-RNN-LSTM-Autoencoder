import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class RealTimeDetector:
    def __init__(self, model, sequence_length, n_features, threshold):
        self.model = model
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.threshold = threshold
        self.buffer = []
    
    def add_flow(self, flow_features):
        """Add a new network flow to the buffer"""
        self.buffer.append(flow_features)
        
        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)
        
        if len(self.buffer) == self.sequence_length:
            sequence = np.array(self.buffer).reshape(1, self.sequence_length, self.n_features)
            reconstruction = self.model.predict(sequence, verbose=0)
            error = np.mean(np.square(sequence - reconstruction))
            is_anomaly = error > self.threshold
            return is_anomaly, error
        
        return False, 0.0

def save_results(results, filename):
    """Save evaluation results"""
    import json
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

def load_results(filename):
    """Load evaluation results"""
    import json
    with open(filename, 'r') as f:
        return json.load(f)