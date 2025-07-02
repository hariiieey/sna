from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import numpy as np

class ModelMonitor:
    def __init__(self):
        self.metrics_history = []
    def log_prediction_metrics(self, predictions, ground_truth):
        predictions = np.array(predictions)  # Ensure numpy array for thresholding
        current_metrics = {
            'timestamp': datetime.now(),
            'auc': roc_auc_score(ground_truth, predictions),
            'precision': precision_score(ground_truth, predictions > 0.5),
            'recall': recall_score(ground_truth, predictions > 0.5)
        }
        self.metrics_history.append(current_metrics)
        if self.detect_performance_drift():
            self.trigger_retraining_alert()
    def detect_performance_drift(self, window_size=100):
        if len(self.metrics_history) < window_size:
            return False
        recent_auc = np.mean([m['auc'] for m in self.metrics_history[-window_size:]])
        historical_auc = np.mean([m['auc'] for m in self.metrics_history[:-window_size]])
        return recent_auc < historical_auc - 0.05
    def trigger_retraining_alert(self):
        print('ALERT: Model performance drift detected! Consider retraining.') 