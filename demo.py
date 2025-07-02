from application import EcommerceRecommendationSystem
from deploy import TemporalLinkPredictionService
from monitor import ModelMonitor

print("\n--- Application Demo ---")
# Demo recommendation system
app = EcommerceRecommendationSystem(model=None, user_encoder=None, product_encoder=None)
recs = app.generate_recommendations(user_id=1, timestamp=0, top_k=5)
print("Recommendations:", recs)
fraud = app.detect_fraud_patterns(user_interactions=[1,2,3,4,5,6,7,8,9,10,11])
print("Fraud Patterns:", fraud)

print("\n--- Deployment Demo ---")
# Demo deployment service
service = TemporalLinkPredictionService(model_path='best_model.pth', config={})
preds = service.predict_future_interactions(user_id=1, timestamp=0)
print("Predicted Future Interactions:", preds)

print("\n--- Monitoring Demo ---")
# Demo monitoring
monitor = ModelMonitor()
monitor.log_prediction_metrics(predictions=[0.9,0.1,0.8,0.2], ground_truth=[1,0,1,0])
monitor.log_prediction_metrics(predictions=[0.7,0.3,0.6,0.4], ground_truth=[1,0,1,0])
print("Metrics History:")
for m in monitor.metrics_history:
    print(m) 