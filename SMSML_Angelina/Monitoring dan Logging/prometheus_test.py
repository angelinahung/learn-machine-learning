#!/usr/bin/env python3
"""
Simple Prometheus Metrics Test Server
Basic implementation for evidence collection
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest
import time
import random
import threading
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Prometheus metrics
request_count = Counter('ml_model_requests_total', 'Total ML model requests', ['model_name', 'endpoint'])
request_latency = Histogram('ml_model_request_duration_seconds', 'ML model request latency', ['model_name'])
error_count = Counter('ml_model_errors_total', 'Total ML model errors', ['error_type'])
model_info = Gauge('ml_model_info', 'ML model information', ['model_name', 'version', 'algorithm'])
prediction_count = Counter('ml_model_predictions_total', 'Total predictions', ['predicted_class'])

def simulate_metrics():
    """Simulate some metrics data"""
    # Set model info
    model_info.labels(model_name='wine_classifier', version='v1.0', algorithm='RandomForest').set(1)
    
    while True:
        try:
            # Simulate request
            request_count.labels(model_name='wine_classifier', endpoint='/invocations').inc()
            
            # Simulate latency (50ms to 500ms)
            latency = random.uniform(0.05, 0.5)
            request_latency.labels(model_name='wine_classifier').observe(latency)
            
            # Simulate prediction (wine classes 0, 1, 2)
            predicted_class = random.choice(['class_0', 'class_1', 'class_2'])
            prediction_count.labels(predicted_class=predicted_class).inc()
            
            # Simulate occasional errors (10% error rate)
            if random.random() < 0.1:
                error_type = random.choice(['timeout', 'invalid_input', 'model_error'])
                error_count.labels(error_type=error_type).inc()
            
            logger.info(f"Metrics updated - Latency: {latency:.3f}s, Class: {predicted_class}")
            
            # Wait 2-5 seconds before next simulation
            time.sleep(random.uniform(2, 5))
            
        except Exception as e:
            logger.error(f"Error in simulation: {e}")
            time.sleep(1)

def main():
    print("="*60)
    print("ML MODEL PROMETHEUS METRICS SERVER")
    print("="*60)
    print("Starting Prometheus metrics server...")
    print("Metrics available at: http://localhost:8000/metrics")
    print("Press Ctrl+C to stop")
    print("="*60)
    
    # Start metrics simulation thread
    metrics_thread = threading.Thread(target=simulate_metrics, daemon=True)
    metrics_thread.start()
    
    # Start Prometheus server
    start_http_server(8000)
    logger.info("Prometheus server started on port 8000")
    
    try:
        # Keep server running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServer stopped by user")

if __name__ == "__main__":
    main()