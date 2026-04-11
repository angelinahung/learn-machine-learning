"""
Prometheus Metrics Exporter for ML Model Monitoring
Exports metrics for MLflow model serving including request count, latency, and errors.
"""

import time
import threading
from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest
from prometheus_client.core import CollectorRegistry, CONTENT_TYPE_LATEST
from flask import Flask, Response
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLModelMetrics:
    """
    Prometheus metrics collector for ML model serving
    """
    
    def __init__(self, registry=None):
        """
        Initialize metrics with optional custom registry
        """
        if registry is None:
            registry = CollectorRegistry()
        self.registry = registry
        
        # Metric 1: Request Count (Counter)
        self.request_count = Counter(
            'ml_model_requests_total',
            'Total number of requests made to the ML model',
            ['model_name', 'model_version', 'endpoint', 'method'],
            registry=self.registry
        )
        
        # Metric 2: Request Latency (Histogram) 
        self.request_latency = Histogram(
            'ml_model_request_duration_seconds',
            'Time spent processing ML model requests',
            ['model_name', 'model_version', 'endpoint'],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry
        )
        
        # Metric 3: Error Count (Counter)
        self.error_count = Counter(
            'ml_model_errors_total',
            'Total number of errors in ML model requests',
            ['model_name', 'model_version', 'endpoint', 'error_type'],
            registry=self.registry
        )
        
        # Additional useful metrics
        self.model_info = Gauge(
            'ml_model_info',
            'Information about the ML model',
            ['model_name', 'model_version', 'algorithm', 'features'],
            registry=self.registry
        )
        
        self.prediction_count = Counter(
            'ml_model_predictions_total',
            'Total number of predictions made',
            ['model_name', 'predicted_class'],
            registry=self.registry
        )
        
        # Set model info
        self.set_model_info(
            model_name="wine_classifier",
            model_version="v1.0",
            algorithm="RandomForest",
            features="13"
        )
        
        logger.info("MLModelMetrics initialized successfully")
    
    def set_model_info(self, model_name, model_version, algorithm, features):
        """
        Set model information metric
        """
        self.model_info.labels(
            model_name=model_name,
            model_version=model_version, 
            algorithm=algorithm,
            features=features
        ).set(1)
        
    def record_request(self, model_name="wine_classifier", model_version="v1.0", 
                      endpoint="/invocations", method="POST"):
        """
        Record a successful request
        """
        self.request_count.labels(
            model_name=model_name,
            model_version=model_version,
            endpoint=endpoint,
            method=method
        ).inc()
        
        logger.info(f"Request recorded: {model_name} {endpoint} {method}")
    
    def record_latency(self, duration, model_name="wine_classifier", 
                      model_version="v1.0", endpoint="/invocations"):
        """
        Record request latency
        """
        self.request_latency.labels(
            model_name=model_name,
            model_version=model_version,
            endpoint=endpoint
        ).observe(duration)
        
        logger.info(f"Latency recorded: {duration:.3f}s for {model_name} {endpoint}")
    
    def record_error(self, error_type, model_name="wine_classifier", 
                    model_version="v1.0", endpoint="/invocations"):
        """
        Record an error
        """
        self.error_count.labels(
            model_name=model_name,
            model_version=model_version,
            endpoint=endpoint,
            error_type=error_type
        ).inc()
        
        logger.error(f"Error recorded: {error_type} for {model_name} {endpoint}")
    
    def record_prediction(self, predicted_class, model_name="wine_classifier"):
        """
        Record a prediction
        """
        self.prediction_count.labels(
            model_name=model_name,
            predicted_class=str(predicted_class)
        ).inc()
        
        logger.info(f"Prediction recorded: class {predicted_class} for {model_name}")

class PrometheusExporter:
    """
    Prometheus exporter HTTP server
    """
    
    def __init__(self, port=8000, metrics_endpoint="/metrics"):
        self.port = port
        self.metrics_endpoint = metrics_endpoint
        self.app = Flask(__name__)
        self.metrics = MLModelMetrics()
        
        # Setup Flask routes
        self.setup_routes()
        
    def setup_routes(self):
        """
        Setup Flask routes for metrics exposure
        """
        @self.app.route(self.metrics_endpoint)
        def metrics():
            """
            Expose metrics in Prometheus format
            """
            try:
                metrics_output = generate_latest(self.metrics.registry)
                return Response(metrics_output, mimetype=CONTENT_TYPE_LATEST)
            except Exception as e:
                logger.error(f"Error generating metrics: {str(e)}")
                return Response("Error generating metrics", status=500)
        
        @self.app.route("/health")
        def health():
            """
            Health check endpoint
            """
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.route("/")
        def index():
            """
            Index page with links to endpoints
            """
            return {
                "service": "ML Model Prometheus Exporter",
                "endpoints": {
                    "metrics": self.metrics_endpoint,
                    "health": "/health"
                },
                "model": "wine_classifier",
                "version": "v1.0"
            }
    
    def start_server(self):
        """
        Start the Prometheus exporter server
        """
        logger.info(f"Starting Prometheus exporter on port {self.port}")
        logger.info(f"Metrics available at: http://localhost:{self.port}{self.metrics_endpoint}")
        
        try:
            self.app.run(host='0.0.0.0', port=self.port, debug=False, threaded=True)
        except Exception as e:
            logger.error(f"Error starting server: {str(e)}")
            raise
    
    def get_metrics(self):
        """
        Get the metrics object for recording metrics
        """
        return self.metrics

def simulate_model_traffic(metrics, duration=30):
    """
    Simulate model traffic for demonstration
    """
    import random
    import time
    
    logger.info(f"Simulating model traffic for {duration} seconds...")
    
    start_time = time.time()
    request_count = 0
    
    while time.time() - start_time < duration:
        try:
            # Simulate request
            request_start = time.time()
            
            # Record request
            metrics.record_request()
            request_count += 1
            
            # Simulate processing time (50ms to 500ms)
            processing_time = random.uniform(0.05, 0.5)
            time.sleep(processing_time)
            
            # Record latency
            metrics.record_latency(processing_time)
            
            # Simulate prediction result (wine classes: 0, 1, 2)
            predicted_class = random.choice([0, 1, 2])
            metrics.record_prediction(predicted_class)
            
            # Simulate occasional errors (5% error rate)
            if random.random() < 0.05:
                error_types = ["timeout", "invalid_input", "model_error"]
                error_type = random.choice(error_types)
                metrics.record_error(error_type)
            
            # Wait before next request (1-3 seconds)
            time.sleep(random.uniform(1, 3))
            
        except Exception as e:
            logger.error(f"Error in traffic simulation: {str(e)}")
            metrics.record_error("simulation_error")
    
    logger.info(f"Traffic simulation completed. Generated {request_count} requests.")

def main():
    """
    Main function to run Prometheus exporter
    """
    print("="*60)
    print("ML MODEL PROMETHEUS METRICS EXPORTER")
    print("="*60)
    
    # Initialize exporter
    exporter = PrometheusExporter(port=8000)
    metrics = exporter.get_metrics()
    
    # Start traffic simulation in background thread
    traffic_thread = threading.Thread(
        target=simulate_model_traffic, 
        args=(metrics, 60)
    )
    traffic_thread.daemon = True
    traffic_thread.start()
    
    print(f"Metrics server starting on port 8000")
    print(f"Metrics endpoint: http://localhost:8000/metrics")
    print(f"Traffic simulation running for 60 seconds")
    print(f"Press Ctrl+C to stop")
    
    try:
        exporter.start_server()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")

if __name__ == "__main__":
    main()