"""
Prometheus Exporter for ML Model Monitoring
Provides custom metrics for machine learning model performance and system health
"""

import time
import threading
from prometheus_client import start_http_server, Counter, Histogram, Gauge, Info
from prometheus_client.core import CollectorRegistry
from datetime import datetime
import logging
import psutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLModelMetrics:
    """
    Custom metrics collector for ML model monitoring
    Tracks prediction requests, response times, errors, and system metrics
    """
    
    def __init__(self, port=8000):
        """
        Initialize Prometheus metrics
        
        Args:
            port: Port for Prometheus metrics server
        """
        self.port = port
        self.registry = CollectorRegistry()
        
        # Model Performance Metrics
        self.prediction_requests_total = Counter(
            'ml_model_prediction_requests_total',
            'Total number of prediction requests',
            ['model_name', 'model_version', 'status'],
            registry=self.registry
        )
        
        self.prediction_duration_seconds = Histogram(
            'ml_model_prediction_duration_seconds',
            'Time spent on prediction requests',
            ['model_name', 'model_version'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            'ml_model_accuracy_percentage',
            'Current model accuracy percentage',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        # System Health Metrics
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percentage',
            'Current CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_bytes',
            'Current memory usage in bytes',
            ['type'],  # used, available, total
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'system_disk_usage_percentage',
            'Current disk usage percentage',
            registry=self.registry
        )
        
        # Model-specific Metrics
        self.active_model_predictions = Gauge(
            'ml_model_active_predictions',
            'Number of currently active predictions',
            ['model_name'],
            registry=self.registry
        )
        
        self.model_load_time_seconds = Gauge(
            'ml_model_load_time_seconds',
            'Time taken to load the model',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        # Error Tracking
        self.prediction_errors_total = Counter(
            'ml_model_prediction_errors_total',
            'Total number of prediction errors',
            ['model_name', 'error_type'],
            registry=self.registry
        )
        
        # Model Information
        self.model_info = Info(
            'ml_model_info',
            'Information about the ML model',
            registry=self.registry
        )
        
        # Data Quality Metrics
        self.input_data_quality_score = Gauge(
            'ml_model_input_data_quality_score',
            'Quality score of input data (0-1)',
            ['model_name'],
            registry=self.registry
        )
        
        self.feature_drift_score = Gauge(
            'ml_model_feature_drift_score',
            'Feature drift detection score',
            ['model_name', 'feature_name'],
            registry=self.registry
        )
        
        # Initialize model info
        self.model_info.info({
            'name': 'seeds_classification_model',
            'version': '1.0.0',
            'algorithm': 'Random Forest',
            'framework': 'scikit-learn',
            'created_date': datetime.now().strftime('%Y-%m-%d'),
            'description': 'Seeds dataset classification model'
        })
        
        # Set initial accuracy value
        self.model_accuracy.labels(model_name='seeds_model', model_version='1.0.0').set(95.2)
        
        # Start background thread for system metrics
        self.system_metrics_thread = threading.Thread(target=self._update_system_metrics, daemon=True)
        self.system_metrics_thread.start()
        
        logger.info(f"ML Model metrics initialized on port {port}")
    
    def _update_system_metrics(self):
        """
        Background thread to update system metrics
        """
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.system_cpu_usage.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.system_memory_usage.labels(type='used').set(memory.used)
                self.system_memory_usage.labels(type='available').set(memory.available) 
                self.system_memory_usage.labels(type='total').set(memory.total)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.system_disk_usage.set(disk_percent)
                
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error updating system metrics: {str(e)}")
                time.sleep(30)  # Wait longer on error
    
    def record_prediction_request(self, model_name='seeds_model', model_version='1.0.0', status='success'):
        """
        Record a prediction request
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            status: Status of the request (success, error, timeout)
        """
        self.prediction_requests_total.labels(
            model_name=model_name,
            model_version=model_version,
            status=status
        ).inc()
        logger.info(f"Recorded prediction request: {model_name} v{model_version} - {status}")
    
    def record_prediction_duration(self, duration, model_name='seeds_model', model_version='1.0.0'):
        """
        Record prediction duration
        
        Args:
            duration: Duration in seconds
            model_name: Name of the model
            model_version: Version of the model
        """
        self.prediction_duration_seconds.labels(
            model_name=model_name,
            model_version=model_version
        ).observe(duration)
        logger.info(f"Recorded prediction duration: {duration:.3f}s for {model_name}")
    
    def update_model_accuracy(self, accuracy, model_name='seeds_model', model_version='1.0.0'):
        """
        Update model accuracy metric
        
        Args:
            accuracy: Accuracy percentage (0-100)
            model_name: Name of the model
            model_version: Version of the model
        """
        self.model_accuracy.labels(model_name=model_name, model_version=model_version).set(accuracy)
        logger.info(f"Updated model accuracy: {accuracy}% for {model_name}")
    
    def record_prediction_error(self, error_type, model_name='seeds_model'):
        """
        Record a prediction error
        
        Args:
            error_type: Type of error (timeout, validation, processing, etc.)
            model_name: Name of the model
        """
        self.prediction_errors_total.labels(model_name=model_name, error_type=error_type).inc()
        logger.error(f"Recorded prediction error: {error_type} for {model_name}")
    
    def set_active_predictions(self, count, model_name='seeds_model'):
        """
        Set number of active predictions
        
        Args:
            count: Number of active predictions
            model_name: Name of the model
        """
        self.active_model_predictions.labels(model_name=model_name).set(count)
    
    def record_model_load_time(self, load_time, model_name='seeds_model', model_version='1.0.0'):
        """
        Record model loading time
        
        Args:
            load_time: Time in seconds to load the model
            model_name: Name of the model  
            model_version: Version of the model
        """
        self.model_load_time_seconds.labels(
            model_name=model_name,
            model_version=model_version
        ).set(load_time)
        logger.info(f"Recorded model load time: {load_time:.3f}s for {model_name}")
    
    def update_data_quality_score(self, score, model_name='seeds_model'):
        """
        Update input data quality score
        
        Args:
            score: Quality score between 0 and 1
            model_name: Name of the model
        """
        self.input_data_quality_score.labels(model_name=model_name).set(score)
    
    def update_feature_drift_score(self, score, feature_name, model_name='seeds_model'):
        """
        Update feature drift score
        
        Args:
            score: Drift score
            feature_name: Name of the feature
            model_name: Name of the model
        """
        self.feature_drift_score.labels(model_name=model_name, feature_name=feature_name).set(score)
    
    def start_server(self):
        """
        Start the Prometheus metrics HTTP server
        """
        try:
            start_http_server(self.port, registry=self.registry)
            logger.info(f"Prometheus metrics server started on port {self.port}")
            logger.info(f"Metrics available at http://localhost:{self.port}/metrics")
            
            # Generate some initial metrics
            self._generate_initial_metrics()
            
            return True
        except Exception as e:
            logger.error(f"Failed to start metrics server: {str(e)}")
            return False
    
    def _generate_initial_metrics(self):
        """
        Generate some initial metrics for demonstration
        """
        # Simulate some prediction requests
        self.record_prediction_request(status='success')
        self.record_prediction_request(status='success')
        self.record_prediction_request(status='error')
        
        # Record some durations
        self.record_prediction_duration(0.234)
        self.record_prediction_duration(0.156)
        self.record_prediction_duration(0.312)
        
        # Set model load time
        self.record_model_load_time(2.45)
        
        # Set data quality scores
        self.update_data_quality_score(0.95)
        
        # Set feature drift scores
        self.update_feature_drift_score(0.12, 'area')
        self.update_feature_drift_score(0.08, 'perimeter')
        self.update_feature_drift_score(0.15, 'compactness')
        
        logger.info("Initial metrics generated successfully")


def main():
    """
    Main function to start the Prometheus exporter
    """
    logger.info("Starting ML Model Prometheus Exporter...")
    
    # Initialize metrics
    metrics = MLModelMetrics(port=8000)
    
    # Start server
    if metrics.start_server():
        logger.info("Prometheus exporter running. Press Ctrl+C to stop.")
        try:
            # Keep the server running
            while True:
                time.sleep(10)
                # Simulate some ongoing activity
                metrics.record_prediction_request()
                metrics.record_prediction_duration(0.2 + (time.time() % 1))
        except KeyboardInterrupt:
            logger.info("Shutting down Prometheus exporter...")
    else:
        logger.error("Failed to start Prometheus exporter")
        exit(1)


if __name__ == "__main__":
    main()