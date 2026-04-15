"""
Model Inference Script with Prometheus Metrics
Sends prediction requests to MLflow served model and updates monitoring metrics
"""

import requests
import json
import time
import pandas as pd
import numpy as np
from prometheus_exporter import MLModelMetrics
import logging
import urllib.request
import random
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInferenceClient:
    """
    Client for making inference requests to MLflow served model
    """
    
    def __init__(self, model_url="http://localhost:1234", metrics=None):
        """
        Initialize inference client
        
        Args:
            model_url: URL of the served MLflow model
            metrics: MLModelMetrics instance for recording metrics
        """
        self.model_url = model_url
        self.predict_url = f"{model_url}/invocations"
        self.health_url = f"{model_url}/health"
        self.metrics = metrics or MLModelMetrics()
        
        logger.info(f"Initialized inference client for {model_url}")
    
    def check_model_health(self):
        """
        Check if the model server is healthy
        """
        try:
            response = requests.get(self.health_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    def generate_sample_data(self, n_samples=1):
        """
        Generate sample seeds data for prediction
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            pandas.DataFrame: Sample data
        """
        # Generate realistic seeds dataset samples
        feature_names = [
            'area', 'perimeter', 'compactness', 'length_of_kernel',
            'width_of_kernel', 'asymmetry_coefficient', 'length_of_kernel_groove'
        ]
        
        # Feature ranges based on Seeds dataset characteristics
        feature_ranges = {
            'area': (10.59, 21.18),
            'perimeter': (12.41, 17.25), 
            'compactness': (0.808, 0.958),
            'length_of_kernel': (4.899, 6.675),
            'width_of_kernel': (2.630, 3.700),
            'asymmetry_coefficient': (0.765, 8.456),
            'length_of_kernel_groove': (4.519, 6.550)
        }
        
        # Generate random samples within feature ranges
        samples = []
        for _ in range(n_samples):
            sample = {}
            for feature_name in feature_names:
                min_val, max_val = feature_ranges[feature_name]
                sample[feature_name] = random.uniform(min_val, max_val)
            samples.append(sample)
        
        return pd.DataFrame(samples)
    
    def make_prediction(self, data):
        """
        Make a prediction request to the served model
        
        Args:
            data: pandas.DataFrame with features for prediction
            
        Returns:
            dict: Response with prediction results and metrics
        """
        start_time = time.time()
        
        try:
            # Prepare request payload
            if isinstance(data, pd.DataFrame):
                # Convert DataFrame to list of lists (MLflow format)
                input_data = data.values.tolist()
            else:
                input_data = data
            
            payload = {
                "dataframe_split": {
                    "columns": data.columns.tolist() if isinstance(data, pd.DataFrame) else None,
                    "data": input_data
                }
            }
            
            # Record request start
            self.metrics.record_request(
                model_name="seeds_classifier",
                model_version="v1.0",
                endpoint="/invocations",
                method="POST"
            )
            
            # Make HTTP request
            response = requests.post(
                self.predict_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            # Calculate latency
            latency = time.time() - start_time
            self.metrics.record_latency(
                duration=latency,
                model_name="seeds_classifier",
                model_version="v1.0",
                endpoint="/invocations"
            )
            
            if response.status_code == 200:
                predictions = response.json()
                
                # Record predictions
                if isinstance(predictions, list):
                    for pred in predictions:
                        self.metrics.record_prediction(
                            predicted_class=pred,
                            model_name="seeds_classifier"
                        )
                
                logger.info(f"Prediction successful: {predictions} (latency: {latency:.3f}s)")
                
                return {
                    "success": True,
                    "predictions": predictions,
                    "latency": latency,
                    "status_code": response.status_code
                }
            else:
                # Record error
                self.metrics.record_error(
                    error_type=f"http_{response.status_code}",
                    model_name="seeds_classifier",
                    model_version="v1.0",
                    endpoint="/invocations"
                )
                
                logger.error(f"Prediction failed: HTTP {response.status_code}")
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "latency": latency,
                    "status_code": response.status_code
                }
                
        except requests.exceptions.RequestException as e:
            latency = time.time() - start_time
            self.metrics.record_error(
                error_type="connection_error",
                model_name="seeds_classifier",
                model_version="v1.0",
                endpoint="/invocations"
            )
            
            logger.error(f"Request failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "latency": latency,
                "status_code": None
            }
        
        except Exception as e:
            latency = time.time() - start_time
            self.metrics.record_error(
                error_type="unknown_error",
                model_name="seeds_classifier",
                model_version="v1.0",
                endpoint="/invocations"
            )
            
            logger.error(f"Unexpected error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "latency": latency,
                "status_code": None
            }

def simulate_inference_load(client, duration=60, requests_per_minute=10):
    """
    Simulate inference load for testing
    
    Args:
        client: ModelInferenceClient instance
        duration: How long to run simulation (seconds)
        requests_per_minute: Target requests per minute
    """
    logger.info(f"Starting inference load simulation for {duration} seconds")
    logger.info(f"Target rate: {requests_per_minute} requests per minute")
    
    start_time = time.time()
    request_count = 0
    success_count = 0
    error_count = 0
    
    # Calculate sleep time between requests
    sleep_time = 60.0 / requests_per_minute
    
    while time.time() - start_time < duration:
        try:
            # Generate sample data
            sample_data = client.generate_sample_data(n_samples=1)
            
            # Make prediction
            result = client.make_prediction(sample_data)
            request_count += 1
            
            if result["success"]:
                success_count += 1
                logger.info(f"Request {request_count}: SUCCESS - Predictions: {result['predictions']}")
            else:
                error_count += 1
                logger.error(f"Request {request_count}: ERROR - {result['error']}")
            
            # Sleep between requests
            time.sleep(sleep_time)
            
        except KeyboardInterrupt:
            logger.info("Simulation stopped by user")
            break
        except Exception as e:
            error_count += 1
            logger.error(f"Error in simulation: {str(e)}")
            time.sleep(1)  # Brief pause on error
    
    # Print summary
    elapsed_time = time.time() - start_time
    logger.info(f"\nSimulation Summary:")
    logger.info(f"Duration: {elapsed_time:.1f} seconds")
    logger.info(f"Total requests: {request_count}")
    logger.info(f"Successful requests: {success_count}")
    logger.info(f"Failed requests: {error_count}")
    logger.info(f"Success rate: {(success_count/request_count*100) if request_count > 0 else 0:.1f}%")
    logger.info(f"Average requests per minute: {(request_count/(elapsed_time/60)):.1f}")

def single_prediction_demo(client):
    """
    Demonstrate a single prediction request
    """
    logger.info("\n" + "="*50)
    logger.info("SINGLE PREDICTION DEMONSTRATION")
    logger.info("="*50)
    
    # Generate sample data
    sample_data = client.generate_sample_data(n_samples=1)
    logger.info(f"Generated sample data shape: {sample_data.shape}")
    logger.info(f"Sample features (first 5): {list(sample_data.columns)[:5]}")
    
    # Make prediction
    result = client.make_prediction(sample_data)
    
    if result["success"]:
        logger.info(f"Prediction successful!")
        logger.info(f"Predicted seeds class: {result['predictions']}")
        logger.info(f"Response time: {result['latency']:.3f} seconds")
    else:
        logger.error(f"❌ Prediction failed: {result['error']}")
        logger.error(f"Response time: {result['latency']:.3f} seconds")
    
    return result

def batch_prediction_demo(client, batch_size=5):
    """
    Demonstrate batch prediction requests
    """
    logger.info("\n" + "="*50)
    logger.info(f"BATCH PREDICTION DEMONSTRATION ({batch_size} samples)")
    logger.info("="*50)
    
    # Generate batch data
    batch_data = client.generate_sample_data(n_samples=batch_size)
    logger.info(f"Generated batch data shape: {batch_data.shape}")
    
    # Make batch prediction
    result = client.make_prediction(batch_data)
    
    if result["success"]:
        logger.info(f"Batch prediction successful!")
        logger.info(f"Predicted seeds classes: {result['predictions']}")
        logger.info(f"Response time: {result['latency']:.3f} seconds")
        logger.info(f"Average time per sample: {result['latency']/batch_size:.3f} seconds")
    else:
        logger.error(f"❌ Batch prediction failed: {result['error']}")
        logger.error(f"Response time: {result['latency']:.3f} seconds")
    
    return result

def main():
    """
    Main function to run inference testing
    """
    print("="*60)
    print("ML MODEL INFERENCE TESTING WITH PROMETHEUS METRICS")
    print("="*60)
    
    # Initialize metrics
    metrics = MLModelMetrics()
    
    # Initialize inference client
    client = ModelInferenceClient("http://localhost:1234", metrics=metrics)
    
    # Check if model server is running
    logger.info("Checking model server health...")
    if not client.check_model_health():
        logger.warning("⚠️  Model server is not responding at http://localhost:1234")
        logger.warning("Please start the MLflow model server first:")
        logger.warning("mlflow models serve -m runs:/<run_id>/model -h 0.0.0.0 -p 1234")
        logger.info("Continuing with mock predictions for metrics demonstration...")
    else:
        logger.info("Model server is healthy")
    
    try:
        # Single prediction demo
        single_prediction_demo(client)
        
        # Batch prediction demo  
        batch_prediction_demo(client, batch_size=3)
        
        # Simulate load testing
        logger.info("\n" + "="*50)
        logger.info("LOAD TESTING SIMULATION")
        logger.info("="*50)
        logger.info("Starting 30-second load test...")
        simulate_inference_load(
            client, 
            duration=30, 
            requests_per_minute=20
        )
        
        logger.info("\nInference testing completed!")
        logger.info("Check Prometheus metrics at: http://localhost:8000/metrics")
        
    except KeyboardInterrupt:
        logger.info("\nInference testing stopped by user")
    except Exception as e:
        logger.error(f"Error in inference testing: {str(e)}")

if __name__ == "__main__":
    main()
