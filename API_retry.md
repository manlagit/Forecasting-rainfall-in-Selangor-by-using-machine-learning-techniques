# API Retry Strategy for Failed Requests

## Overview
This document outlines a comprehensive retry strategy for handling failed API requests with exponential backoff, circuit breaker patterns, and intelligent error handling.

## 1. EXPONENTIAL BACKOFF RETRY MECHANISM

### Basic Retry Configuration
```python
import time
import random
from typing import Optional, Callable, Any
from functools import wraps

class APIRetryConfig:
    def __init__(self):
        self.max_retries = 3
        self.base_delay = 1.0  # seconds
        self.max_delay = 60.0  # seconds
        self.exponential_base = 2
        self.jitter = True
        self.retry_on_status = [429, 500, 502, 503, 504]
```

### Retry Decorator Implementation
```python
def api_retry(config: APIRetryConfig = None):
    """
    Decorator for automatic API retry with exponential backoff
    """
    if config is None:
        config = APIRetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(config.max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # Don't retry on final attempt
                    if attempt == config.max_retries:
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    # Add jitter to prevent thundering herd
                    if config.jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    
                    print(f"API request failed (attempt {attempt + 1}/{config.max_retries + 1}). "
                          f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
            
            # All retries exhausted
            raise last_exception
        
        return wrapper
    return decorator
```

## 2. CIRCUIT BREAKER PATTERN

### Circuit Breaker Implementation
```python
import threading
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        with self.lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self):
        return (datetime.now() - self.last_failure_time).seconds >= self.timeout
    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

## 3. INTELLIGENT ERROR HANDLING

### Error Classification
```python
class APIErrorHandler:
    RETRYABLE_ERRORS = {
        # HTTP Status Codes
        429: "Rate Limited",
        500: "Internal Server Error",
        502: "Bad Gateway",
        503: "Service Unavailable",
        504: "Gateway Timeout",
        
        # Connection Errors
        "ConnectionError": "Network connection failed",
        "Timeout": "Request timeout",
        "SSLError": "SSL/TLS error"
    }
    
    NON_RETRYABLE_ERRORS = {
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        422: "Unprocessable Entity"
    }
    
    @classmethod
    def should_retry(cls, error) -> bool:
        """Determine if an error should trigger a retry"""
        if hasattr(error, 'status_code'):
            return error.status_code in cls.RETRYABLE_ERRORS
        
        error_type = type(error).__name__
        return error_type in cls.RETRYABLE_ERRORS
```

## 4. COMPREHENSIVE API CLIENT WITH RETRY

### Enhanced API Client
```python
import requests
from typing import Dict, Any, Optional

class ResilientAPIClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.circuit_breaker = CircuitBreaker()
        self.retry_config = APIRetryConfig()
    
    @api_retry()
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with circuit breaker protection"""
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        def request_func():
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
            )
            
            # Check for HTTP errors
            if response.status_code in APIErrorHandler.RETRYABLE_ERRORS:
                raise requests.HTTPError(f"HTTP {response.status_code}: {response.text}")
            
            response.raise_for_status()
            return response
        
        return self.circuit_breaker.call(request_func)
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """GET request with retry logic"""
        response = self._make_request("GET", endpoint, params=params)
        return response.json()
    
    def post(self, endpoint: str, data: Optional[Dict] = None, json: Optional[Dict] = None) -> Dict[str, Any]:
        """POST request with retry logic"""
        response = self._make_request("POST", endpoint, data=data, json=json)
        return response.json()
    
    def put(self, endpoint: str, data: Optional[Dict] = None, json: Optional[Dict] = None) -> Dict[str, Any]:
        """PUT request with retry logic"""
        response = self._make_request("PUT", endpoint, data=data, json=json)
        return response.json()
    
    def delete(self, endpoint: str) -> Dict[str, Any]:
        """DELETE request with retry logic"""
        response = self._make_request("DELETE", endpoint)
        return response.json() if response.content else {}
```

## 5. USAGE EXAMPLES

### Basic Usage
```python
# Initialize client
api_client = ResilientAPIClient("https://api.example.com")

# Configure custom retry settings
custom_config = APIRetryConfig()
custom_config.max_retries = 5
custom_config.base_delay = 2.0
api_client.retry_config = custom_config

# Make API calls with automatic retry
try:
    data = api_client.get("/weather/forecast", params={"city": "Selangor"})
    print("API call successful:", data)
except Exception as e:
    print(f"API call failed after all retries: {e}")
```

### Advanced Usage with Custom Error Handling
```python
@api_retry(APIRetryConfig())
def fetch_rainfall_data(location: str) -> Dict[str, Any]:
    """Fetch rainfall data with automatic retry"""
    try:
        response = requests.get(
            f"https://weather-api.com/rainfall/{location}",
            timeout=30
        )
        
        if response.status_code == 429:
            raise requests.HTTPError("Rate limit exceeded")
        
        response.raise_for_status()
        return response.json()
        
    except requests.ConnectionError as e:
        print(f"Connection error: {e}")
        raise
    except requests.Timeout as e:
        print(f"Request timeout: {e}")
        raise

# Usage
try:
    rainfall_data = fetch_rainfall_data("selangor")
    print("Rainfall data retrieved successfully")
except Exception as e:
    print(f"Failed to fetch rainfall data: {e}")
```

## 6. MONITORING AND LOGGING

### Request Metrics
```python
import logging
from datetime import datetime

class APIMetrics:
    def __init__(self):
        self.total_requests = 0
        self.failed_requests = 0
        self.retry_attempts = 0
        self.circuit_breaker_trips = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def log_request(self, endpoint: str, method: str, status_code: int, duration: float):
        """Log API request metrics"""
        self.total_requests += 1
        
        if status_code >= 400:
            self.failed_requests += 1
        
        self.logger.info(
            f"API Request: {method} {endpoint} - "
            f"Status: {status_code} - Duration: {duration:.2f}s"
        )
    
    def log_retry(self, attempt: int, delay: float):
        """Log retry attempt"""
        self.retry_attempts += 1
        self.logger.warning(f"Retry attempt {attempt} after {delay:.2f}s delay")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics"""
        success_rate = ((self.total_requests - self.failed_requests) / 
                       max(self.total_requests, 1)) * 100
        
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": f"{success_rate:.2f}%",
            "retry_attempts": self.retry_attempts,
            "circuit_breaker_trips": self.circuit_breaker_trips
        }
```

## 7. CONFIGURATION FOR DIFFERENT ENVIRONMENTS

### Environment-Specific Settings
```yaml
# config/api_retry.yaml
development:
  max_retries: 2
  base_delay: 0.5
  max_delay: 10.0
  circuit_breaker_threshold: 3
  circuit_breaker_timeout: 30

production:
  max_retries: 5
  base_delay: 1.0
  max_delay: 60.0
  circuit_breaker_threshold: 10
  circuit_breaker_timeout: 120

testing:
  max_retries: 1
  base_delay: 0.1
  max_delay: 1.0
  circuit_breaker_threshold: 2
  circuit_breaker_timeout: 5
```

## 8. BEST PRACTICES

### Implementation Guidelines
1. **Exponential Backoff**: Use exponential backoff with jitter to prevent thundering herd problems
2. **Circuit Breaker**: Implement circuit breaker pattern for failing services
3. **Error Classification**: Distinguish between retryable and non-retryable errors
4. **Timeout Management**: Set appropriate timeouts for different types of requests
5. **Monitoring**: Log retry attempts and circuit breaker state changes
6. **Rate Limiting**: Respect API rate limits and implement client-side rate limiting
7. **Graceful Degradation**: Provide fallback mechanisms when all retries fail

### Error Recovery Strategies
```python
class APIFallbackHandler:
    def __init__(self):
        self.cache = {}
        self.fallback_data = {}
    
    def get_with_fallback(self, api_client: ResilientAPIClient, endpoint: str, fallback_key: str):
        """Get data with fallback to cached or default values"""
        try:
            # Try API call
            data = api_client.get(endpoint)
            self.cache[fallback_key] = data  # Cache successful response
            return data
            
        except Exception as e:
            print(f"API call failed: {e}")
            
            # Try cached data
            if fallback_key in self.cache:
                print("Using cached data")
                return self.cache[fallback_key]
            
            # Use fallback data
            if fallback_key in self.fallback_data:
                print("Using fallback data")
                return self.fallback_data[fallback_key]
            
            # No fallback available
            raise Exception("No fallback data available")
```

This comprehensive API retry strategy provides robust error handling, intelligent retry mechanisms, and monitoring capabilities for production-ready applications.
