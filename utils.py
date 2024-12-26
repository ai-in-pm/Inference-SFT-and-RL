import time
import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type variable for generic return type
T = TypeVar('T')

class RateLimiter:
    """Rate limiter implementation using token bucket algorithm."""
    
    def __init__(self, tokens_per_second: float):
        self.tokens_per_second = tokens_per_second
        self.tokens = tokens_per_second
        self.last_update = time.time()
    
    def acquire(self) -> float:
        """Acquire a token and return the wait time if needed."""
        now = time.time()
        time_passed = now - self.last_update
        self.tokens = min(
            self.tokens_per_second,
            self.tokens + time_passed * self.tokens_per_second
        )
        self.last_update = now
        
        if self.tokens < 1:
            wait_time = (1 - self.tokens) / self.tokens_per_second
            time.sleep(wait_time)
            self.tokens = 0
            return wait_time
        else:
            self.tokens -= 1
            return 0

class RetryWithBackoff:
    """Implements exponential backoff retry logic."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1,
        max_delay: float = 30,
        exponential_base: float = 2
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == self.max_retries:
                        logger.error(f"Max retries ({self.max_retries}) exceeded")
                        raise
                    
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
            
            raise last_exception  # Should never reach here
        
        return wrapper

class APIError(Exception):
    """Base class for API-related errors."""
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        provider: Optional[str] = None
    ):
        self.status_code = status_code
        self.provider = provider
        super().__init__(message)

class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    pass

class AuthenticationError(APIError):
    """Raised when API authentication fails."""
    pass

class QuotaExceededError(APIError):
    """Raised when API quota is exceeded."""
    pass

def validate_api_response(
    response: Dict[str, Any],
    provider: str
) -> None:
    """Validate API response and raise appropriate errors."""
    
    # Common error patterns for different providers
    error_patterns = {
        'openai': {
            'rate_limit': ['rate limit', 'too many requests'],
            'auth': ['authentication', 'invalid api key'],
            'quota': ['quota exceeded', 'insufficient_quota']
        },
        'anthropic': {
            'rate_limit': ['rate_limit_error'],
            'auth': ['invalid_api_key'],
            'quota': ['quota_exceeded']
        },
        'google': {
            'rate_limit': ['quota_exceeded', 'resource_exhausted'],
            'auth': ['invalid_key', 'unauthenticated'],
            'quota': ['billing_required', 'billing_disabled']
        }
    }
    
    if 'error' in response:
        error_msg = str(response['error']).lower()
        patterns = error_patterns.get(provider.lower(), {})
        
        # Check for rate limiting
        if any(pattern in error_msg for pattern in patterns.get('rate_limit', [])):
            raise RateLimitError(
                f"Rate limit exceeded for {provider}",
                status_code=429,
                provider=provider
            )
        
        # Check for authentication
        if any(pattern in error_msg for pattern in patterns.get('auth', [])):
            raise AuthenticationError(
                f"Authentication failed for {provider}",
                status_code=401,
                provider=provider
            )
        
        # Check for quota
        if any(pattern in error_msg for pattern in patterns.get('quota', [])):
            raise QuotaExceededError(
                f"API quota exceeded for {provider}",
                status_code=429,
                provider=provider
            )
        
        # Generic API error
        raise APIError(
            f"API error from {provider}: {response['error']}",
            provider=provider
        )
