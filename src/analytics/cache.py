"""High-Performance Multi-Tier Cache Engine for Heavy ML Inference (SHAP, PDP & Predictions).

Features:
1. Tier 1: Thread-safe in-memory LRU (Least Recently Used) cache with TTL expiration.
2. Tier 2: Resilient Redis client with transparent fallback to in-memory cache when Redis is unavailable.
3. Cryptographic deterministic hashing (SHA-256) for complex Python objects and Pandas structures.
4. Telemetry metrics: hits, misses, hit ratio %, and memory eviction tracking.
"""

import functools
import hashlib
import json
import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def make_cache_key(namespace: str, *args, **kwargs) -> str:
    """Generate deterministic SHA-256 hash key from function arguments."""
    hasher = hashlib.sha256()
    hasher.update(namespace.encode("utf-8"))

    # Serialize args
    for arg in args:
        if hasattr(arg, "to_dict"):
            serialized = json.dumps(arg.to_dict(), sort_keys=True, default=str)
        elif hasattr(arg, "to_json"):
            serialized = str(arg.to_json())
        elif isinstance(arg, (dict, list, tuple, str, int, float, bool)) or arg is None:
            serialized = json.dumps(arg, sort_keys=True, default=str)
        elif hasattr(arg, "__class__"):
            # Normalize class instances (e.g. self) to class name to ignore memory address
            serialized = f"class:{arg.__class__.__qualname__}"
        else:
            serialized = str(arg)
        hasher.update(serialized.encode("utf-8"))

    # Serialize kwargs
    if kwargs:
        serialized_kw = json.dumps(kwargs, sort_keys=True, default=str)
        hasher.update(serialized_kw.encode("utf-8"))

    return f"{namespace}:{hasher.hexdigest()}"


class MLInferenceCache:
    """Thread-safe Multi-Tier LRU & Redis Cache Manager for SupTech inference."""

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl_seconds: int = 3600,
        redis_url: Optional[str] = None,
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl_seconds
        self.lock = threading.RLock()
        self._memory_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        # Telemetry metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        # Optional Redis connector
        self.redis_client = None
        self.redis_enabled = False
        target_redis = redis_url or os.getenv("REDIS_URL")

        if target_redis:
            try:
                import redis
                self.redis_client = redis.from_url(target_redis, socket_timeout=2)
                self.redis_client.ping()
                self.redis_enabled = True
                logger.info(f"Redis Tier-2 cache connected successfully at {target_redis}")
            except Exception as e:
                logger.warning(f"Redis unavailable, falling back to In-Memory LRU Tier-1: {e}")
                self.redis_enabled = False

    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached object from Tier 1 (Memory) or Tier 2 (Redis)."""
        current_time = time.time()

        # 1. Check Tier 1 (Memory LRU)
        with self.lock:
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                if current_time < entry["expires_at"]:
                    # Cache Hit (Memory)
                    self._memory_cache.move_to_end(key)
                    self.hits += 1
                    return entry["value"]
                else:
                    # Expired entry
                    del self._memory_cache[key]

        # 2. Check Tier 2 (Redis) if enabled
        if self.redis_enabled and self.redis_client:
            try:
                cached_bytes = self.redis_client.get(key)
                if cached_bytes:
                    value = json.loads(cached_bytes.decode("utf-8"))
                    # Populate back to Tier 1
                    self.set(key, value, ttl_seconds=self.default_ttl, propagate_to_redis=False)
                    with self.lock:
                        self.hits += 1
                    return value
            except Exception as e:
                logger.warning(f"Error reading from Redis cache: {e}")

        with self.lock:
            self.misses += 1
        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        propagate_to_redis: bool = True,
    ) -> None:
        """Store value in Tier 1 (Memory) and Tier 2 (Redis)."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expires_at = time.time() + ttl

        # 1. Set in Tier 1 (Memory LRU)
        with self.lock:
            if key in self._memory_cache:
                self._memory_cache.move_to_end(key)
            self._memory_cache[key] = {
                "value": value,
                "expires_at": expires_at,
                "created_at": time.time(),
            }

            # Enforce max size limit (LRU eviction)
            if len(self._memory_cache) > self.max_size:
                self._memory_cache.popitem(last=False)
                self.evictions += 1

        # 2. Set in Tier 2 (Redis)
        if propagate_to_redis and self.redis_enabled and self.redis_client:
            try:
                serialized = json.dumps(value, default=str)
                self.redis_client.setex(key, ttl, serialized)
            except Exception as e:
                logger.warning(f"Error writing to Redis cache: {e}")

    def clear(self) -> None:
        """Purge all memory and Redis cache keys."""
        with self.lock:
            self._memory_cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0

        if self.redis_enabled and self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Error clearing Redis cache: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Return real-time cache performance and telemetry metrics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_ratio = (self.hits / total_requests * 100) if total_requests > 0 else 0.0
            return {
                "tier1_memory_items": len(self._memory_cache),
                "tier1_max_size": self.max_size,
                "tier2_redis_enabled": self.redis_enabled,
                "hits": self.hits,
                "misses": self.misses,
                "total_requests": total_requests,
                "hit_ratio_pct": round(hit_ratio, 2),
                "evictions": self.evictions,
                "status": "OPERATIONAL",
            }


# Global Singleton Cache Instance
inference_cache = MLInferenceCache()


def cached_inference(namespace: str, ttl_seconds: int = 3600):
    """Decorator to automatically cache heavy ML inference functions."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = make_cache_key(namespace, *args, **kwargs)
            cached_val = inference_cache.get(key)
            if cached_val is not None:
                return cached_val

            # Compute and cache result
            result = func(*args, **kwargs)
            inference_cache.set(key, result, ttl_seconds=ttl_seconds)
            return result
        return wrapper
    return decorator
