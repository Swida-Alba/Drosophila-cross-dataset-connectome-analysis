"""
API Utilities Module

Provides timeout and retry functionality for API calls to prevent hanging
when network issues occur during batch fetching from NeuPrint or other APIs.

Usage:
    from src.utils.api_utils import api_call_with_retry, APITimeoutError

    # Wrap any API call with timeout and retry
    result = api_call_with_retry(
        lambda: client.fetch_custom(query),
        timeout=60,
        max_retries=5,
        description="Fetching connections"
    )
"""

import threading
import time
import warnings
from typing import Any, Callable, Optional, TypeVar

T = TypeVar('T')


class APITimeoutError(Exception):
    """Raised when an API call exceeds the timeout limit."""
    pass


class APIRetryExhaustedError(Exception):
    """Raised when all retry attempts for an API call have failed."""
    pass


class APICancelError(Exception):
    """Raised when a cancel_event is set while an API call is in flight.

    Callers that support cooperative cancellation (e.g. the Settings-tab
    pulls) pass ``cancel_event``; the retry loop then aborts within ~0.5 s
    instead of waiting out the current attempt's timeout or a backoff sleep.
    """
    pass


def _call_with_timeout(
    func: Callable[[], T],
    timeout: float,
    cancel_event: Optional[Any],
    description: str,
    attempt: int,
    max_retries: int,
) -> T:
    """Run *func* on a daemon thread and stop waiting after *timeout*.

    The worker must be a plain daemon thread, not a ThreadPoolExecutor
    worker: since Python 3.9 concurrent.futures joins every pool worker at
    interpreter exit.  A request that never returns (a half-open socket with
    no server-side timeout) would then keep a DROCAT backend alive after its
    final "[DROCAT] Done." line — the UI runner waits for the process to
    exit, so the run would sit in "Running" forever and never write its run
    guide or pair report.  A daemon thread is abandoned at shutdown instead.
    """
    finished = threading.Event()
    outcome: dict = {}

    def worker() -> None:
        try:
            outcome["result"] = func()
        except BaseException as exc:  # re-raised in the calling thread
            outcome["error"] = exc
        finally:
            finished.set()

    threading.Thread(
        target=worker, name="drocat-api-call", daemon=True
    ).start()
    deadline = time.monotonic() + timeout
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise APICancelError(
                f"{description} cancelled while waiting for the API"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise APITimeoutError(
                f"{description} timed out after {timeout}s "
                f"(attempt {attempt}/{max_retries})"
            )
        # Wake immediately on completion; poll cancel_event every <=0.5 s.
        if finished.wait(min(0.5, remaining)):
            break
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def api_call_with_retry(
    func: Callable[[], T],
    timeout: float = 60.0,
    max_retries: int = 5,
    retry_delay: float = 2.0,
    description: str = "API call",
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    verbose: bool = True,
    cancel_event: Optional[Any] = None
) -> T:
    """
    Execute an API call with timeout and retry logic.
    
    This function wraps an API call to:
    1. Enforce a timeout to prevent hanging
    2. Retry on failure with exponential backoff
    3. Provide informative error messages
    
    Args:
        func: A callable (lambda or function) that performs the API call.
              Should take no arguments and return the result.
        timeout: Maximum time in seconds to wait for each attempt (default: 60)
        max_retries: Maximum number of retry attempts (default: 5)
        retry_delay: Initial delay between retries in seconds (default: 2.0)
                     Uses exponential backoff: delay * 2^(attempt-1)
        description: Human-readable description for error messages
        on_retry: Optional callback(attempt, exception) called before each retry
        verbose: If True, print retry warnings
    
    Returns:
        The result of the API call
    
    Raises:
        APITimeoutError: If all attempts timeout
        APIRetryExhaustedError: If all retry attempts fail with other errors
    
    Example:
        >>> # Wrap a NeuPrint fetch with timeout
        >>> result = api_call_with_retry(
        ...     lambda: client.fetch_custom(query),
        ...     timeout=30,
        ...     max_retries=2,
        ...     description="Fetching neurons"
        ... )
        
        >>> # Wrap fetch_simple_connections
        >>> result = api_call_with_retry(
        ...     lambda: fetch_simple_connections(
        ...         upstream_criteria=criteria,
        ...         min_weight=1
        ...     ),
        ...     timeout=60,
        ...     description=f"Fetching batch {batch_idx}"
        ... )
    """
    last_exception = None

    def _interruptible_sleep(delay: float) -> None:
        """Sleep in small slices so a set cancel_event aborts the backoff."""
        if cancel_event is None:
            time.sleep(delay)
            return
        deadline = time.monotonic() + delay
        while True:
            if cancel_event.is_set():
                raise APICancelError(
                    f"{description} cancelled during retry backoff"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.5, remaining))

    for attempt in range(1, max_retries + 1):
        if cancel_event is not None and cancel_event.is_set():
            raise APICancelError(f"{description} cancelled")
        try:
            # The call is bounded by a daemon worker (see _call_with_timeout):
            # a hung API call must not block the retry loop, and abandoning it
            # must never block interpreter shutdown.
            return _call_with_timeout(
                func, timeout, cancel_event, description, attempt, max_retries)
        except APICancelError:
            raise
        except APITimeoutError as e:
            last_exception = e
            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                if verbose:
                    warnings.warn(
                        f"⚠️ {description} timed out (attempt {attempt}/{max_retries}). "
                        f"Retrying in {delay:.1f}s..."
                    )
                if on_retry:
                    on_retry(attempt, e)
                _interruptible_sleep(delay)
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))
                if verbose:
                    warnings.warn(
                        f"⚠️ {description} failed: {type(e).__name__}: {e} "
                        f"(attempt {attempt}/{max_retries}). Retrying in {delay:.1f}s..."
                    )
                if on_retry:
                    on_retry(attempt, e)
                _interruptible_sleep(delay)
    
    # All retries exhausted
    if isinstance(last_exception, APITimeoutError):
        raise last_exception
    else:
        raise APIRetryExhaustedError(
            f"{description} failed after {max_retries} attempts. "
            f"Last error: {type(last_exception).__name__}: {last_exception}"
        ) from last_exception


def escape_cypher_string(value: str) -> str:
    """
    Escape special characters in a string for use in Cypher queries.
    
    This handles:
    - Single quotes (') -> escaped as \\' 
    - Backslashes (\\) -> escaped as \\\\
    
    This is critical for neuron types containing special characters like:
    - KCa'b'-ap1 (Kenyon cell subtypes with apostrophes)
    
    Args:
        value: The string value to escape
    
    Returns:
        The escaped string safe for use in Cypher queries
    
    Example:
        >>> escape_cypher_string("KCa'b'-ap1")
        "KCa\\'b\\'-ap1"
        >>> # In a query:
        >>> query = f"MATCH (n:Neuron) WHERE n.type = '{escape_cypher_string(neuron_type)}'"
    """
    if not isinstance(value, str):
        return str(value)
    
    # Escape backslashes first (before they could be confused with escape sequences)
    value = value.replace('\\', '\\\\')
    # Escape single quotes
    value = value.replace("'", "\\'")
    
    return value


def build_cypher_type_condition(
    neuron: Any,
    alias: str = 'n',
    type_column: str = 'type'
) -> str:
    """
    Build a Cypher WHERE condition for matching neuron types safely.
    
    Handles:
    - String type names with special character escaping
    - Regex patterns (containing .* or *)
    - Integer bodyIds
    - Lists of bodyIds
    
    Args:
        neuron: The neuron identifier - type name (str), bodyId (int), or list of bodyIds
        alias: The Cypher node alias (default: 'n')
        type_column: The column name for type matching (default: 'type')
    
    Returns:
        A Cypher condition string (without WHERE keyword)
    
    Example:
        >>> build_cypher_type_condition("KCa'b'-ap1", alias='n')
        "n.type = 'KCa\\'b\\'-ap1'"
        >>> build_cypher_type_condition("KC.*", alias='n')
        "n.type =~ 'KC.*'"
        >>> build_cypher_type_condition(12345, alias='n')
        "n.bodyId = 12345"
    """
    if isinstance(neuron, str):
        # Type-based query with regex support
        escaped = escape_cypher_string(neuron)
        if '.*' in neuron or '*' in neuron:
            return f"{alias}.{type_column} =~ '{escaped}'"
        else:
            return f"{alias}.{type_column} = '{escaped}'"
    elif isinstance(neuron, int):
        return f"{alias}.bodyId = {neuron}"
    elif isinstance(neuron, list):
        bodyids_str = ', '.join(str(b) for b in neuron)
        return f"{alias}.bodyId IN [{bodyids_str}]"
    else:
        raise ValueError(f"Unsupported neuron type: {type(neuron)}")


# Convenience function for batch processing with progress
def process_batches_with_retry(
    batches: list,
    process_func: Callable[[Any], Any],
    timeout: float = 60.0,
    max_retries: int = 5,
    description_prefix: str = "Processing batch",
    show_progress: bool = True,
    verbose: bool = True
) -> list:
    """
    Process a list of batches with timeout and retry for each batch.
    
    Args:
        batches: List of batch items to process
        process_func: Function to call for each batch, takes batch as argument
        timeout: Timeout per batch in seconds
        max_retries: Max retries per batch (default: 5)
        description_prefix: Prefix for progress description
        show_progress: Show tqdm progress bar
        verbose: Print retry warnings
    
    Returns:
        List of results from successful batches (failed batches are skipped)
    
    Example:
        >>> results = process_batches_with_retry(
        ...     batches=[[1,2,3], [4,5,6]],
        ...     process_func=lambda batch: fetch_connections(batch),
        ...     timeout=30,
        ...     description_prefix="Fetching"
        ... )
    """
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, **kwargs):
            return iterable
    
    results = []
    failed_batches = []
    
    iterator = tqdm(
        enumerate(batches),
        total=len(batches),
        desc=f"{description_prefix}",
        disable=not show_progress
    ) if len(batches) > 1 else enumerate(batches)
    
    for idx, batch in iterator:
        try:
            result = api_call_with_retry(
                lambda b=batch: process_func(b),
                timeout=timeout,
                max_retries=max_retries,
                description=f"{description_prefix} {idx+1}/{len(batches)}",
                verbose=verbose
            )
            if result is not None:
                results.append(result)
        except (APITimeoutError, APIRetryExhaustedError) as e:
            if verbose:
                warnings.warn(f"⚠️ Batch {idx+1} failed permanently: {e}")
            failed_batches.append(idx)
    
    if failed_batches and verbose:
        warnings.warn(f"⚠️ {len(failed_batches)} batches failed: {failed_batches}")
    
    return results
