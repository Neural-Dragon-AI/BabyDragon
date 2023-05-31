multithreading
==============

.. code-block:: python

	
	
	class RateLimiter:
	    def __init__(self, calls_per_minute: int, verbose: bool = False):
	        self.calls_per_minute = calls_per_minute
	        self.interval = 60 / calls_per_minute
	        self.lock = Lock()
	        self.last_call_time = None
	        self.verbose = verbose
	
	    def __call__(self, func):
	        @functools.wraps(func)
	        def wrapper(*args, **kwargs):
	            with self.lock:
	                if self.last_call_time is not None:
	                    time_since_last_call = time.time() - self.last_call_time
	                    if time_since_last_call < self.interval:
	                        time_to_wait = self.interval - time_since_last_call
	                        if self.verbose:
	                            print(
	                                f"RateLimiter: Waiting for {time_to_wait:.2f} seconds before next call."
	                            )
	                        time.sleep(time_to_wait)
	                    elif self.verbose:
	                        print(
	                            f"RateLimiter: No wait required, time since last call: {time_since_last_call:.2f} seconds."
	                        )
	                else:
	                    if self.verbose:
	                        print("RateLimiter: This is the first call, no wait required.")
	                self.last_call_time = time.time()
	            return func(*args, **kwargs)
	
	        return wrapper
	

.. code-block:: python

	
	
	class RateLimitedThreadPoolExecutor(ThreadPoolExecutor):
	    def __init__(self, max_workers=None, *args, **kwargs):
	        super().__init__(max_workers)
	        self.rate_limiter = RateLimiter(kwargs.get("calls_per_minute", 20), kwargs.get("verbose", False))
	
	    def submit(self, fn, *args, **kwargs):
	        rate_limited_fn = self.rate_limiter(fn)
	        return super().submit(rate_limited_fn, *args, **kwargs)
	

.. automodule:: multithreading
   :members:
