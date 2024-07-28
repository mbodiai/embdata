from collections import deque
from functools import wraps


class LazyCall:
    """A class that allows queuing and applying function calls with their respective arguments."""

    def __init__(self):
        """Initializes a new instance of the LazyCall class."""
        self.function_calls = deque()
        self.kwargs = deque()

    def add_call(self, function, instance, *args, **kwargs) -> None:
        """Adds a function call to the queue with the specified arguments.

        Parameters:
        function (callable): The function to be called.
        instance (object): The instance the function is called on.
        *args: Positional arguments to be passed to the function.
        **kwargs: Keyword arguments to be passed to the function.
        """
        self.function_calls.append((function, instance, args, kwargs))

    def apply(self) -> None:
        """Applies the queued function calls with their respective arguments."""
        while self.function_calls:
            function, instance, args, kwargs = self.function_calls.popleft()
            function(instance, *args, **kwargs)

    def __call__(self, function):
        """Decorator to add a function call to the queue with the specified arguments.

        Parameters:
        function (callable): The function to be called.

        Returns:
        callable: The wrapped function.
        """

        @wraps(function)
        def wrapper(instance, *args, **kwargs):
            instance.lazy_call.add_call(function, instance, *args, **kwargs)

        return wrapper
