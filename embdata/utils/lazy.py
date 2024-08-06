from collections import deque
from functools import wraps
from importlib import import_module
from typing import Any, Callable


class LazyObject:
    def __init__(self, import_path: str):
        self._import_path = import_path
        self._real_object = None

    def __getattr__(self, name: str) -> Any:
        if self._real_object is None:
            self._real_object = import_module(self._import_path)
        return getattr(self._real_object, name)

def import_lazy(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        lazy_args = [LazyObject(arg) if isinstance(arg, str) else arg for arg in args]
        lazy_kwargs = {k: LazyObject(v) if isinstance(v, str) else v for k, v in kwargs.items()}
        return func(*lazy_args, **lazy_kwargs)
    return wrapper

@import_lazy
@wraps(import_module)
def lazy_import(name: str, package: str | None = None) -> Any:
    try:
        import_module(name, package)
    except ImportError:
        return LazyObject(name)


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
