"""
A basic pipeline handler. Takes all the components and runs
them in the pre-specified order.
"""

import tempfile
from typing import Any

from typing_extensions import Self, TypeVar

import pyinstrument

from .base import BaseRunner


T = TypeVar("T", infer_variance=True)


class _unmapped(tuple[T]):
    """
    Wrapper for iterables. Copied from :code:`prefect.utilities.annotations` to avoid
    hard dependency.

    Indicates that this input should be sent as-is to all runs created during a mapping
    operation instead of being split.
    """

    def __new__(cls, value: T) -> Self:
        return super().__new__(cls, (value,))

    @property
    def value(self) -> T:
        return self[0]

    def __getitem__(self, _: object) -> Any:
        return super().__getitem__(0)


class DummyPrefectTask:
    """A dummy Prefect task that does nothing, for when profiling is disabled."""

    def __init__(self, func: callable, name: str | None = None):
        self.func = func
        self.__name__ = name or getattr(func, "__name__", "dummy_task")

    def __call__(self, *args, **kwargs):
        """Execute the function directly."""
        return self.func(*args, **kwargs)

    def map(self, *args, **kwargs):
        """Map over inputs directly."""
        results = []
        map_axes = [
            ii for ii in range(len(args)) if not isinstance(args[ii], _unmapped)
        ]
        carry_args = [arg.value if isinstance(arg, _unmapped) else None for arg in args]
        for arg_set in zip(*[args[ii] for ii in map_axes]):
            for ii, arg in zip(map_axes, arg_set):
                carry_args[ii] = arg
            result = self.func(*carry_args, **kwargs)
            results.append(result)
        return DummyPrefectTaskResult(results)


class DummyPrefectTaskResult:
    """A dummy Prefect task result that mimics Prefect task result interface."""

    def __init__(self, results: list):
        self.results = results

    def result(self):
        """Return the results."""
        return self.results

    def wait(self):
        """No-op wait method for compatibility."""
        pass


def dummy_task(func: callable, name: str | None = None) -> callable:
    """A decorator to turn a function into a dummy Prefect task."""
    return DummyPrefectTask(func, name=name)


def dummy_profile_task(func: callable) -> callable:
    """
    A decorator to add pyinstrument profiling to a function.

    The profiling results will be saved to a temporary HTML file, and a link
    to the file will be added as an artifact in the Prefect UI along with a
    basic markdown summary.

    Profiling is enabled by setting the environment variable

    .. code-block:: shell

        export sotrplib_profile=1

    Notes
    =====

    Currently this writes the results to a permanent tempfile. This should
    probably be replaced with some configurable location.

    For some reason, the link doesn't open properly directly from the Prefect
    UI, but copying the link and pasting it into a new browser tab works fine.
    """

    def wrapped(*args, **kwargs):
        with pyinstrument.Profiler() as prof:
            result = func(*args, **kwargs)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            prof_path = f.name
        prof.write_html(prof_path, timeline=True)
        print(f"Profile results written to {prof_path}")
        return result

    return dummy_task(wrapped, name=func.__name__)


class PipelineRunner(BaseRunner):
    @property
    def profilable_task(self):
        if self.profile:
            return dummy_profile_task
        else:
            return dummy_task

    @property
    def basic_task(self):
        return dummy_task

    @property
    def flow(self):
        return lambda f: f

    @property
    def unmapped(self):
        return _unmapped
