import tempfile
from datetime import datetime

import pyinstrument
from prefect import artifacts, task, unmapped

markdown_template = """# Profiling Results for {name}

`{func}`

The full results for this task can be found [here]({link}).

## Summary

| <!-- -->    | <!-- -->    |
|:-----------:|:-----------:|
| Duration | {duration:.4f}s |
| Sample count | {sample_count} |
| CPU time | {cpu_time:.4f}s |
| Start time | {start_time} |
"""


class DummyPrefectTask:
    """A dummy Prefect task that does nothing, for when profiling is disabled."""

    def __init__(self, func: callable, name: str | None = None):
        self.func = func
        self.__name__ = name or getattr(func, '__name__', 'dummy_task')

    def __call__(self, *args, **kwargs):
        """Execute the function directly."""
        return self.func(*args, **kwargs)

    def map(self, *args, **kwargs):
        """Map over inputs directly."""
        results = []
        map_axes = [ii for ii in range(len(args)) if not isinstance(args[ii], unmapped)]
        carry_args = [arg.value if isinstance(arg, unmapped) else None for arg in args]
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


def profile_task(func: callable) -> callable:
    """
    A decorator to turn a function into a Prefect task with profiling enabled.

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
        session = prof.last_session
        markdown = markdown_template.format(
            name=func.__name__.replace("_", " ").title(),
            func=f"{func.__module__}:{func.__code__.co_firstlineno}",
            link=f"file:///{prof_path}",
            duration=session.duration,
            sample_count=session.sample_count,
            cpu_time=session.cpu_time,
            start_time=datetime.fromtimestamp(session.start_time),
        )
        artifacts.create_markdown_artifact(
            markdown=markdown,
            key=f"{func.__name__}-profile".replace("_", "-"),
        )
        return result

    return task(wrapped, name=func.__name__)
