import tempfile
from datetime import datetime

import pyinstrument
from prefect import artifacts, flow, task

from .base import BaseRunner

__all__ = ["PrefectRunner", "profile_task"]

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


class PrefectRunner(BaseRunner):
    @property
    def profilable_task(self):
        if self.profile:
            return profile_task
        else:
            return task

    @property
    def basic_task(self):
        return task

    @property
    def flow(self):
        return flow
