import os
import tempfile

import pyinstrument
from prefect import artifacts, task


def profile_task(func: callable) -> callable:
    """
    A decorator to turn a function into a Prefect task with profiling enabled.

    The profiling results will be saved to a temporary HTML file, and a link
    to the file will be added as an artifact in the Prefect UI.

    Profiling is enabled by setting the environment variable

    .. code-block:: shell

        export SOTRPLIB_PROFILE=1

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
        artifacts.create_link_artifact(
            link=f"file:///{prof_path}",
            link_text="Profiling Results",
            key=f"{func.__name__}-profile".replace("_", "-"),
        )
        return result

    if os.environ.get("SOTRPLIB_PROFILE", "0") == "1":
        return task(wrapped, name=func.__name__)
    else:
        return task(func)
