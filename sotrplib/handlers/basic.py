"""
A basic pipeline handler. Takes all the components and runs
them in the pre-specified order.
"""

from ..utils.prefect import dummy_profile_task, dummy_task
from .base import BaseRunner


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
