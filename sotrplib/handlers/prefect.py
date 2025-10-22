from prefect import flow, task

from ..utils.prefect import profile_task
from .base import BaseRunner


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
