"""
A basic wrapper to run the pipeline using Prefect.
"""
from prefect import flow, task

from sotrplib.cli import parse_args
from sotrplib.config.config import Settings


@flow
def basic():
    args = parse_args()
    config = Settings.from_file(args.config)
    pipeline = config.to_basic()
    task(pipeline.run, name="basic")()
