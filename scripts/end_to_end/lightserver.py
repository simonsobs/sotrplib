import asyncio

from lightcurvedb.cli.ephemeral import core as db
from uvicorn import Config, Server


async def run_servers():
    config1 = Config("lightgest.api:app", host="0.0.0.0", port=8001, log_level="info")
    config2 = Config("lightserve.api:app", host="0.0.0.0", port=8000, log_level="info")

    server1 = Server(config1)
    server2 = Server(config2)

    await asyncio.gather(
        server1.serve(),
        server2.serve(),
    )


if __name__ == "__main__":
    import os

    with db(backend_type=os.environ.get("LIGHTCURVEDB_BACKEND_TYPE"), number=0):
        if os.environ.get("LIGHTCURVEDB_BACKEND_TYPE") in ["postgres", "timescale"]:
            with open("./port", "w") as f:
                f.write(
                    str(os.environ.get("LIGHTCURVEDB_POSTGRES_PORT", "5432")) + "\n"
                )
        asyncio.run(run_servers())
