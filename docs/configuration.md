# Configuration

Each command reads a JSON config file. A Pydantic Settings model in the
library reads the file and makes the library objects:

| Command | Model | Code |
|---|---|---|
| `sotrp` | `Settings` | `sotrplib/config/config.py` |
| `sotrp-coadd` | `CoaddSettings` | `sotrplib/config/coadd.py` |

The models for each part of the pipeline (maps, preprocessors, forced
photometry and others) are in `sotrplib/config/`. Each model has a method
(for example, `to_generator()` or `to_preprocessor()`) that makes the library
object. If you use the library directly, you can make these objects without
a config.

The configuration can change. More information will be added when it is
stable.
