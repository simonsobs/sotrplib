from sotrplib.config.config import Settings

# These are the default settings
print(Settings().model_dump())

# This is reading in from a file
MySettings = Settings.from_file("../sotrplib/config/test_config.json")

print(MySettings)

MySettings = Settings.from_file("../sotrplib/config/tiger3_actpol_config.json")

print(MySettings)
