from os.path import join, abspath
from typing import Any
import yaml

PROJECT_ROOT = abspath(join(__file__, '../../..')) # yawning-detection/yawnn/
CONFIG_PATH = f"{PROJECT_ROOT}/yawnnlib/config.yaml"
file = open(CONFIG_PATH, "r")
config = yaml.load(file, Loader=yaml.FullLoader)

def get(key : str) -> Any:
    result = config[key]
    if type(result) == str:
        # replace strings with {PROJECT_ROOT} with the actual path
        result = result.format(**globals())
    return result

def set(key : str, value : Any):
    # this should only be used in e.g. tests, where the user values in the config should be ignored
    config[key] = value

if __name__ == "__main__":
    print(vars()["PROJECT_ROOT"])
    print(get("DATA_PATH"))