from os.path import join, abspath
import yaml

PROJECT_ROOT = abspath(join(__file__, '../../..')) # yawning-detection/yawnn/
CONFIG_PATH = f"{PROJECT_ROOT}/yawnnlib/config.yaml"
file = open(CONFIG_PATH, "r")
config = yaml.load(file, Loader=yaml.FullLoader)

def get(key : str):
    result = config[key]
    if type(result) == str:
        # replace strings with {PROJECT_ROOT} with the actual path
        result = result.format(**globals())
    return result

if __name__ == "__main__":
    print(vars()["PROJECT_ROOT"])
    print(get("DATA_PATH"))