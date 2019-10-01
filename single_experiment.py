import json
import sys

def main():
    config = json.load(sys.argv[1])
    data_config = config["data"]
    parameters_config = config["parameters"]

if __name__ == '__main__':
    main()