import sys

from jale.ale import run_ale


def cli_ale():
    if len(sys.argv) == 2:
        yaml_path = sys.argv[1]
    else:
        raise ValueError("No input file provided. Please specify an input file.")
    run_ale(yaml_path=yaml_path)


if __name__ == "__main__":
    cli_ale()
