import os
import click
import configparser
from langtest import cli


@cli.group("config")
def config():
    pass


@config.command("set")
@click.argument(
    "args", nargs=-1, required=True
)  # nargs=-1 means all arguments are passed as a tuple
def set_env(args):
    try:
        # extract key and value from name by splitting on "="
        for key_value in args:
            key, value = key_value.split("=")
            write_config(key, value)
    except ValueError:
        print("Please provide key and value as key=value")


@config.command("get")
@click.argument("args", nargs=1, required=True)
def get_env(args):
    print(read_config(args))


config_path = os.path.dirname(os.path.abspath(__file__)) + "/config.ini"


def read_config(option, section="Default"):
    """Reads a value from a configuration file.

    Args:
        filename (str): The name of the configuration file.
        section (str): The section of the configuration file.
        option (str): The option to read from the section.

    Returns:
        str: The value of the option, or None if not found.
    """

    config = configparser.ConfigParser()
    config.read(config_path)
    return config.get(section, option, fallback=None)


def write_config(option, value, section="Default"):
    """Writes a value to a configuration file.

    Args:
        filename (str): The name of the configuration file.
        section (str): The section to write to.
        option (str): The option to write.
        value (str): The value to write.
    """

    config = configparser.ConfigParser()
    config.read(config_path)
    if not config.has_section(section):
        config.add_section(section)
    config.set(section, option, value)
    with open(config_path, "w") as configfile:
        config.write(configfile)

    print(f"Set {option} to {value}")


def update_config(option, value):
    """Updates a value in a configuration file.

    Args:
        filename (str): The name of the configuration file.
        section (str): The section to update.
        option (str): The option to update.
        value (str): The new value to set.
    """

    # If the section or option doesn't exist, create them using write_config
    if not read_config(option):
        write_config(option, value)
    else:
        write_config(option, value)  # Overwrite existing value
