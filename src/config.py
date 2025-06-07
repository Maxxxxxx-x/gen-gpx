import configparser
import os


class Config():
    def __init__(self, parser: configparser.RawConfigParser, path: str):
        parser.read(path)
        self.sequence_length = int(
            parser["DEFAULT"].get("sequence_length", 100))
        self.query_limit = int(parser["DEFAULT"].get("query_limit", 1000))
        self.batch_size = int(parser["DEFAULT"].get("batch_size", 1000))
        self.queue_max_size = int(
            parser["DEFAULT"].get("queue_max_size", 1000))
        self.input_size = int(parser["DEFAULT"].get("input_size", 6))
        self.hidden_size = int(parser["DEFAULT"].get("hidden_size", 64))
        self.num_layers = int(parser["DEFAULT"].get("num_layers", 16))
        self.checkpoint_dir = parser["DEFAULT"].get(
            "checkpoint_dir", "./checkpoint")


def get_config() -> Config:
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.ini")
    return Config(configparser.ConfigParser(), config_path)
