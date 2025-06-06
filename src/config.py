import configparser
import os


config_path = os.path.join(os.path.dirname(__file__), "..", "config.ini")

config = configparser.ConfigParser()
config.read(config_path)

sequence_length = int(config["DEFAULT"].get("sequence_length", 100))
query_limit = int(config["DEFAULT"].get("query_limit", 1000))
batch_size = int(config["DEFAULT"].get("batch_size", 1000))
queue_max_size = int(config["DEFAULT"].get("queue_max_size", 1000))
input_size = int(config["DEFAULT"].get("input_size", 6))
hidden_size = int(config["DEFAULT"].get("hidden_size", 64))
num_layers = int(config["DEFAULT"].get("num_layers", 16))
checkpoint_dir = config["DEFAULT"].get("checkpoint_dir", "./checkpoint")
