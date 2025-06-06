import logging
import dotenv

import gpx_processor
import trainer
import thread
import config
import db
import os


def create_logger() -> logging.Logger:
    os.makedirs("./logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s, [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("./logs/recent.log"),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    return logger


def main() -> None:
    return


if __name__ == "__main__":
    dotenv.load_dotenv()
    main()
