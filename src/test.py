from config import get_config
import dotenv
import db


def main():
    dotenv.load_dotenv()
    config = get_config()
    print(db.get_records_from_database(db.connect_database(
        db.get_connection_str()), config.query_limit, offset=0))


main()
