import logging
import dotenv
from queue import Queue

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


def fetch_data_from_db(connection, offset_tracker):
    """Fetching thread function - 從資料庫獲取資料"""
    try:
        records = db.get_records_from_database(connection, offset_tracker['offset'])
        logging.debug(f"Fetched {len(records)} records from database at offset {offset_tracker['offset']}")
        if not records:
            return None
        offset_tracker['offset'] += len(records)
        return records
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None


def process_raw_data(rawdata_queue):
    """Processing thread function - 處理原始資料並提取特徵"""
    try:
        if rawdata_queue.empty():
            return None
        
        records = rawdata_queue.get(timeout=1)
        all_deltas = []
        
        for record_id, trail_data in records:
            try:
                # 解析 GPX 資料
                track_points = gpx_processor.parse_gpx(trail_data)
                if len(track_points) < 2:
                    continue
                
                # 計算增量
                deltas = gpx_processor.compute_deltas(track_points)
                all_deltas.extend(deltas)
                
            except Exception as e:
                logging.warning(f"Error processing record {record_id}: {e}")
                continue
        
        rawdata_queue.task_done()
        return all_deltas if all_deltas else None
    
    except Exception as e:
        logging.error(f"Error in processing: {e}")
        return None


def train_model(feature_queue, model_state):
    """Training thread function - 訓練機器學習模型"""
    try:
        if feature_queue.empty():
            return None
            
        deltas = feature_queue.get(timeout=1)
        if not deltas or len(deltas) < config.sequence_length:
            feature_queue.task_done()
            return None
        
        # 分割序列用於訓練
        X, y = gpx_processor.split_sequences(deltas, config.sequence_length)
        if not X or not y:
            feature_queue.task_done()
            return None
        
        # 獲取設備並訓練模型
        device = trainer.get_device()
        model = trainer.train(
            X, y,
            checkpoint_dir=config.checkpoint_dir,
            epochs=5,
            batch_size=config.batch_size,
            device=device
        )
        
        model_state['trained_models'] = model_state.get('trained_models', 0) + 1
        feature_queue.task_done()
        
        return f"Model trained successfully. Total models: {model_state['trained_models']}"
    
    except Exception as e:
        logging.error(f"Error in training: {e}")
        return None


def main() -> None:
    logger = create_logger()
    logger.info("Starting Gen-GPX application...")
    
    # 初始化資料庫連接
    conn_str = db.get_connection_str()
    if not conn_str:
        logger.error("Failed to get database connection string")
        return
    
    try:
        connection = db.connect_database(conn_str)
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        return
    
    # 建立佇列
    rawdata_queue = Queue(maxsize=config.queue_max_size)
    feature_queue = Queue(maxsize=config.queue_max_size)
    
    # 建立共享狀態
    offset_tracker = {'offset': 0}
    model_state = {'trained_models': 0}
    
    # 建立執行緒處理器
    thread_handler = thread.ThreadHandler(rawdata_queue, feature_queue)
    
    try:
        # 啟動執行緒
        thread_handler.start(
            producer_func=fetch_data_from_db,
            processor_func=process_raw_data,
            consumer_func=train_model,
            producer_args=(connection, offset_tracker),
            processor_args=(rawdata_queue,),
            consumer_args=(feature_queue, model_state)
        )
        
        logger.info("All threads started. Press Ctrl+C to stop...")
        
        # 主執行緒等待
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
    
    except Exception as e:
        logger.error(f"Error during execution: {e}")
    
    finally:
        # 停止執行緒
        thread_handler.stop()
        
        # 關閉資料庫連接
        if connection:
            connection.close()
            logger.info("Database connection closed")
        
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    dotenv.load_dotenv()
    main()
