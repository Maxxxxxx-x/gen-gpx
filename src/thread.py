from typing import Callable
from queue import Queue
import threading
import logging


logger = logging.getLogger(__name__)


class ThreadHandler:
    def __init__(self, rawdata_queue: Queue = None, feature_queue: Queue = None):
        self.stop_producer_event = threading.Event()
        self.stop_processor_event = threading.Event()
        self.stop_consumer_event = threading.Event()
        self.fetching_thread = None
        self.processing_thread = None
        self.training_thread = None
        self.rawdata_queue = rawdata_queue
        self.feature_queue = feature_queue

    def start(
            self,
            producer_func: Callable,
            processor_func: Callable,
            consumer_func: Callable,
            producer_args=(),
            processor_args=(),
            consumer_args=()
    ):
        self.stop_producer_event.clear()
        self.stop_processor_event.clear()
        self.stop_consumer_event.clear()

        def producer_wrapper():
            logger.info("Starting data fetching thread...")
            while not self.stop_producer_event.is_set():
                data = producer_func(*producer_args)
                if data is None:
                    continue
                for data_item in data:
                    self.rawdata_queue.put(data_item)
            logger.info("Stopping data fetching thread...")

        def processor_wrapper():
            logger.info("Starting data processing thread...")
            while not self.stop_processor_event.is_set():
                feature = processor_func(processor_args)
                if feature is None:
                    continue
                self.feature_queue.put(feature)
            logger.info("Stopping data processing thread...")

        def consumer_wrapper():
            logger.info("Starting training thread...")
            while not self.stop_consumer_event.is_set():
                result = consumer_func(*consumer_args)
                if result is None:
                    continue
                logger.info(f"Training result: {result}")
            logger.info("Stopping training thread...")

        self.fetching_thread = threading.Thread(
            target=producer_wrapper, daemon=True)
        self.processing_thread = threading.Thread(
            target=processor_wrapper, daemon=True)
        self.training_thread = threading.Thread(
            target=consumer_wrapper, daemon=True)

        self.fetching_thread.start()
        self.processing_thread.start()
        self.training_thread.start()
        logger.info("All threads started successfully.")
        return self

    def stop(self):
        logger.info("Stopping all threads...")
        self.stop_producer_event.set()
        self.stop_processor_event.set()
        self.stop_consumer_event.set()
        if self.fetching_thread:
            self.fetching_thread.join()
        if self.processing_thread:
            self.processing_thread.join()
        if self.training_thread:
            self.training_thread.join()
        logger.info("All threads stopped successfully.")
        return self
