from typing import Callable
from queue import Queue
import threading
import logging


logger = logging.getLogger(__name__)


class ThreadHandler:
    def __init__(self, max_queue_size: int = 5):
        self.raw_queue = Queue(maxsize=max_queue_size)
        self.processed_data_queue
        self.fetching_thread = None
        self.training_thread = None
        self.stop_event = threading.Event()

    def start(
            self,
            producer_func: Callable,
            processor_func: Callable,
            producer_args=(),
            processor_args=(),
            consumer_func: Callable | None = None,
            consumer_args=()
    ):
        self.stop_event.clear()

        def producer_wrapper():
            logger.info("Starting data fetching thread...")
            while not self.stop_event.is_set():
                data = producer_func(*producer_args)
                if data is None:
                    logger.info("No new data obtained. Stopping...")
                    break
                self.raw_data_queue.put(data)
            if self.stop_event.is_set():
                logger.warning("Stop event recieved")
            logger.info("Stopping data fetching thread...")

        def processor_wrapper():
            logger.info(f"Recieved ")

        def consumer_wrapper():
            logger.info("Starting training thread...")
            while not self.stop_event.is_set():
                try:
                    data = self.queue.get(timeout=1)
                    if consumer_func:
                        consumer_func(data, *consumer_args)
                    self.queue.task_done()
                except Exception:
                    continue
            if self.stop_event.is_set():
                logger.warning("Stop event received")
            logger.info("Stopping training thread...")

        self.fetching_thread = threading.Thread(
            target=producer_wrapper, daemon=True)
        self.training_thread = threading.Thread(
            target=consumer_wrapper, daemon=True)

        self.fetching_thread.start()
        self.training_thread.start()

    def stop(self):
        logger.info("Sending event...")
        self.stop_event.set()
        logger.info("Waiting for threads to stop")
        if not (self.fetching_thread and self.training_thread):
            return
        self.fetching_thread.join()
        self.training_thread.join()
        logger.info("All threads has been stopped...")
