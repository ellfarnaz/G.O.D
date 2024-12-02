import queue
import threading
from uuid import UUID
from datetime import datetime
import time

import docker
from fiber.logging_utils import get_logger

from core.models.utility_models import Job
from core.models.utility_models import JobStatus
from miner.logic.job_handler import start_tuning_container


logger = get_logger(__name__)


class TrainingWorker:
    def __init__(self):
        logger.info("=" * 80)
        logger.info("STARTING A TRAINING WORKER")
        logger.info("=" * 80)

        self.job_queue: queue.Queue[Job] = queue.Queue()
        self.job_store: dict[str, Job] = {}
        self.active_jobs: dict[str, dict] = {}
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.docker_client = docker.from_env()

    def _worker(self):
        while True:
            job = self.job_queue.get()
            if job is None:
                break
            try:
                gpu_memory = self._get_gpu_memory_usage()
                if gpu_memory > 75.0:
                    logger.warning(f"High GPU memory usage: {gpu_memory}%")
                
                self.active_jobs[job.job_id] = {
                    'start_time': datetime.now(),
                    'gpu_usage': [],
                    'memory_usage': []
                }
                
                start_tuning_container(job)
                job.status = JobStatus.COMPLETED
                
            except Exception as e:
                logger.error(f"Error processing job {job.job_id}: {str(e)}")
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                
                if "CUDA out of memory" in str(e):
                    logger.info("CUDA OOM detected, waiting for memory cleanup...")
                    time.sleep(300)
                    self.job_queue.put(job)
                    
            finally:
                if job.job_id in self.active_jobs:
                    del self.active_jobs[job.job_id]
                self.job_queue.task_done()

    def _get_gpu_memory_usage(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return (info.used / info.total) * 100
        except:
            return 0.0

    def enqueue_job(self, job: Job):
        self.job_queue.put(job)
        self.job_store[job.job_id] = job

    def get_status(self, job_id: UUID) -> JobStatus:
        job = self.job_store.get(str(job_id))

        return job.status if job else JobStatus.NOT_FOUND

    def shutdown(self):
        self.thread.join()
        self.docker_client.close()
