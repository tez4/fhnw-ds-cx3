import time
import psutil
import GPUtil
from pathlib import Path


class SystemUsageLogger:
    def __init__(self, experiment_name):
        self.start_time = time.time()
        self.name = f'{experiment_name}_{self.start_time}'
        self.folder_path = f'./output/system_usage/{self.name}'

        directory = Path(self.folder_path)
        directory.mkdir(parents=True, exist_ok=True)

    def log(self):
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        with open(f"{self.folder_path}/cpu.txt", "a") as f:
            f.write(f"{time.time() - self.start_time}    {cpu_usage} {memory_usage}\n")

        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_usage = gpu.load * 100
            gpu_memory_usage = gpu.memoryUtil * 100
            gpu_name = gpu.name
            with open(f"{self.folder_path}/gpu.txt", "a") as f:
                f.write(f"{time.time() - self.start_time}    {gpu_name} {gpu_usage} {gpu_memory_usage}\n")


if __name__ == '__main__':
    system_usage_logger = SystemUsageLogger('test')
    for i in range(20):
        time.sleep(1)
        system_usage_logger.log()
