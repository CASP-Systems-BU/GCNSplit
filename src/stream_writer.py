import csv
from queue import Empty

from torch.multiprocessing import Queue


def write_batches_from_queue_to_file(queue: Queue, file_path, throughput_report):
    time_file = open(file_path + "_TIME", "a")
    i = 0
    time_sum = 0
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)
        while True:
            try:
                batch = queue.get(block=True, timeout=60)
                writer.writerows(batch[0])
                i += 1
                time_sum += batch[1]
                if i % throughput_report == 0:
                    # time_file.write(str(batch[2]) + "," + str(batch[1]) + "\n")
                    time_file.write(str(time_sum) + "\n")
                    time_sum = 0
            except Empty:
                print("Timeout during reading from WRITING queue.")
                print(file_path)
                return
    pass
