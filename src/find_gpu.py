#!python3

import time
from subprocess import Popen, PIPE


def get_used_gpus(output):
    gpus = []
    in_processes = False
    for line in output:
        if "Processes:" in line:
            in_processes = True
            continue

        if in_processes is True:
            try:
                gpu = int(line.strip("|").split()[0])
                gpus.append(gpu)
            except:
                continue

    return set(sorted(gpus))


def get_gpu_id(cmd="nvidia-smi"):
    cmd=[cmd]

    tries = 200
    wait_time = 1

    start_time = time.perf_counter()

    for t in range(tries):

        proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
        out, err = proc.communicate()

        out = out.decode().split('\n')

        gpu_use_list = get_used_gpus(out)

        for i in range(16):
            if i not in gpu_use_list:
                # i seem to have good luch with GPU 4
                print(f"Using GPU {i}")
                return i
        time.sleep(wait_time)

        if t == 25:
            # after ~ 1 min increase wait time
            wait_time = 5
        if t == 100:
            wait_time = 30
        if t == 140:
            wait_time = 120

    end_time = time.perf_counter()
    raise SystemError(f"After {t+1} attempts over {(end_time-start_time)/60:.2f}"
                      f" minutes, an unused GPU could not be found")


def find_gpu():
    print("Finding GPU to use...")
    try:
        return get_gpu_id()
    except FileNotFoundError:
        return get_gpu_id(cmd="nvidia-smi.exe")


if __name__ == "__main__":
    find_gpu()

