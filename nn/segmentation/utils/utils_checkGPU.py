#!/usr/bin/python3
import subprocess
import time
import random

'''
Wait until a gpu is available
to wait in the queue, you just need to use wait4FreeGPU to wait
until you get a gpu device id,
then you are supposed to invoke registerGPU,
if you get True response, feel free to use the GPU,
and if you failed to register (i.e. False returned),
then you have to wait4FreeGPU again.
'''

def get_free_mem_GPU():
    result = subprocess.check_output(['nvidia-smi', \
            '--query-gpu=memory.free,memory.used', \
            '--format=csv,nounits,noheader'])
    result = result.decode('utf-8')
    result = [[int(y) for y in x.strip().split(',')] \
            for x in result.strip().split('\n')]
    return result 

def wait4FreeGPU(mem_MB, used_MB, wall=0, interval=60, double_check=10):
    starttime = time.time()
    while wall<=0 or time.time() - starttime <= wall:
        mem = get_free_mem_GPU()
        for index in random.sample(range(len(mem)), len(mem)):
            if mem[index][0] >= mem_MB and mem[index][1] <= used_MB:
                time.sleep(double_check)
                mem = get_free_mem_GPU()
                if mem[index][0] >= mem_MB and mem[index][1] <= used_MB:
                    return index
        time.sleep(interval)
    return -1

if __name__ == '__main__':
    print(wait4FreeGPU(9000, 1000))
