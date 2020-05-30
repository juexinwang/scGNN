import torch.multiprocessing as mp
import torch
import time
import numpy as np

def foo(worker,tl):
    # print(worker)
    # tl[worker] += (worker+1) * 1000
    tl[worker,:] += (worker+1) * 1000

if __name__ == '__main__':
    # tl = [torch.randn(100000000), torch.randn(100000000), torch.randn(100000000), torch.randn(100000000), torch.randn(100000000), torch.randn(100000000)]
    tl = np.random.randn(6,1000000)
    tl = torch.from_numpy(tl)

    # for t in tl:
    #     t.share_memory_()
    tl.share_memory_()

    print("before mp: tl=")
    print(tl)
    processes = []
    ctime = time.time()

    # p0 = mp.Process(target=foo, args=(0, tl))
    # p1 = mp.Process(target=foo, args=(1, tl))
    # p0.start()
    # p1.start()
    # p0.join()
    # p1.join()

    for i in range(6):
        px = mp.Process(target=foo, args=(i, tl))
        px.start()
        processes.append(px)
    for px in processes:
        px.join()
        # foo(i,tl)

    rtime = time.time()-ctime

    print("after mp: tl=")
    print(tl)
    print(rtime)