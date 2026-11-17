"""Bound native worker pools and record available host resources."""
import os
import torch


def worker_init(worker_id):
    # Top-level function is picklable under Windows spawn.
    torch.set_num_threads(1)
    try:
        import SimpleITK as sitk
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    except ImportError:
        pass


def host_resources():
    result={'logical_cpus':os.cpu_count()}
    try:
        import psutil
        mem=psutil.virtual_memory()
        result.update(available_ram_gib=mem.available/2**30,total_ram_gib=mem.total/2**30,
                      process_rss_gib=psutil.Process().memory_info().rss/2**30)
    except ImportError:
        pass
    return result
