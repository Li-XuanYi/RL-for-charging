"""Runtime-only CPU RSS and PyTorch CUDA allocation observations."""
import functools
import os
from pathlib import Path
import threading
import time
import torch
from common import dump


def rss_bytes():
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
                (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
                'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi=ctypes.WinDLL('psapi',use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            raise OSError(ctypes.get_last_error(),'GetProcessMemoryInfo failed')
        return int(counters.WorkingSetSize)
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    raise RuntimeError('Linux process RSS unavailable')


class MemoryProbe:
    def __init__(self,device):
        self.device=torch.device(device);self.event=threading.Event();self.error=None;self.peak=None
    def sample(self):
        try:
            value=rss_bytes();self.peak=max(self.peak or 0,value);return value
        except (OSError,RuntimeError,ValueError) as exc:
            self.error=str(exc);return None
    def __enter__(self):
        self.start=self.sample();self.started=time.perf_counter()
        if self.device.type=='cuda':
            torch.cuda.synchronize(self.device);torch.cuda.reset_peak_memory_stats(self.device)
        def poll():
            while not self.event.wait(.25):self.sample()
        self.thread=threading.Thread(target=poll,daemon=True);self.thread.start();return self
    def __exit__(self,kind,value,tb):
        self.event.set();self.thread.join(timeout=2);self.end=self.sample()
        self.result={'cpu_rss_start_bytes':self.start,'cpu_rss_end_bytes':self.end,
                     'cpu_rss_sampled_peak_bytes':self.peak,'rss_sample_interval_s':.25,
                     'device':str(self.device),'wall_s':time.perf_counter()-self.started,
                     'rss_error':self.error,'scope':'Current process including solver and caches; sampled RSS peak is a lower bound, not isolated deployed network memory.'}
        if self.device.type=='cuda':
            torch.cuda.synchronize(self.device)
            self.result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
                cuda_memory_scope='PyTorch allocator for this process; excludes driver/context and other jobs')


def profile_training(function):
    @functools.wraps(function)
    def wrapped(spec,vectors,cfg,out,validation_cases,seed):
        device='cpu' if spec['method'] in ('cccv','cccv_continuous','soc_rule') else spec.get('device',cfg.get('device','cpu'))
        with MemoryProbe(device) as monitor:
            result=function(spec,vectors,cfg,out,validation_cases,seed)
        result['resource_usage']=monitor.result
        dump(Path(out)/'resource_usage.json',monitor.result);dump(Path(out)/'entry.json',result)
        return result
    return wrapped
