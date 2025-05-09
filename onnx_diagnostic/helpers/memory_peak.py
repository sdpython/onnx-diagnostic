import multiprocessing
import os
from typing import Dict, Optional


def get_memory_rss(pid: int) -> int:
    """
    Returns the physical memory used by a process.

    :param pid: process id, current one is `os.getpid()`
    :return: physical memory

    It relies on the module :epkg:`psutil`.
    """
    import psutil

    process = psutil.Process(pid)
    mem = process.memory_info().rss
    return mem


class Monitor:
    def __init__(self):
        self.max_peak = 0
        self.average = 0
        self.n_measures = 0
        self.begin = 0
        self.end = 0

    def to_dict(self, unit: int = 1):
        funit = float(unit)
        return dict(
            peak=self.max_peak / funit,
            mean=self.average * 1.0 / self.n_measures / funit,
            n=self.n_measures,
            begin=self.begin / funit,
            end=self.end / funit,
        )

    @property
    def delta_peak(self):
        return self.max_peak - self.begin

    @property
    def delta_end(self):
        return self.end - self.begin

    @property
    def delta_avg(self):
        return self.average / self.n_measures - self.begin

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(begin={self.begin}, end={self.end}, "
            f"peak={self.max_peak}, average={self.average}, n={self.n_measures}, "
            f"d_end={self.delta_end}, d_peak={self.delta_peak}, d_avg={self.delta_avg}"
            f")"
        )

    def update(self, mem):
        if self.n_measures == 0:
            self.begin = mem
        self.max_peak = max(mem, self.max_peak)
        self.average += mem
        self.end = mem
        self.n_measures += 1

    def send(self, conn):
        conn.send(self.max_peak)
        conn.send(self.average)
        conn.send(self.n_measures)
        conn.send(self.begin)
        conn.send(self.end)

    @classmethod
    def recv(cls, conn):
        m = cls()
        m.max_peak = conn.recv()
        m.average = conn.recv()
        m.n_measures = conn.recv()
        m.begin = conn.recv()
        m.end = conn.recv()
        return m


def _process_memory_spy(conn):
    # Sends the value it started.
    conn.send(-2)

    # process id to spy on
    pid = conn.recv()

    # delay between two measures
    timeout = conn.recv()

    # do CUDA
    cuda = conn.recv()

    import psutil

    process = psutil.Process(pid)

    if cuda:
        from pynvml import (
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlInit,
            nvmlShutdown,
        )

        nvmlInit()
        n_gpus = nvmlDeviceGetCount()
        handles = [nvmlDeviceGetHandleByIndex(i) for i in range(n_gpus)]

        def gpu_used():
            return [nvmlDeviceGetMemoryInfo(h).used for h in handles]

        gpus = [Monitor() for i in range(n_gpus)]
    else:
        gpus = []

    cpu = Monitor()

    conn.send(-2)

    # loop
    while True:
        mem = process.memory_info().rss
        cpu.update(mem)
        if cuda:
            for r, g in zip(gpu_used(), gpus):
                g.update(r)
        if conn.poll(timeout=timeout):
            code = conn.recv()
            if code == -3:
                break

    # final iteration
    end = process.memory_info().rss
    cpu.update(end)
    if cuda:
        for r, g in zip(gpu_used(), gpus):
            g.update(r)

    # send
    cpu.send(conn)
    conn.send(len(gpus))
    for g in gpus:
        g.send(conn)
    if cuda:
        nvmlShutdown()
    conn.close()


class MemorySpy:
    """
    Information about the spy. It class method `start`.
    Method `stop` can be called to end the measure.

    :param pid: process id  of the process to spy on
    :param delay: spy on every delay seconds
    :param cuda: enable cuda monitoring
    """

    def __init__(self, pid: int, delay: float = 0.01, cuda: bool = False):
        self.pid = pid
        self.delay = delay
        self.cuda = cuda
        self.start()

    def start(self) -> "MemorySpy":
        """Starts another process and tells it to spy."""
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.child_process = multiprocessing.Process(
            target=_process_memory_spy, args=(self.child_conn,)
        )
        self.child_process.start()
        data = self.parent_conn.recv()
        if data != -2:
            raise RuntimeError(f"The child processing is supposed to send -2 not {data}.")
        self.parent_conn.send(self.pid)
        self.parent_conn.send(self.delay)
        self.parent_conn.send(1 if self.cuda else 0)
        data = self.parent_conn.recv()
        if data != -2:
            raise RuntimeError(
                f"The child processing is supposed to send -2 again not {data}."
            )
        return self

    def stop(self):
        """Stops spying on."""
        self.parent_conn.send(-3)

        cpu = Monitor.recv(self.parent_conn)

        n_gpus = self.parent_conn.recv()
        gpus = []
        for _i in range(n_gpus):
            gpus.append(Monitor.recv(self.parent_conn))

        self.parent_conn.close()
        self.child_process.join()
        res = dict(cpu=cpu)
        if self.cuda:
            res["gpus"] = gpus
        return res


def start_spying_on(
    pid: Optional[int] = None, delay: float = 0.01, cuda: bool = False
) -> MemorySpy:
    """
    Starts the memory spy. The function starts another
    process spying on the one sent as an argument.

    :param pid: process id to spy or the the current one.
    :param delay: delay between two measures.
    :param cuda: True or False to get memory for cuda devices

    Example:

    .. code-block:: python

        from onnx_diagnostic.helpers.memory_peak import start_spying_on, flatten

        p = start_spying_on()
        # ...
        # code to measure
        # ...
        stat = p.stop()
        print(stat)
        print(flatten(stat))
    """
    if pid is None:
        pid = os.getpid()
    return MemorySpy(pid, delay, cuda)


def flatten(ps, prefix: str = "") -> Dict[str, float]:
    obs = ps["cpu"].to_dict(unit=2**20)
    if "gpus" in ps:
        for i, g in enumerate(ps["gpus"]):
            for k, v in g.to_dict(unit=2**20).items():
                obs[f"gpu{i}_{k}"] = v
    if prefix:
        obs = {f"{prefix}{k}": v for k, v in obs.items()}
    return obs
