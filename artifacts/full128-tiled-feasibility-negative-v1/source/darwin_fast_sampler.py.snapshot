"""Direct, bounded macOS kernel/libproc sampling for one supervised child group.

Uses public SDK `sysctlbyname`, `proc_listpgrppids`, and `proc_pidinfo` APIs.
No shell process listing is called by the fast safety loop.
"""

import ctypes
import ctypes.util
import errno
import os
import time


class XswUsage(ctypes.Structure):
    _fields_ = [("xsu_total", ctypes.c_uint64), ("xsu_avail", ctypes.c_uint64),
                ("xsu_used", ctypes.c_uint64), ("xsu_pagesize", ctypes.c_uint32),
                ("xsu_encrypted", ctypes.c_int32)]


class ProcTaskInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "pti_virtual_size", "pti_resident_size", "pti_total_user", "pti_total_system",
        "pti_threads_user", "pti_threads_system")] + [
        (name, ctypes.c_int32) for name in (
            "pti_policy", "pti_faults", "pti_pageins", "pti_cow_faults",
            "pti_messages_sent", "pti_messages_received", "pti_syscalls_mach",
            "pti_syscalls_unix", "pti_csw", "pti_threadnum", "pti_numrunning",
            "pti_priority")]


class ProcBsdInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "pbi_flags", "pbi_status", "pbi_xstatus", "pbi_pid", "pbi_ppid",
        "pbi_uid", "pbi_gid", "pbi_ruid", "pbi_rgid", "pbi_svuid", "pbi_svgid",
        "rfu_1")] + [("pbi_comm", ctypes.c_char * 16),
                      ("pbi_name", ctypes.c_char * 32)] + [
        (name, ctypes.c_uint32) for name in (
            "pbi_nfiles", "pbi_pgid", "pbi_pjobc", "e_tdev", "e_tpgid", "pbi_nice")] + [
        ("pbi_start_tvsec", ctypes.c_uint64), ("pbi_start_tvusec", ctypes.c_uint64)]


class FastDarwinSampler:
    def __init__(self):
        system = ctypes.util.find_library("System")
        proc = ctypes.util.find_library("proc")
        if not system or not proc:
            raise RuntimeError("macOS libSystem or libproc unavailable")
        self.system = ctypes.CDLL(system, use_errno=True)
        self.proc = ctypes.CDLL(proc, use_errno=True)
        self.system.sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p,
                                           ctypes.POINTER(ctypes.c_size_t),
                                           ctypes.c_void_p, ctypes.c_size_t]
        self.system.sysctlbyname.restype = ctypes.c_int
        self.proc.proc_listpgrppids.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        self.proc.proc_listpgrppids.restype = ctypes.c_int
        self.proc.proc_listchildpids.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        self.proc.proc_listchildpids.restype = ctypes.c_int
        self.proc.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                          ctypes.c_void_p, ctypes.c_int]
        self.proc.proc_pidinfo.restype = ctypes.c_int
        if (ctypes.sizeof(XswUsage) != 32 or ctypes.sizeof(ProcTaskInfo) != 96 or
                ctypes.sizeof(ProcBsdInfo) != 136):
            raise RuntimeError("unexpected SDK structure sizes")

    def process_start_identity(self, pid):
        info = ProcBsdInfo()
        ctypes.set_errno(0)
        got = self.proc.proc_pidinfo(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info))
        if got == 0:
            return None
        if got != ctypes.sizeof(info) or info.pbi_pid != pid:
            raise OSError(ctypes.get_errno(), "PROC_PIDTBSDINFO failed for PID %d" % pid)
        return (int(info.pbi_start_tvsec), int(info.pbi_start_tvusec))

    def signal_if_same_process(self, pid, expected_identity, sig):
        """Recheck kernel start identity immediately before targeting a detached PID."""
        try:
            observed = self.process_start_identity(pid)
        except OSError as error:
            return {"pid": pid, "identity_matched": False, "signaled": False,
                    "error": "%s: %s" % (type(error).__name__, error)}
        if observed is None or observed != tuple(expected_identity):
            return {"pid": pid, "identity_matched": False, "signaled": False}
        try:
            os.kill(pid, sig)
        except OSError as error:
            return {"pid": pid, "identity_matched": True, "signaled": False,
                    "error": "%s: %s" % (type(error).__name__, error)}
        return {"pid": pid, "identity_matched": True, "signaled": True}

    def _sysctl(self, name, value):
        size = ctypes.c_size_t(ctypes.sizeof(value))
        ctypes.set_errno(0)
        rc = self.system.sysctlbyname(name.encode("ascii"), ctypes.byref(value),
                                     ctypes.byref(size), None, 0)
        if rc != 0 or size.value != ctypes.sizeof(value):
            raise OSError(ctypes.get_errno(), "sysctlbyname failed or returned wrong size: " + name)
        return value

    def host(self):
        start = time.monotonic()
        mask = int(self._sysctl("kern.memorystatus_vm_pressure_level", ctypes.c_int32()).value)
        available = int(self._sysctl("kern.memorystatus_level", ctypes.c_int32()).value)
        swap = self._sysctl("vm.swapusage", XswUsage())
        if available < 0 or available > 100 or swap.xsu_pagesize <= 0:
            raise ValueError("invalid direct host memory reading")
        return {"kernel_pressure_mask": mask, "available_percent": available,
                "swap_used_bytes": int(swap.xsu_used),
                "swap_total_bytes": int(swap.xsu_total),
                "host_read_seconds": time.monotonic() - start}

    def _pid_list(self, function, identifier):
        capacity = 256
        pids = (ctypes.c_int32 * capacity)()
        ctypes.set_errno(0)
        count = function(identifier, pids, ctypes.sizeof(pids))
        if count < 0 or count > capacity:
            raise OSError(ctypes.get_errno(), "libproc PID enumeration failed or returned invalid count")
        if count == capacity:
            raise RuntimeError("libproc PID enumeration buffer may have truncated")
        return [int(pid) for pid in pids[:count] if pid > 0]

    def group(self, pgid, root_pid):
        start = time.monotonic()
        group_pids = set(self._pid_list(self.proc.proc_listpgrppids, pgid))
        descendants = set()
        todo = [int(root_pid)]
        visited = set()
        while todo:
            parent = todo.pop()
            if parent in visited:
                continue
            visited.add(parent)
            if len(visited) > 256:
                raise RuntimeError("descendant tree exceeded bounded 256-process traversal")
            children = self._pid_list(self.proc.proc_listchildpids, parent)
            descendants.update(children)
            todo.extend(child for child in children if child not in visited)
        candidates = group_pids | descendants | {int(root_pid)}
        resident_bytes = 0
        live_pids = []
        detached = []
        detached_identities = {}
        for pid in sorted(candidates):
            try:
                actual_pgid = os.getpgid(pid)
                if pid not in descendants and actual_pgid != pgid:
                    continue
            except ProcessLookupError:
                continue
            info = ProcTaskInfo()
            ctypes.set_errno(0)
            got = self.proc.proc_pidinfo(pid, 4, 0, ctypes.byref(info), ctypes.sizeof(info))
            if got != ctypes.sizeof(info):
                if got == 0 and ctypes.get_errno() == errno.ESRCH:
                    continue
                raise OSError(ctypes.get_errno(), "proc_pidinfo failed for PID %d" % pid)
            live_pids.append(int(pid))
            if pid in descendants and actual_pgid != pgid:
                detached.append(int(pid))
                identity = self.process_start_identity(pid)
                if identity is None:
                    raise RuntimeError("detached descendant start identity unavailable")
                detached_identities[str(pid)] = identity
            resident_bytes += int(info.pti_resident_size)
        return {"pgid": int(pgid), "pids": live_pids,
                "detached_descendant_pids": detached,
                "detached_descendant_start_identities": detached_identities,
                "process_group_resident_bytes": resident_bytes,
                "group_read_seconds": time.monotonic() - start}

    def sample(self, pgid, root_pid):
        host = self.host()
        group = self.group(pgid, root_pid)
        return dict(host, **group)
