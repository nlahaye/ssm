"""
Optional multi-process E-step for hmm_old.ARHMM._fit_em.

The E-step is independent across sequences, so the sequences are split into contiguous chunks and each
worker runs expected_states on its chunk with the current parameters. The results are identical to the serial
loop (same per-sequence computation, same order). Data are sent to each worker once, when the pool starts;
each EM iteration only sends the model (parameters, a few hundred kB).

Control (in order): ARHMM.estep_n_jobs (class or instance attribute) > env SSM_ESTEP_JOBS > 1 (serial).
Parallel runs only when there are at least SSM_ESTEP_MIN_SEQS sequences (default 200): for small data the
per-iteration pickling costs more than it saves. Workers use the "spawn" start method unless
SSM_ESTEP_START_METHOD is set (fork is unsafe once JAX / CUDA are initialised in the parent). Each worker
caps BLAS / OpenMP / numba threads at 1.
"""
import os
import pickle

import numpy as np

_W = {}


def n_jobs_for(model, n_seqs):
    n = getattr(model, "estep_n_jobs", None)
    if n is None:
        n = int(os.environ.get("SSM_ESTEP_JOBS", "1") or 1)
    n = int(n)
    if n < 0:
        n = os.cpu_count() or 1
    min_seqs = int(os.environ.get("SSM_ESTEP_MIN_SEQS", "200"))
    if n <= 1 or n_seqs < max(min_seqs, 2):
        return 1
    return min(n, n_seqs)


def _init_worker(payload):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[k] = "1"
    _W["datas"], _W["inputs"], _W["masks"], _W["tags"] = pickle.loads(payload)


def _run_chunk(args):
    model_bytes, lo, hi = args
    model = pickle.loads(model_bytes)
    d, i, m, t = _W["datas"], _W["inputs"], _W["masks"], _W["tags"]
    return lo, [model.expected_states(d[j], i[j], m[j], t[j]) for j in range(lo, hi)]


class EStepPool(object):
    """Context manager: pool = EStepPool(model, datas, ...); pool.expectations(model) -> list per sequence."""

    def __init__(self, model, datas, inputs, masks, tags):
        self.n = n_jobs_for(model, len(datas))
        self.N = len(datas)
        self.pool = None
        if self.n > 1:
            import multiprocessing as mp
            ctx = mp.get_context(os.environ.get("SSM_ESTEP_START_METHOD", "spawn"))
            payload = pickle.dumps((list(datas), list(inputs), list(masks), list(tags)), protocol=pickle.HIGHEST_PROTOCOL)
            self.pool = ctx.Pool(self.n, initializer=_init_worker, initargs=(payload,))
            # a few chunks per worker for load balance (variable-length sequences)
            n_chunks = min(self.N, 4 * self.n)
            edges = np.linspace(0, self.N, n_chunks + 1).astype(int)
            self.chunks = [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def expectations(self, model, datas, inputs, masks, tags):
        if self.pool is None:
            return [model.expected_states(data, input, mask, tag)
                    for data, input, mask, tag in zip(datas, inputs, masks, tags)]
        mb = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        out = [None] * self.N
        for lo, res in self.pool.imap_unordered(_run_chunk, [(mb, a, b) for a, b in self.chunks]):
            out[lo:lo + len(res)] = res
        return out
