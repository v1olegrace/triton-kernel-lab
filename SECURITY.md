# Security policy

This is a research and portfolio project, not a hardened production runtime.
GPU kernels process tensors supplied by the caller and assume a trusted local
CUDA environment.

Report security-sensitive issues privately to `maurulycan@gmail.com`.
Include:

- affected version or commit;
- operating system, driver, GPU, PyTorch, and Triton versions;
- a minimal reproducer;
- expected and observed behavior.

Do not include credentials, private datasets, or proprietary model weights in
reports.
