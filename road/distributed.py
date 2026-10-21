"""Small collective helpers that propagate ordinary Python I/O failures."""

import torch.distributed as dist


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def rank_zero_call(operation):
    """Run non-collective I/O once; all ranks receive its result or error.

    All ranks must call this helper. It cannot recover process termination,
    NCCL failure, or an exception inside an unmatched collective.
    """
    if world_size() == 1:
        return operation()
    message = [None]
    if rank() == 0:
        try:
            message[0] = (True, operation())
        except Exception as error:
            message[0] = (False, f"{type(error).__name__}: {error}")
    if dist.is_initialized():
        dist.broadcast_object_list(message, src=0)
    succeeded, result = message[0]
    if not succeeded:
        raise RuntimeError(f"Rank-zero operation failed: {result}")
    return result


def local_call(operation):
    """Run non-collective work per rank and report any failure on every rank."""
    if world_size() == 1:
        return operation()
    value = None
    error = None
    try:
        value = operation()
    except Exception as exc:
        error = f"rank {rank()}: {type(exc).__name__}: {exc}"
    errors = [error]
    if dist.is_initialized():
        errors = [None] * world_size()
        dist.all_gather_object(errors, error)
    failures = [item for item in errors if item is not None]
    if failures:
        raise RuntimeError("; ".join(failures))
    return value
