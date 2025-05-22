import torch
from ..helpers import string_type


def py_vmap(func, in_dims=0, out_dims=0):
    """
    Python implementation of :func:`torch.vmap`.
    """

    def wrapped(*args):
        assert all(not isinstance(a, dict) for a in args), (
            f"dictionaries are not implemented in "
            f"args={string_type(args, with_shape=True)}"
        )

        in_dims_ = (
            ([in_dims] * len(args))
            if not isinstance(in_dims, (list, tuple))
            else list(in_dims)
        )
        assert len(in_dims_) == len(args)

        batch_size = None
        batched_args = []
        for arg, in_dim in zip(args, in_dims_):
            if in_dim is None:
                batched_args.append(arg)
                continue

            assert batch_size is None or batch_size == arg.size(in_dim), (
                f"Unable to continue, batch_size={batch_size}, in_dim={in_dim}, "
                f"arg.size(in_dim)={arg.size(in_dim)}"
            )
            if batch_size is None:
                batch_size = arg.size(in_dim)
            arg = arg.movedim(in_dim, 0)
            batched_args.append(arg)

        results = []
        for i in range(batch_size):
            input_slice = [
                (arg[i] if isinstance(arg, torch.Tensor) and in_dim is not None else arg)
                for arg, in_dim in zip(batched_args, in_dims_)
            ]
            result = func(*input_slice)
            results.append(result)

        if isinstance(results[0], torch.Tensor):
            stacked = torch.stack(results)
            if out_dims != 0:
                return stacked.movedim(0, out_dims)
            return stacked
        return results

    return wrapped
