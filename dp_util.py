import os
import torch


def init_group():
    rank = int(os.environ.get("RANK") or 0)
    world_size = int(os.environ.get("WORLD_SIZE") or 1)
    device_count = torch.cuda.device_count()
    if device_count > 0:
        device = rank % device_count
        torch.cuda.set_device("cuda:{}".format(device))
        print('> initializing rank {} ...'.format(device), flush=True)
    init_method = "tcp://{}:{}".format(os.getenv('MASTER_ADDR', 'localhost'),
                                       os.getenv('MASTER_PORT', '6000'))
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size, rank=rank,
        init_method=init_method)
    print('> initialized rank {} ...\n'.format(torch.distributed.get_rank()), flush=True)


def all_reduce_grads(grads):
    coalesced = torch._utils._flatten_dense_tensors(grads)
    coalesced /= torch.distributed.get_world_size()
    torch.distributed.all_reduce(coalesced, group=torch.distributed.distributed_c10d._get_default_group())
    torch.cuda.synchronize()
    for buf, synced in zip(grads, torch._utils._unflatten_dense_tensors(coalesced, grads)):
        buf.copy_(synced)


def is_first_rank():
    return torch.distributed.get_rank() == 0


def is_last_rank():
    return torch.distributed.get_rank() == torch.distributed.get_world_size()


def bcast_and_slice(data, dp=False):
    if dp:
        torch.distributed.broadcast(data, src=0, group=torch.distributed.distributed_c10d._get_default_group())
        data_size_per_rank = data.shape[0] // torch.distributed.get_world_size()
        assert data_size_per_rank > 0, "expect at least one batch size per rank: {}".format(data_size_per_rank)
        rank = torch.distributed.get_rank()
        start_index = rank * data_size_per_rank
        end_index = (rank + 1) * data_size_per_rank
        if is_last_rank():
            end_index = data.shape[0]
        return data[start_index:end_index]
    return data
