import torch
import torch.distributed as dist


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def wait():
    dist.barrier()


def setup():
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def initialize_distributed(device_ids, local_rank):
    setup()
    device = torch.device(device_ids[local_rank])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(device_ids[local_rank])

    return device, rank, world_size


def cleanup():
    dist.destroy_process_group()