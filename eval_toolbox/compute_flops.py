import argparse
import sys

import torch
from tqdm import tqdm


sys.path.append("./dimsum")

from calflops import calculate_flops
from create_model import create_model


@torch.no_grad
def measure_gpu_throughput(model, batch_size, device, args):
    model = model.to("cuda")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_latency = 0.0
    for i in tqdm(range(300)):
        x = torch.randn(batch_size, args.num_in_channels, args.image_size // 8, args.image_size // 8).to(device)
        t = torch.rand((batch_size,)).to(device)
        c = torch.full((batch_size,), 0).to(device, dtype=torch.int)
        start.record()
        x_0 = model(x, t, c)
        end.record()
        torch.cuda.synchronize()
        total_latency += start.elapsed_time(end) / 300.0 / 1000  # in s
    throughput = batch_size / total_latency
    return throughput


def measure_mem(model, batch_size, device, args):
    mem = 0.0
    for _ in tqdm(range(300)):
        x = torch.randn(batch_size, args.num_in_channels, args.image_size // 8, args.image_size // 8).to(device)
        t = torch.rand((batch_size,)).to(device)
        c = torch.full((batch_size,), 0).to(device, dtype=torch.int)
        x_0 = model(x, t, c)
        mem += torch.cuda.max_memory_allocated(device) / 2**30 / 300.0
    return mem


def none_or_str(value):
    if value == "None":
        return None
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="DiM-L/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-in-channels", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--label-dropout", type=float, default=-1)
    parser.add_argument("--learn-sigma", action="store_true", default=False)
    parser.add_argument(
        "--bimamba-type", type=str, default="v2", choices=["v2", "none", "zigma_8", "sweep_8", "jpeg_8", "sweep_4"]
    )
    parser.add_argument("--pe-type", type=str, default="ape", choices=["ape", "cpe", "rope"])
    parser.add_argument("--learnable-pe", action="store_true")
    parser.add_argument(
        "--block-type",
        type=str,
        default="linear",
        choices=["linear", "raw", "wave", "combined", "window", "combined_fourier", "combined_einfft"],
    )
    parser.add_argument("--cond-mamba", action="store_true")
    parser.add_argument("--scanning-continuity", action="store_true")
    parser.add_argument("--enable-fourier-layers", action="store_true")
    parser.add_argument("--rms-norm", action="store_true")
    parser.add_argument("--fused-add-norm", action="store_true")
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument("--use-final-norm", action="store_true")
    parser.add_argument(
        "--use-attn-every-k-layers",
        type=int,
        default=-1,
    )
    parser.add_argument("--not-use-gated-mlp", action="store_true")

    group = parser.add_argument_group("MoE arguments")
    group.add_argument("--num-moe-experts", type=int, default=8)
    group.add_argument("--mamba-moe-layers", type=none_or_str, nargs="*", default=None)
    group.add_argument("--is-moe", action="store_true")
    group.add_argument(
        "--routing-mode", type=str, choices=["sinkhorn", "top1", "top2", "sinkhorn_top2"], default="top1"
    )
    group.add_argument("--gated-linear-unit", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(42)
    device = "cuda"
    model = create_model(args).to(device)
    model.eval()

    for p in model.parameters():
        p.require_grad = False

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Model size: {:.3f}MB".format(pytorch_total_params / 1000**2))

    # Calc 300 times for better acc
    mem = 0  # measure_mem(model, 1, device, args)
    throughput = measure_gpu_throughput(model, args.batch_size, device, args)
    print("Mem usage: {} (GB), Throughput: {} (imgs/s)".format(mem, throughput))

    x = torch.randn(1, args.num_in_channels, args.image_size // 8, args.image_size // 8).to(device)
    t = torch.ones(1).to(device)
    c = torch.zeros(1).to(device, dtype=torch.int)

    flops, macs, params = calculate_flops(
        model=model, kwargs={"x": x, "t": t, "y": c}, output_as_string=False, output_precision=4
    )
    print("GFLOPs:%.2f Params:%.2fM \n" % (flops / 2 / 10**9, params / 10**6))
