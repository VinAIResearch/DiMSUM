import argparse
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
import sys
sys.path.append("./vim")

# from torchtoolbox.tools import summary
from thop.profile import profile
from create_model import create_model


def none_or_str(value):
    if value == 'None':
        return None
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="MambaDiffV1_XL_2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-in-channels", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--label-dropout", type=float, default=-1)
    parser.add_argument("--learn-sigma", action='store_true', default=False)
    parser.add_argument("--bimamba-type", type=str, default="v2", choices=['v2', 'none'])
    parser.add_argument("--pe-type", type=str, default="ape", choices=["ape", "cpe", "rope"])
    parser.add_argument("--block-type", type=str, default="linear", choices=["linear", "raw"])

    group = parser.add_argument_group("MoE arguments")
    group.add_argument("--num-moe-experts", type=int, default=8)
    group.add_argument("--mamba-moe-layers", type=none_or_str, nargs="*", default=None)
    group.add_argument("--is-moe", action="store_true")
    group.add_argument("--routing-mode", type=str, choices=['sinkhorn', 'top1', 'top2', 'sinkhorn_top2'], default='top1')
    group.add_argument("--gated-linear-unit", action="store_true")


    args = parser.parse_args()

    torch.manual_seed(42)
    device = 'cuda'
    model = create_model(args).to(device)
    model.eval()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Model size: {:.3f}MB".format(pytorch_total_params / 1000**2))
    
    # Calc 300 times for better acc
    mem = 0.
    for _ in range(300):
        x_t_1 = torch.randn(args.batch_size, args.num_in_channels, args.image_size//8, args.image_size//8).to(device)
        # t = torch.rand((args.batch_size,)).to(device)
        t = torch.tensor([1.0]).to(device)
        x_0 = model(x_t_1, t)
        mem += torch.cuda.max_memory_allocated(device) / 2**30
    print("Mem usage: {} (GB)".format(mem/300.))

    x = torch.randn(args.batch_size, args.num_in_channels, args.image_size//8, args.image_size//8).to(device)
    t = torch.ones(args.batch_size).to(device)

    flops = FlopCountAnalysis(model, (x, t))
    print(flop_count_table(flops))
    print(flops.total())
    
    # print("%s | %s" % ("Params(M)", "FLOPs(G)"))
    # print("---|---")
    # total_ops, total_params = profile(model, (x, t), verbose=False)
    # print(
    #     "%.2f | %.2f" % (total_params / (1000 ** 2), total_ops / (1000 ** 3))
    # )
