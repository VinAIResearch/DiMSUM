import argparse
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
import sys
sys.path.append("./vim")

# from torchtoolbox.tools import summary
from thop.profile import profile
<<<<<<< HEAD
# from models_dim import DiM_models
# from models_dmm import mamba_models
from create_model import create_model


def none_or_str(value):
    if value == 'None':
        return None
    return value
=======
from models_dim import DiM_models
from models_dmm import mamba_models
>>>>>>> origin/trungdt21


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
<<<<<<< HEAD
    parser.add_argument("--model", type=str, default="MambaDiffV1_XL_2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-in-channels", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--label-dropout", type=float, default=-1)
    parser.add_argument("--learn-sigma", action='store_true', default=False)
    parser.add_argument("--bimamba-type", type=str, default="v2", choices=['v2', 'none'])

    group = parser.add_argument_group("MoE arguments")
    group.add_argument("--num-moe-experts", type=int, default=8)
    group.add_argument("--mamba-moe-layers", type=none_or_str, nargs="*", default=None)
    group.add_argument("--is-moe", action="store_true")
    group.add_argument("--routing-mode", type=str, choices=['sinkhorn', 'top1', 'top2', 'sinkhorn_top2'], default='top1')
    group.add_argument("--gated-linear-unit", action="store_true")


=======
    parser.add_argument("--model", type=str, choices=list(mamba_models.keys())+list(DiM_models.keys()), default="MambaDiffV1_XL_2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learn-sigma", action='store_true', default=False)
>>>>>>> origin/trungdt21
    args = parser.parse_args()

    torch.manual_seed(42)
    device = 'cuda'
<<<<<<< HEAD
    model = create_model(args).to(device)
    model.eval()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Model size: {:.3f}MB".format(pytorch_total_params / 1000**2))
=======
    model = DiM_models[args.model](learn_sigma = args.learn_sigma).to(device)
    model.eval()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params / 1024**2)
    print("Model size: {:.3f}MB".format(pytorch_total_params))
>>>>>>> origin/trungdt21
    
    # Calc 300 times for better acc
    # mem = 0.
    # for _ in range(300):
    #     x_t_1 = torch.randn(args.batch_size, args.num_in_channels, args.image_size//args.f, args.image_size//args.f).to(device)
    #     # t = torch.rand((args.batch_size,)).to(device)
    #     t = torch.tensor(1.0).to(device)
    #     x_0 = model(t, x_t_1)
    #     mem += torch.cuda.max_memory_allocated(device) / 2**30
    # print("Mem usage: {} (GB)".format(mem/300.))

    x = torch.randn(args.batch_size, 4, args.image_size//8, args.image_size//8).to(device)
    t = torch.ones(args.batch_size).to(device)

<<<<<<< HEAD
    flops = FlopCountAnalysis(model, (x, t))
    print(flop_count_table(flops))
    print(flops.total())
=======
    # flops = FlopCountAnalysis(model, (t, x))
    # print(flop_count_table(flops))
    # print(flops.total())
>>>>>>> origin/trungdt21
    
    print("%s | %s" % ("Params(M)", "FLOPs(G)"))
    print("---|---")
    total_ops, total_params = profile(model, (x, t), verbose=False)
    print(
        "%.2f | %.2f" % (total_params / (1000 ** 2), total_ops / (1000 ** 3))
    )
<<<<<<< HEAD
=======
 

>>>>>>> origin/trungdt21
