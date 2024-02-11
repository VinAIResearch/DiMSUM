import argparse
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
import sys
sys.path.append("./vim")

# from torchtoolbox.tools import summary
from thop.profile import profile
from models_dim import DiM_models
from models_dmm import mamba_models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(mamba_models.keys())+list(DiM_models.keys()), default="MambaDiffV1_XL_2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learn-sigma", action='store_true', default=False)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = 'cuda'
    model = DiM_models[args.model](learn_sigma = args.learn_sigma).to(device)
    model.eval()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params / 1024**2)
    print("Model size: {:.3f}MB".format(pytorch_total_params))
    
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

    # flops = FlopCountAnalysis(model, (t, x))
    # print(flop_count_table(flops))
    # print(flops.total())
    
    print("%s | %s" % ("Params(M)", "FLOPs(G)"))
    print("---|---")
    total_ops, total_params = profile(model, (x, t), verbose=False)
    print(
        "%.2f | %.2f" % (total_params / (1000 ** 2), total_ops / (1000 ** 3))
    )
 

