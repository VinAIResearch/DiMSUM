import torch.nn as nn


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim))
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ["proj.%d.weight" % i for i in range(4)]


class AdaInPosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(AdaInPosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim, bias=True))
        self.norm = nn.LayerNorm(embed_dim)
        self.s = s

    def forward(self, x, c, H, W):
        B, N, C = x.shape
        feat_token = x
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = modulate(self.norm(x), shift, scale)
        return x

    def no_weight_decay(self):
        return ["proj.%d.weight" % i for i in range(4)]
