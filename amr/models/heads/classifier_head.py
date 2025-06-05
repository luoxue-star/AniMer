from torch import nn
    

class ClassTokenHead(nn.Module):
    def __init__(self, embed_dim=1280, hidden_dim=4096, output_dim=256, num_layers=3, last_bn=True):
        super().__init__()
        mlp = []
        for l in range(num_layers):
            dim1 = embed_dim if l == 0 else hidden_dim
            dim2 = output_dim if l == num_layers - 1 else hidden_dim
            mlp.append(nn.Linear(dim1, dim2, bias=False))
            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))
        self.head = nn.Sequential(*mlp)

    def forward(self, x):
        cls_feats = self.head(x)
        return cls_feats