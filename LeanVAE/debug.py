import torch 

from modules.backbones import NAF

NAF_ob = NAF(num_layer=12,
             dim=512)

print(NAF_ob)
x = torch.randn(2, 8, 1024, 1024, 512)
output = NAF_ob(x)
print(output.shape)
