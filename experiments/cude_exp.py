from __future__ import print_function
import torch

x = torch.empty(5,3)
print(x)

x = torch.rand(5,3)
print(x)

a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

# Cuda is available
if torch.cuda.is_available():
    print("Cuda is available")
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
else:
    print("Cuda is NOT available")


x = torch.rand(1,5)
print(x)
x = torch.rand(5,1)
print(x)

