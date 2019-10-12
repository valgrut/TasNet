import torch

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000: # L2 norm (soucet druhych mocnin a z toho odmocnina)
    y = y * 2

print(y)      # vypise tensor - vcetne vlastnosti
print(y.data) # vypise tensor - pouze hodnoty

gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)

print(x.grad)
