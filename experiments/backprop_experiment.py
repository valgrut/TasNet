import torch


x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y) # Note the fn=Add... (AddBackward)

z = y / 2
print(z) # Note the fn=Div... (DivBackward0)

z = y * y * 3  # non-scalar in z
out = z.mean() # now contains single scalar
print(z, out)

out.backward() #only usable on single-scalar without params
print(x.grad)

