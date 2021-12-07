import torch



xs = torch.randn(2,2,3)
value,indx = xs.max(2)

print(xs)
print(value)
print(indx)
