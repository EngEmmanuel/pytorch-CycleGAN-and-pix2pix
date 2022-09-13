import torch
import torch.cuda as gc
import math

print(torch.version.cuda)

print("Cuda version: ", torch.version.cuda)

available = gc.is_available()
numGPU = gc.device_count()

print("GPU Availability: ", available, "Number of GPUs: ", numGPU, sep='\n')
for i in range(numGPU):
    print("Device Name:", gc.get_device_name(i), "Device Capability:", gc.get_device_capability(i), sep='\t')

# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# For this example, the output y is a linear function of (x, x^2, x^3), so # we can consider it as a linear layer neural network. Let's prepare the
# tensor (x, x^2, x^3).
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

print("Cuda operation")
xx.cuda()