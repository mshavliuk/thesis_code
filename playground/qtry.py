import torch
import torch.nn as nn

class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)

model_fp32 = SimpleLinearModel()


#%%
from torch.ao.quantization import QuantStub, DeQuantStub, get_default_qconfig

class QuantizedLinearModel(nn.Module):
    def __init__(self):
        super(QuantizedLinearModel, self).__init__()
        # self.quant = QuantStub()  # Converts input to quantized format
        self.linear = nn.Linear(4, 4)
        self.dequant = DeQuantStub()  # Converts output back to float

    def forward(self, x):
        # x = self.quant(x)
        x = self.linear(x)
        x = self.dequant(x)
        return x

model_fp32 = QuantizedLinearModel()
model_fp32.qconfig = get_default_qconfig("fbgemm")  # Use "qnnpack" for ARM devices

#%%

from torch.ao.quantization import prepare

model_fp32.eval()  # Static quantization requires eval mode
model_prepared = prepare(model_fp32)

#%%

# Example calibration data
calibration_data = torch.rand(10, 4)  # Representative dataset
for data in calibration_data:
    model_prepared(data)

#%%
from torch.ao.quantization import convert

model_int8 = convert(model_prepared)

#%%
# Example of pre-quantizing input tensor
scale, zero_point = 1.0 / 255, 0  # Example scale and zero-point
input_uint8 = torch.quantize_per_tensor(torch.rand(1, 4), scale, zero_point, torch.quint8)

# Run the quantized model with uint8 input
output = model_int8(input_uint8)
print(output)


# %%
# same on cuda
cuda_model_int8 = model_int8.to('cuda')
cuda_input_uint8 = input_uint8.to('cuda')

cuda_output = cuda_model_int8(cuda_input_uint8)
print(cuda_output)





