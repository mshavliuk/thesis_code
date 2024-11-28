# import the modules used here in this recipe
import os
import tempfile

import torch
import torch.nn as nn
import torch.quantization


# define a very, very simple LSTM for demonstration purposes
# in this case, we are wrapping ``nn.LSTM``, one layer, no preprocessing or postprocessing
# inspired by
# `Sequence Models and Long Short-Term Memory Networks tutorial <https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html`_, by Robert Guthrie
# and `Dynamic Quanitzation tutorial <https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html>`__.
class lstm_for_demonstration(nn.Module):
    """Elementary Long Short Term Memory style model which simply wraps ``nn.LSTM``
       Not to be used for anything other than demonstration.
    """
    
    def __init__(self, in_dim, out_dim, depth):
        super(lstm_for_demonstration, self).__init__()
        self.lstm = nn.LSTM(in_dim, out_dim, depth)
    
    def forward(self, inputs, hidden):
        out, hidden = self.lstm(inputs, hidden)
        return out, hidden


torch.manual_seed(29592)  # set the seed for reproducibility

# shape parameters
model_dimension = 8
sequence_length = 20
batch_size = 1
lstm_depth = 1

# random data for input
inputs = torch.randn(sequence_length, batch_size, model_dimension)
# hidden is actually is a tuple of the initial hidden state and the initial cell state
hidden = (torch.randn(lstm_depth, batch_size, model_dimension),
          torch.randn(lstm_depth, batch_size, model_dimension))

# %%

# here is our floating point instance
float_lstm = lstm_for_demonstration(model_dimension, model_dimension, lstm_depth)

# this is the call that does the work
quantized_lstm = torch.quantization.quantize_dynamic(
    float_lstm, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)

# show the changes that were made
print('Here is the floating point version of this module:')
print(float_lstm)
print('')
print('and now the quantized version:')
print(quantized_lstm)


# %%
def print_size_of_model(model, label=""):
    with tempfile.NamedTemporaryFile() as f:
        torch.save(model.state_dict(), f)
        size = os.path.getsize(f.name)
    print("model: ", label, ' \t', 'Size (KB):', size / 1e3)
    return size


# compare the sizes
f = print_size_of_model(float_lstm, "fp32")
q = print_size_of_model(quantized_lstm, "int8")
print("{0:.2f} times smaller".format(f / q))

# %%
import time

start_time = time.time()
out, hid = float_lstm.forward(inputs, hidden)
print("Floating point (FP32) inference time: %s seconds" % (time.time() - start_time))

start_time = time.time()
out, hid = quantized_lstm.forward(inputs, hidden)
print("Quantized (INT8) inference time: %s seconds" % (time.time() - start_time))

#%%
cuda_float_lstm = float_lstm.to('cuda')
cuda_inputs = inputs.to('cuda')
cuda_hidden = [h.to('cuda') for h in hidden]

start_time = time.time()
out, hid = cuda_float_lstm.forward(cuda_inputs, cuda_hidden)
print("Floating point (FP32) inference time on GPU: %s seconds" % (time.time() - start_time))

#%%
cuda_quantized_lstm = quantized_lstm.to('cuda')
cuda_inputs = inputs.to('cuda')
cuda_hidden = [h.to('cuda') for h in hidden]

start_time = time.time()
out, hid = cuda_quantized_lstm.forward(cuda_inputs, cuda_hidden)
print("Quantized (INT8) inference time on GPU: %s seconds" % (time.time() - start_time))
