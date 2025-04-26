import torch
import torch.nn as nn

def accumulation():
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float32)
    print(s)

    s = torch.tensor(0,dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)

    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float32)
    print(s)

    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01,dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print("FC1 output dtype: ", self.fc1(x).dtype)
        x = self.relu(self.fc1(x))
        print('relu dtype: ', x.dtype)
        x = self.ln(x)
        print("LN output dtype: ", x.dtype)
        x = self.fc2(x)
        print("logits dtype: ", x.dtype)
        return x

def benchmarking():
    device = "cuda"
    model = ToyModel(10, 10).to(device)
    dtype = torch.float16
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(1, 10).to(device)

    # use torch autocasting mixed precision
    with torch.amp.autocast(device_type=device, dtype=dtype):
        print("original dtypes: \n")
        for name, parameter in model.named_parameters():
            print(name, parameter.dtype)
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, torch.randn_like(output))
        print("loss dtype: ", loss.dtype)

        loss.backward()
        optimizer.step()

        print("gradient dtypes: \n")
        for name, parameter in model.named_parameters():
            print(name, parameter.grad.dtype)

if __name__ == "__main__":
    benchmarking()