import torch
import numpy as np
import math
import time


class Net(torch.nn.Module):
    def __init__(self):
        self.fc1 = torch.nn.Linear(784, 400)
        self.fc2 = torch.nn.Linear(400, 200)
        self.fc3 = torch.nn.Linear(200, 100)
        self.fc4 = torch.nn.Linear(100, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x

#glbal variable will only init once when called interperter
device = torch.device("cuda")         
model = Net()
check_point = torch.load('model.pt')
model.load_state_dict(check_point)
model.to(device=device)
print('call init')

def inference(args):
    '''
    use model to inference
    '''
    
    print(args.shape)
    args = np.reshape(args, (-1, 784)) #reshape to formed size
    with torch.no_grad():
        test_tensor = torch.from_numpy(args).to(device=device) #turn ndarray to matrix
        inference_result = model(test_tensor) #inference
        result = inference_result.cpu().numpy() #turn to CPU
        return result
