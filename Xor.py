import numpy as np
class Neural:
    def __init__(self,no_of_input,no_of_neuron):
        self.weight_input_hidden = np.random.random((no_of_input,no_of_neuron))
        self.bias_hidden = np.random.random((1,no_of_neuron))
        self.weight_hidden_output = np.random.random((no_of_neuron,1))
        self.bias_output = np.random.random((1,1))

    def threshold(self,x):
        return np.where(x>0.5,1,0)

    def forward(self,x):
        self.y1 = np.dot(x,self.weight_input_hidden) + self.bias_hidden
        self.y1 = self.threshold(self.y1)
        # print(self.y1)
        self.y2 = np.dot(self.y1,self.weight_hidden_output) + self.bias_output
        self.y2 = self.threshold(self.y2)
        return self.y2
    
    def backpropogate(self,x,y):
        error_output = self.y2 - y
        error_hidden = self.weight_hidden_output*error_output   
        self.weight_hidden_output -= lr*error_output*self.y1.T
        self.weight_input_hidden -= lr*error_hidden.T*x
        self.bias_output -=    lr*error_output
        self.bias_hidden -= lr*error_hidden.T
    def train(self,x,y,epoch):
        for _ in range(epoch):
            if _%10000 ==0:
                print(_," Epoch Completed of " , epoch)
            for i in range(len(x)):
                self.forward(x[i])
                self.backpropogate(x[i],y[i])
        
    def test(self,x):
        for i in range(len(x)):
            print(self.forward(x[i]))

lr = 0.01
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
layer1 = Neural(2,2)
layer1.train(x,y,120000)
layer1.test(x)

