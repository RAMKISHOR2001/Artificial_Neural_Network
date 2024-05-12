import numpy as np
x = np.array([
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ],
    [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ]
    ,
    [
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1]
    ]
    ,
    [
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ]
    ,
    [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 1],
        [0, 0, 1]

    ]
    ,
    [
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1]

    ]
    ])
output = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])


class Neural:
    def __init__(self,no_of_input,no_of_neuron) -> None:
        self.weight = np.random.random((no_of_input,no_of_neuron))
        self.bias = np.random.random((1,no_of_neuron))
    
    def activation(self,x):
        a =  np.exp(x)
        b = a.sum(axis = 1)
        return a/b
    

    def forward(self,x):
        self.predicted = np.dot(x,self.weight) + self.bias
        self.predicted = self.activation(self.predicted)
        return self.predicted

    def backward(self,x,y):
        error = self.predicted - y
        error  = error*lr
        self.weight = self.weight.T
        for i in range(len(error[0])):
            self.weight[i]  -= error[0][i]*x
        self.weight = self.weight.T
        
    def train(self,x,y,epoch):
        for i in range(epoch):
            for i in range(len(y)):
                self.forward(x[i].flatten())
                self.backward(x[i].flatten(),y[i])
    def test(self,x):
        for i in range(len(x)):
            a = (self.forward(x[i].flatten()))
            print(a)
            print(np.argmax(a))
        
lr = 0.01
model = Neural(15,6)
model.train(x,output,10000)
model.test(x)

