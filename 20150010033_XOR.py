
import numpy as np  

INPUT_NODES = 2       
HIDDEN_NODES = 3   
OUTPUT_NODES = 1
MAX_ITER = 10000
ALPHA = 0.5


X = np.array([[0,0],[1,0],[0,1],[1,1]])
y = np.array([[0],[1],[1],[0]]);
yHat = [None]*X.shape[0]

print ("XOR Neural Learning")

class xor(object):
    def __init__(self): 
        # __init__ constructor method is used for initializing all methods and variables
        self.inputLayerSize = INPUT_NODES
        self.outputLayerSize = OUTPUT_NODES
        self.hiddenLayerSize = HIDDEN_NODES

        # Weights (Parameters)
        self.W1 = np.random.random((self.inputLayerSize, self.hiddenLayerSize))
        self.W2 = np.random.random((self.hiddenLayerSize, self.outputLayerSize))

    def forward(self, X):           			
        self.z2 = np.dot(X.T, self.W1)			
        self.a2 = self.sigmoid(self.z2)			
        self.z3 = np.dot(self.a2, self.W2)		
        yHat = self.sigmoid(self.z3)			
        return yHat

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def sigmoidPrime(self,z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def costFunction(self, X, y):
        self.yHat = self.forward(X)
        J = 0.5*sum((y - self.yHat)**2)
        return J 

    def costFunctionPrime(self, X, y):

        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X, delta2)

        return dJdW1, dJdW2




a = xor()

for i in range(MAX_ITER+1):
    for j in range(X.shape[0]):
        x = np.reshape(X[j], (2, 1))
        Y = np.reshape(y[j], (1, 1))
        yHat[j] = a.forward(x)
        cost1 = a.costFunction(x,Y)
        dJdW1, dJdW2 = a.costFunctionPrime(x,Y)
        a.W1 = a.W1 - ALPHA*dJdW1
        a.W2 = a.W2 - ALPHA*dJdW2
    if(i%1000 == 0):
      print(i, cost1)
        
    
print("Input:")
print(X)
print("Output Predictions:")
print(yHat)
