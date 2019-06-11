import numpy as np
import matplotlib.pylab as plt
import sys

def run():
    visualizeTarget = sys.argv[1]
    print(visualizeTarget)
    
    if(visualizeTarget=='step'):
        x=np.arange(-5.0,5.0,0.1)
        y=step(x)
        plt.plot(x,y)
        plt.ylim(-0.1,1.1)
        plt.show()
    elif(visualizeTarget=='sigmoid'):
        x=np.arange(-5.0,5.0,0.1)
        y=sigmoid(x)
        plt.plot(x,y)
        plt.ylim(-0.1,1.1)
        plt.show()
    elif(visualizeTarget=='relu'):
        x=np.arange(-5.0,5.0,0.1)
        y=relu(x)
        plt.plot(x,y)
        # plt.ylim(-0.1,1.1)
        plt.show()
    elif(visualizeTarget=='all'):
        x=np.arange(-5.0,5.0,0.1)
        y=step(x)
        plt.plot(x,y)
        # plt.ylim(-0.1,1.1)
        x=np.arange(-5.0,5.0,0.1)
        y=sigmoid(x)
        plt.plot(x,y)
        x=np.arange(-5.0,5.0,0.1)
        y=relu(x)
        plt.plot(x,y)
        # plt.ylim(-0.1,3.0)
        plt.show()
# for x in sys.argv:
#     print(x)


class variable():
    def __init__(self, value):
        self.data = value
        pass

    def read(self):
        return self.data

def test():
    v = variable(424)
    print(v.read() == 424)
    a = np.array([2,3,1,4,2])
    print(a)
    print(sigmoid(a))

def TestSimpleANDGate():
    print('simple AND gate test')
    print(SimpleANDGate(0,0))
    print(SimpleANDGate(0,1))
    print(SimpleANDGate(1,0))
    print(SimpleANDGate(1,1))

def SimpleANDGate(x1,x2):
    w1,w2,theta = 0.5,0.5,0.7
    tmp = x1*w1+x2*w2
    if(tmp<=theta): return 0
    elif(tmp>theta): return 1

def TestANDGate():
    print('and gate test')
    print(ANDGate(0,0))
    print(ANDGate(0,1))
    print(ANDGate(1,0))
    print(ANDGate(1,1))

def ANDGate(x1,x2):
    x = np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7
    tmp=np.sum(w*x)+b
    if(tmp<=0): return 0
    else: return 1


def TestNANDGate():
    print('nand gate test')
    print(NANDGate(0,0))
    print(NANDGate(0,1))
    print(NANDGate(1,0))
    print(NANDGate(1,1))

def NANDGate(x1,x2):
    x = np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    b=0.7
    tmp=np.sum(w*x)+b
    if(tmp<=0): return 0
    else: return 1


def TestORGate():
    print('OR gate test')
    print(ORGate(0,0))
    print(ORGate(0,1))
    print(ORGate(1,0))
    print(ORGate(1,1))

def ORGate(x1,x2):
    x = np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.2
    tmp=np.sum(w*x)+b
    if(tmp<=0): return 0
    else: return 1

def XORGate(x1,x2):
    a = ORGate(x1,x2)
    b = NANDGate(x1,x2)
    return ANDGate(a,b)

def step(x):
    y=x>0
    return y.astype(np.int)

def simple_step(value):
    if(value <= 0): return 0
    else: return 1

def sigmoid(value):
    return 1/(1+np.exp(-value))

def relu(x):
    return np.maximum(0,x)

class MultiplyLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x=x
        self.y=y
        out=x*y

        return out

    def backward(self, dout):
        dx=dout*self.y
        dy=dout*self.x
        return dx,dy


def matrixTest1():
    print('mat')
    b = np.array([[1,2],[3,4],[5,6]])
    print(b)
    print(np.ndim(b)) # 배열의 차원 수
    print(b.shape)     # 배열의 형상 (모든 차원의 각 길이)

def matrixMultiplyTest():
    print('multiply')
    a = np.array([[1,2],[3,4]])
    print(a.shape)
    b=np.array([[5,6],[7,8]])
    print(b.shape)
    print(np.dot(a,b))  # 행렬 곱셈

    a = np.array([[1,2],[3,4],[5,6]])
    print(a.shape)
    b=np.array([7,8])
    print(b.shape)
    print(np.dot(a,b))
    x = np.array([1,2])
    W = np.array([[1,3,5],[2,4,6]])
    y = np.dot(x,W)
    print(y)


if(__name__=='main'):    
    # test()

    # TestSimpleANDGate()
    # matrixTest1()
    # matrixMultiplyTest()
    run()
