import numpy as np
A = np.array([[0.5,0.1,0.4],[0.3,0.5,0.2],[0.2,0.2,0.6]])
B = np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])
pi = np.array([0.2,0.3,0.5])
y = np.array([0,1,0,0,1,0,1,1])#0代表红，1代表白
T = 8

def forward(A, B, pi, y, t, T):#前向算法
    alpha = np.zeros((T, np.shape(A)[0]))
    alpha[0] = pi * B[:, y[0]]
    for i in range(1,T):
        alpha[i] = np.matmul(alpha[i-1] , A) * B[:,y[i]]
    return alpha[t-1],alpha

def backward(A, B, pi, y, t, T):#后向算法
    beta = np.zeros((T, np.shape(A)[0]))
    beta[T-1] = np.array([1,1,1])
    for i in range(T-2, -1, -1):
        beta[i] = np.matmul(A, B[:,y[i+1]]) * beta[i+1]
    return beta[t-1],beta

def Viterbi(A, B, pi, y, T):#维比特算法
    alpha = np.zeros((T, np.shape(A)[0]))
    phi = np.zeros((T, np.shape(A)[0]))
    it = np.zeros(T)
    alpha[0] = pi * B[:, y[0]]
    phi[0] = 0
    for i in range(1,T):
        alpha[i] = np.max(alpha[i-1].T * A, 1) * B[:,y[i]]
        phi[i] = np.argmax(alpha[i-1].T * A, 1)
    P = np.max(alpha[T-1])
    it[T-1] = np.argmax(alpha[T-1])
    for i in range(T-2, -1, -1):
        it[i] = phi[i+1, int(it[i+1])]
    return it

[alpha_i, alpha] = forward(A,B,pi,y,4,T)
[beta_i, beta] = backward(A,B,pi,y,4,T)
gamma = np.zeros((T,np.shape(A)[0]))
gamma = alpha * beta / np.resize(np.sum(alpha * beta, 1),(T,1))

print('T=4,状态为3的概率:',gamma[3,2])

it = Viterbi(A, B, pi, y,8)
print('最优路径：',it)