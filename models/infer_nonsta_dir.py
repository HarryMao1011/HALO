## test the kernel embedding the gram matrix
import numpy as np
from .ScoreUtils import kernel, pdinv
import numpy.matlib

def infer_nonsta_dir(X, Y, c_indx, width=0.1, IF_GP=False):
    """learn the nonstationary driving force of the causal mechanism
    X: parents; Y; effect
    width: the kernel width for X and Y
    c_indx: surrogate variable to capture the distribution shift; 
    If If_GP = True (TODO), learning the kernel width for P(Y|X). Set it to False can speed up the process!!!
    """
    width = 0.1
    T,d = X.shape
    ## normalization
    X = X - np.matlib.repmat( np.mean(X,0), X.shape[0], 1)
    X = np.matmul(X, np.diag(1./np.std(X, axis=0, ddof=1)))
    ## normalization
    Y = Y - np.matlib.repmat( np.mean(Y,0), Y.shape[0], 1)
    Y = np.matmul(Y, np.diag(1/np.std(Y, axis=0,  ddof=1)))
    theta = 1/width**2
    Lambda = 1
    Wt = 1
    theta = 1/width**2
    Kxx, _ = kernel(X,X, [theta,1])
    Kxx , _ = kernel(X, X, [theta,1])
    Kyy, _ = kernel(Y, Y, [theta,1])
    Ktt, _ = kernel(c_indx, c_indx, [1/Wt**2,1])


    invK = pdinv( np.multiply(Kxx,Ktt) +  Lambda * np.eye(T))

    Kxx3 = np.linalg.matrix_power(Kxx,3)
    prod_invK =  np.matmul(np.matmul(invK, Kyy), invK)
    ###now finding Ml
    Ml = np.matmul(np.matmul(1/T**2*Ktt, np.multiply(Kxx3, prod_invK)), Ktt)
    ###the square distance
    D = np.matmul(np.diag(np.diagonal(Ml)),np.ones(Ml.shape)) + np.matmul(np.ones(Ml.shape), np.diag(np.diagonal(Ml))) - 2*Ml

    DD = D[np.where(np.tril(np.ones(D.shape),-1) != 0)]
    sigma2_square = np.median(np.array(DD))
    Mg = np.exp(-D/sigma2_square/2)


    invK2 = pdinv(Ktt +  Lambda * np.eye(T))
    Ml2 = np.matmul(Ktt, invK2)
    Ml2 = np.matmul(Ml2, Kxx)
    Ml2 = np.matmul(Ml2, invK2)
    Ml2 = np.matmul(Ml2, Ktt)
    D2 =np.matmul(np.diag(np.diagonal(Ml2)),np.ones(Ml2.shape)) + np.matmul(np.ones(Ml2.shape), np.diag(np.diagonal(Ml2))) - 2*Ml2
    # % Gaussian kernel
    DD2 = D2[np.where(np.tril(np.ones(D2.shape),-1) != 0)]
    sigma2_square2 = np.median(np.array(DD2))

    # sigma2_square2 = median( D2(find(tril(ones(size(D2)),-1))) );
    Mg2 = np.exp(-D2/sigma2_square2/2)

    # %% 
    H = np.eye(T)-1/T*np.ones([T,T])
    Mg = np.matmul(np.matmul(H, Mg), H)
    Mg2 = np.matmul(np.matmul(H, Mg2), H)
    testStat = 1/T**2*np.sum(np.sum(np.multiply(Mg, Mg2)))


    return testStat, Mg, Mg2
        