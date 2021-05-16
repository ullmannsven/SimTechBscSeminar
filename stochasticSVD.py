# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:38:24 2021

@author: ullma
"""

import numpy as np 
from matplotlib.image import imread
import matplotlib.pyplot as plt
import scipy as sp
from scipy.sparse.linalg import svds


#Im Folgenden soll ein Bild einmal mithilfe "normaler SVD" approximiert werden und 
# anschließend mit Hilfe stochastischer SVD. Dabei soll die verwendete Rechenzeit verglichen werden. 

image = imread("Albert_Einstein.jpg")

#A = image[:,:,1]
A = image.astype(np.float)

m, n = A.shape 
print(m,n)

#parameter setting

k = 30 #Anzahl an singulärwerten für unsere Approximation
i = 0 #anzahl an power iterations, sollte <= 3 sein, sonst gehen die Vorteile des stochastischen SVD kaputt
l = 0#oversamling parameter, typische werte sind 2, 5, 10 je nach größe der Matrix 


#Normale SVD/ approximationmit 50 singulärwerten 
def standardSVD(A):
    u, sigma, vT = sp.linalg.svd(A)
    plt.imshow(A,'gray')
    plt.title('Orginalbild')
    plt.axis('off')
    plt.show()
    
    #scipy implementation
    u_sp, s_sp, vT_sp = svds(A, k)
    s_new = np.sort(s_sp)
    print(np.linalg.matrix_rank(u_sp@ np.diag(s_new)@vT_sp))
    plt.imshow(u_sp@ np.diag(s_new)@vT_sp, 'gray')
    plt.title('Rank %d Approximation' %k)
    plt.axis('off')
    plt.show()
    
    # plt.imshow(u[:,:k] @ np.diag(sigma[:k]) @vT[:k,:],'gray')
    # plt.title('normal komprimiertes Bild, %d Singulärwerten' %k)
    # plt.axis('off')
    # plt.show()

standardSVD(A)




#######################################

def stochasticSVD(A,i,l,k):
    
    Omega = np.random.randn(A.shape[1],k+l) #+l: oversampling
    Y = A @ Omega
    
    #ohne stack
    for q in range(i): #power iteration
        Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y,mode='reduced')
    print(np.linalg.norm(Q@Q.T-np.eye(m)))
    
    #stack methode
    # H = Y
    # for q in range(i): #power iteration
    #     Y = A @ (A.T @ Y)
    #     H = np.hstack((H,Y))  
    # Q, _ = np.linalg.qr(H,mode='reduced') #extrem wichtig das reduced statt complete ausgewählt wird
    
    
    T = A.T @ Q
    v_tilde, s_tilde, wT = np.linalg.svd(T, full_matrices = 0)
    u_tilde = Q @ wT.T
    return u_tilde[:m,:k], s_tilde[:k], v_tilde[:n,:k]


def plotStochasticSVD(A,i,l,k): 
    u, sigma, v = stochasticSVD(A,i,l,k) #v noch nich transponiert
    vT = v.T
    #print(np.linalg.matrix_rank(u @ np.diag(sigma) @ vT))
    plt.imshow(u @ np.diag(sigma) @ vT, 'gray')
    plt.title('Stochastische Rank %d Approximation, i=3' %k)
    #plt.title('Oversampling %d' %l)
    plt.axis('off')
    plt.show()
    
#plotStochasticSVD(A, i, 10, k)
plotStochasticSVD(A, 3, l, k)



# define SVD for testing area
def normalSVD(A):
     u, sigma, vT = np.linalg.svd(A)
     return u, sigma, vT

def truncatedSVD(A,k):
    u, sigma, vT = svds(A,k)
    return u, sigma, vT
    

#testing area 
# def compareFunctions():
#     import timeit
    
#     setup1 = """
# from __main__ import normalSVD
# from __main__ import A
# """
#     setup2 = """
# from __main__ import stochasticSVD
# from __main__ import A, i, l, k
# """

#     setup3 = """
# from __main__ import truncatedSVD
# from __main__ import A,k
# """

#     stmt1 ="normalSVD(A)"
#     stmt2 ="stochasticSVD(A,i,l,k)"
#     stmt3 ="truncatedSVD(A,k)"
    
#     times1 = timeit.repeat(stmt=stmt1, setup=setup1, number=1, repeat=10)
#     times2 = timeit.repeat(stmt=stmt2, setup=setup2, number=1, repeat=10)
#     times3 = timeit.repeat(stmt=stmt3, setup=setup3, number=1, repeat=10)
#     print(f"Time taken by normal SVD is {min(times1)} seconds")
#     print(f"Time taken by stochastic SVD is {min(times2)} seconds")
#     print(f"Time taken by truncated SVD ist {min(times3)} seconds")

# compareFunctions()
    


    
    
    
    
    
    
    
    
    
    



