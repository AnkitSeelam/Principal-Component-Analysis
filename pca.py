from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    x = np.load(filename)
    x = x - np.mean(x, axis=0) 
    return x
    


def get_covariance(dataset):
    x_cov = np.transpose(dataset)
    dot = np.dot(x_cov, dataset)
    covariance = dot/(len(dataset) - 1)
    return covariance


def get_eig(S, m):
    eigenValue, eigenVector = eigh(S)
    val = eigenValue.argsort()[::-1]  
    eigenValue = eigenValue[val]
    eigenVector = eigenVector[:,val]
    evalue = eigenValue[0:m]
    eigenVec = eigenVector[:,0:m]
    finalVal = np.diag(evalue)
    return finalVal,eigenVec
    


def get_eig_perc(S, perc):
    eigenValue, eigenVector = eigh(S)
    idx = eigenValue.argsort()[::-1]
    eigenValue = eigenValue[idx]
    eigenVector = eigenVector[:,idx]
    final = []
    for i in eigenValue:
        if((i/sum(eigenValue)) > perc):
            final.append(i)
    eigenVec = eigenVector[:,0:len(final)]
    val = np.diag(final)
    return val,eigenVec

def project_image(img, U):
    vList = U
    proj = np.dot(np.transpose(vList),img)
    final = []
    for i in vList:
        final.append(np.dot(i,proj))
    
    return final


def display_image(orig, proj):
    proj = np.array(proj)
    ro = np.transpose(orig.reshape(32,32))
    rp = np.transpose(proj.reshape(32,32))
    fig, (ax1, ax2) = plt.subplots(ncols = 2)
    oi = ax1.imshow(ro, aspect = "equal")
    ax1.set_title("Original")
    fig.colorbar(oi, ax = ax1)
    pi = ax2.imshow(rp, aspect = "equal")
    ax2.set_title("Projection")
    fig.colorbar(pi, ax = ax2)
    plt.show()

