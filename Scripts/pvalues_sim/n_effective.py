import python as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def n_eff_spectral(C, alpha = 0.05, method='1'):
    """
    Calculate the effective number of tests from a spectral decomposition
    of the correlation matrix. The function also returns Sidak and Bonferroni 
    corrected significance level.

    methods:
    1. https://www.nature.com/articles/6889010
    2. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1181954/
    3. https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-021-03975-2
    
    Parameters:
    C: numpy array of shape (features, features)

    """
    # Get the eigenvalues of the correlation matrix
    eigvals = np.linalg.eigvals(C)
    M = len(eigvals)
    # Calculate the effective number of tests
    if method == '1':
        M_eff = M*(1 - (M - 1)*np.var(eigvals) / M**2)
    elif method == '2':
        M_eff = 1 + (M - 1)*(1 - np.var(eigvals) / M)
    elif method == '3':
        num = (np.sqrt(eigvals).sum()/np.log(eigvals.max()))**2
        den = eigals.sum()/eigvals.max() + np.sqrt(eigvals.max())
        M_eff = num/den
    else:
        raise ValueError('Invalid method')

    return M_eff, 1 - (1 - alpha)**(1/M_eff), alpha/M_eff

def n_eff_cluster(X, threshold = 0.1, mask = None, method = 'complete', metric = 'correlation', plot = False, return_labels = False):
    """
    Calculate the effective number of tests by performing hierarchical clustering.
    The function also returns Sidak and Bonferroni corrected significance level.

    Parameters:
    X: numpy array of features (obs, features) if metric = 'correlation'
    or otherwise a distance/kernel matrix (features, features)
    """
    if metric == 'correlation':
        # Calculate the correlation matrix
        COR = np.corrcoef(X.T, rowvar=False)
        CD = 1 - COR
    else:
        CD = X
    
    # Calculate the distance matrix
    D = squareform(pdist(CD, 'euclidean'))
    # Perform hierarchical clustering
    Z = hierarchy.linkage(D, method='complete')
    # Get the clusters
    clusters = hierarchy.fcluster(Z, t=threshold, criterion='distance')
    # Get the number of clusters
    M_eff = len(np.unique(clusters))
    
    if plot:
        hierarchy.dendrogram(Z, truncate_mode='none', labels=np.asarray(np.arange(CD.shape[0])))
        plt.axhline(y=th, c='grey', lw=1, linestyle='dashed')
        plt.yticks(np.arange(0,2.1,0.2))
        plt.title("{} has {} clusters with threshold of {}".format(tle, np.unique(labels_complete).shape[0], th))
        plt.show()
    
    if return_labels:
        return M_eff, 1 - (1 - threshold)**(1/M_eff), threshold/M_eff, clusters

    return M_eff, 1 - (1 - threshold)**(1/M_eff), threshold/M_eff
