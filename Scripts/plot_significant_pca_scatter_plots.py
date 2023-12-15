import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import argparse
import os

def get_features(masstime, data, significant=True):
    if not significant:
        return data
    inds=[0,1,2]
    for i in range(3,data.shape[0]):
        mt=data.iloc[i,4] + '@' + data.iloc[i,5]
        if mt in masstime:
            inds.append(i)
    inds=np.unique(np.asarray(inds, dtype=int))
    return data.iloc[inds,:]


def get_ids(data):
    ID=[]
    for line in data.iloc[2,8:]:
        ID.append(line.split('_')[-1].split('.')[0])
    return np.array(ID)


# parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='input file (including path)', required=True)
parser.add_argument('-s', '--significant', help='significant features file (including path)', required=True)
parser.add_argument('-o', '--output_folder', help='output folder (including path). If not given, plot is only shown and not saved.' + \
                                                    'If folder does not exist, it is created', required=False)
parser.add_argument('-t', '--title', help='plot title', required=False)
parser.add_argument('-id', '--includeID', help='include ID in the plot', action='store_true')
parser.add_argument('-f', '--format', help='plot saving format, options: eps (default), tif, jpeg', required=False)
parser.add_argument('-fig', '--figsize', help='Integer indicating the (square) figure size, default is 13(,13)', required=False)

notes = """

The input file should be a csv file with the clean celldata format -- first 3 rows are skipped, and columns 8:48 are kept.\n

The significant features file should be a csv file with results from ttesting. Preferably use files from /PDproj/cellresults/ttest/withGF/only32/...\n

The output folder is optional. If not given, the plot is only shown and not saved. If given, the folder is created if it does not exist.\n
If no title is given, name of the plot will be the same as the significant features file. If no format is given, the default is eps.

The markersize can be changed by changing the s=100 in the ax.scatter commands. The fontsize of the legend can be changed by changing the fontsize=15 in the ax.legend command.\n
But, preferably try changing the plot size with the -fig argument -- this will effectively also change the markersize.

For testing, do not include the -o argument. The plot will then be shown and not saved.\n

"""

parser.epilog = notes

# check if output file is given that the folder exists
# if not create it
if parser.parse_args().output_folder is not None:
    if not os.path.exists(os.path.dirname(parser.parse_args().output_folder)):
        os.makedirs(os.path.dirname(parser.parse_args().output_folder))

filename_data = parser.parse_args().input
filename_significant = parser.parse_args().significant
folder_out = parser.parse_args().output_folder
plot_format = parser.parse_args().format
if parser.parse_args().figsize is None:
    figsize = (13,13)
else:
    figsize = (int(parser.parse_args().figsize), int(parser.parse_args().figsize))

if plot_format is None:
    plot_format = 'eps'

# Parse title
if parser.parse_args().title is None:
    # check if system is using '/' or '\'
    if '\\' in filename_significant:
        tle = filename_significant.split('\\')[-1]
    else:
        tle = filename_significant.split('/')[-1]
    tle = tle.split('ALPHA')[0]
else:
    tle = parser.parse_args().title



# read the data
data = pd.read_csv(filename_data, sep=';', header=None)
df_significance = pd.read_csv(filename_significant, sep=';')

masstime = np.unique(df_significance[' masstime'].to_numpy())
significant_features = get_features(masstime, data, True)
x1 = significant_features.iloc[3:,8:48].to_numpy(dtype=float).T
x_all = data.iloc[3:,8:48].to_numpy(dtype=float).T
ID=get_ids(data.iloc[:,:48])

# print information
print()
print("input file: {}".format(filename_data))
print("significant features file: {}".format(filename_significant))
print("plot title: {}".format(tle))
print("{} significant features".format(len(masstime)))
print()

SAVE = parser.parse_args().output_folder is not None
INCLUDE_ID = parser.parse_args().includeID


pca = PCA().fit_transform(x1)
#pca = add_noise(pca, 1)
assert(x1.shape[0]==40==pca.shape[0])

k=2
fig, ax = plt.subplots(figsize=figsize)
ax.scatter(pca[:10,0],pca[:10,k], marker='^', c='red', s=100)
ax.scatter(pca[10:20,0],pca[10:20,k], marker='o', c='blue', s=100)
ax.scatter(pca[20:30,0],pca[20:30,k], marker='v', c='green', s=100)
ax.scatter(pca[30:,0],pca[30:,k], marker='s', c='gold', s=100)
ax.legend(['aSYN','comb.','IFNg','UT'], fontsize=15)

if INCLUDE_ID:
    for i, txt in enumerate(ID):
        ax.annotate(txt, (pca[i,0], pca[i,k]),fontsize=12)

plt.xlabel('PC 1', fontsize=25)
plt.ylabel('PC {}'.format(k), fontsize=25)
plt.title(tle, fontsize=20)

if SAVE:
    tle1 = tle + '_PCA1vs{}'.format(k)
    filename_output = folder_out + tle1
    print('Saving plot to {}'.format(filename_output))
    # pick the format
    if plot_format == 'eps':
        fig.savefig(filename_output + '.eps')
    elif plot_format == 'tif':
        fig.savefig(filename_output + '.tif')
    elif plot_format == 'jpeg':
        fig.savefig(filename_output + '.jpg')
    else:
        print('Unknown format, use eps, tif or jpeg')
else:
    plt.show()

        
k=3
fig, ax = plt.subplots(figsize=figsize)
ax.scatter(pca[:10,0],pca[:10,k], marker='^', c='red', s=100)
ax.scatter(pca[10:20,0],pca[10:20,k], marker='o', c='blue', s=100)
ax.scatter(pca[20:30,0],pca[20:30,k], marker='v', c='green', s=100)
ax.scatter(pca[30:,0],pca[30:,k], marker='s', c='gold', s=100)
ax.legend(['aSYN','comb.','IFNg','UT'], fontsize=15)

if INCLUDE_ID:
    for i, txt in enumerate(ID):
        ax.annotate(txt, (pca[i,0], pca[i,k]),fontsize=12)
        
plt.xlabel('PC 1', fontsize=25)
plt.ylabel('PC {}'.format(k), fontsize=25)
plt.title(tle, fontsize=20)

if SAVE:
    tle2 = tle + '_PCA1vs{}'.format(k)
    filename_output2 = folder_out + tle2
    print('Saving plot to {}'.format(filename_output2))
    # pick the format
    if plot_format == 'eps':
        fig.savefig(filename_output2 + '.eps')
    elif plot_format == 'tif':
        fig.savefig(filename_output2 + '.tif')
    elif plot_format == 'jpeg':
        fig.savefig(filename_output2 + '.jpg')
    else:
        print('Unknown format, use eps, tif or jpeg')

else:
    plt.show()




