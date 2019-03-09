#/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import requests

# download file from source and load to dataframe
url="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
filename =requests.get(url).content
df =pd.read_csv(io.StringIO(filename.decode('utf-8')),sep=',', header=None)

# insert headers and print first rows
header = ['Id number', 'Diagnosis'] + [type + ' ' + feature  for type in ['Mean', 'SD', 'Worst']
                                       for feature in ['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness',
                                                       'Compactness', 'Concavity', 'Concavity Points', 'Symmetry', 'Fractal Dimension']]
header = pd.Series(header)
df.columns = header

# Extract features to draw
extracted_features = ['Worst Compactness', 'Worst Concavity', 'Worst Concavity Points', 'Worst Symmetry', 'Worst Fractal Dimension']
Id = ['Diagnosis']
extracted_cols = Id + extracted_features
data = df[extracted_cols]


count_figs = 0
# HEATMAP OF COVARIANCE MATRIX
covariance = round(data.corr(),2)
annot_types = [True, False]
cmap_types = ["coolwarm", "YlGnBu"]
linewidths_ = [None, 1]
for annot in annot_types:
    for cmap in cmap_types:
        for linewidth in linewidths_:
            count_figs+=1
            plt.figure(figsize=(10,10))
            title = 'Heatmap_with_annot_'+ str(annot) +'_cmap_'+cmap+'_linewidth_'+str(linewidth)
            sns.heatmap(covariance, annot=annot, cmap=cmap,fmt='.2f', linewidths=linewidth)
            plt.title(title)
            plt.savefig(title+'.png')
            print('saved figure number ', count_figs)
            plt.close(fig=None)


# PAIRPLOT
x_vars = ["Worst Compactness", "Worst Concavity"]
y_vars = extracted_features
kind_ = ['reg', 'scatter']
diag_kind_ = ['kde', 'hist']
hue_ = ['Diagnosis', None]
markers_ = ['o', 's', 'D']

for kind in kind_:
    for diag_kind in diag_kind_:
        for hue in hue_:
            for marker in markers_:
                plt.figure(figsize=(10,10))
                title = 'Pairplot_with_kind_' + str(kind)+ '_diag_kind_'+ str(kind) + '_hue_'+ str(hue) +'_marker_' +str(marker)
                pp = sns.pairplot(data, kind=kind, diag_kind= diag_kind, hue=hue, markers=marker)
                pp.fig.suptitle(title)
                plt.savefig(title+'.png')
                count_figs+=1
                print('saved figure number ', count_figs)
                plt.close(fig=None)

features = ['Worst Compactness', 'Worst Concavity', 'Worst Concavity Points', 'Worst Symmetry', 'Worst Fractal Dimension']
data_casted = pd.melt(data,id_vars="Diagnosis", var_name="features", value_name='value')

# BOXPLOT

count_figs+=1
plt.figure(figsize=(10,10))
sns.boxplot(x="features", y="value", hue="Diagnosis", data=data_casted)
plt.savefig('Boxplot.png')
plt.title('Boxplot')
plt.close(fig=None)

# VIOLINPLOT

bw_ = ["scott", "silverman"]
scales_ = ['area', 'count', 'width']
inner_ = ["box", "quartile", "point"]
for bw in bw_:
    for scale in scales_:
        for inner in inner_:
            count_figs+=1
            title = 'Violinplot_with_bw_' +str(bw)+'_scale_'+ str(scale)+'_inner_'+ str(inner)
            plt.figure(figsize=(10,10))
            sns.violinplot(x="features", y="value", hue="Diagnosis", bw=bw, scale=scale, data=data_casted,split=True, inner=inner)
            plt.title(title)
            plt.xticks(rotation=90)
            plt.savefig(title+'.png')
            print('saved figure number ', count_figs)
            plt.close(fig=None)



# YELLOWBRICK
from yellowbrick.features.rankd import Rank2D
from yellowbrick.features.pcoords import ParallelCoordinates

labels = ['M', 'B']
X = data[features].as_matrix()
y = data.Diagnosis

visualizer = Rank2D(features=features, algorithm='covariance')
visualizer.fit(X, y)
visualizer.transform(X)
visualizer.poof(outpath = 'Rank2D_figure_with_algo_=_covariance.png')
count_figs+=1
print('saved figure number ', count_figs)

visualizer = Rank2D(features=features, algorithm='pearson')
visualizer.fit(X, y)
visualizer.transform(X)
visualizer.poof(outpath = 'Rank2D_figure_with algo_=_Pearson')
count_figs+=1
print('saved figure number ', count_figs)

visualizer = ParallelCoordinates(classes=labels, features=features)
visualizer.fit(X, y)
visualizer.transform(X)
visualizer.poof(outpath= 'Parallel_Coordinates.png')
count_figs+=1
print('saved figure number ', count_figs)
