from clusters import *

# CreateClusterV1("cluster_info_v4", features, 'LBone2000')


rows = ImportSpreadsheet(spreadsheet_v1_path)

kmeans_1, features, cluster_labels, _ = LoadFromJson('/content/drive/MyDrive/Stanford Audiology Models/Colab Notebooks/Cluster_info_.json')

cluster_labels

rows = RenameDuplicateColumns(rows, 'LBone2000')

features = ConvertSpaces(rows[0])

data = MakeDataClass(features, rows)

df = ConvertToPanda(data, features)

# Either run the above three commands or only the command below to do the data import and cleaning
# data = ReadPreprocessData()
# df = ConvertToPanda(data, features)

#With acutal data
df = HLossClassifier(df)
HLPlot(df)



"""
Reading the right ear HL measurements of the following frequencies:
250, 500, 1000, 2000, 3000, 4000, 6000, 8000
"""
hl_right_labels = ['R250',	'R500',	'R1000',	'R2000',	'R3000',	'R4000',	'R6000',	'R8000']
hl_right = df.dropna(subset = hl_right_labels)
hl_right = hl_right[hl_right_labels]

PlotClusterCenters(hl_right_labels,kmeans_1, cluster_labels,kmeans_1.n_clusters)

# #6 clusters
# n = 6
# kmeans_6 = CreateKMeans(n, hl_right)
# columns = df.columns.tolist()

hl_right = KMeansPredictions(kmeans_1, hl_right, cluster_labels)
df['predictions'] = hl_right['predictions']

count = CountPredictions(hl_right, cluster_labels)

df = AssignClusterLabelsToDF(df, cluster_labels)

SaveAsJson(cluster_labels,
           kmeans_1,
           features,
           df.columns.tolist(),
           count,
           "cluster_info_v2",
           '/content/drive/MyDrive/Stanford Audiology Models/Colab Notebooks/')