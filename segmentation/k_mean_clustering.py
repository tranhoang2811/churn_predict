from sklearn.cluster import KMeans
import pandas as pd
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster.elbow import kelbow_visualizer
import numpy as np
import matplotlib.pyplot as plt

credit_data = pd.read_csv('../dataset/original/extracted.csv')
credit_data.drop(columns='CLIENTNUM', inplace=True)
credit_data.drop(columns='Churn_Label', inplace=True)
model = KMeans()
kelbow_visualizer(model, credit_data, k=(1, 21), timings=False)

# visualizer = KElbowVisualizer(model, k=(1, 21), timings=False)
# visualizer.fit(credit_data)
# visualizer.show()

# distortions = []
#
# for k in range(1, 21):
#     kmeans = KMeans(k)
#     kmeans.fit(credit_data)
#     distortions.append(kmeans.inertia_)
#
# plt.plot(range(1, 21), distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('distortions')
# plt.legend()
# plt.show()
