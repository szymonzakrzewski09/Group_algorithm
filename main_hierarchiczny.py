import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def remove_blank_lines_in_files(files: str):
    with open(files, 'r+') as file:
        for line in file:
            if not line.isspace():
                file.write(line)


def read_dataset(files: str = 'dataset.txt', attributes: str = 'dataset-type.txt'):
    remove_blank_lines_in_files(files)
    data_frame_attributes = pd.read_csv(attributes, header=None, sep='\t')
    data_frame = pd.read_csv(files, header=None, sep=' ', names=data_frame_attributes[0], skipinitialspace=True)
    return data_frame


def euc_distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def create_distance_matrix(groups):
    num_groups = len(groups)
    distance_matrix = np.zeros((num_groups, num_groups))
    for i in range(num_groups):
        for j in range(num_groups):
            distance_matrix[i, j] = euc_distance(np.mean(groups[i], axis=0), np.mean(groups[j], axis=0))

    return distance_matrix


def find_min_value_index_exclude_diagonal(array):

    min_value = float('inf')

    min_index = None

    for i in range(len(array)):
        for j in range(len(array[i])):
            if i != j:
                if array[i][j] < min_value:
                    min_value = array[i][j]
                    min_index = [i, j]


    return min_index


def hierarchical_clustering(X, num_clusters):
    groups = [[x] for x in X]

    while len(groups) > num_clusters:
        distances = create_distance_matrix(groups)
        closest_pair = find_min_value_index_exclude_diagonal(distances)
        group_a, group_b = groups[closest_pair[0]], groups[closest_pair[1]]

        print(group_a)
        print(group_b)

        groups = [group for group in groups if not np.array_equal(group, group_a)]
        groups = [group for group in groups if not np.array_equal(group, group_b)]

        merged_group = group_a + group_b

        groups.append(merged_group)

        print(groups)

    return groups



X = read_dataset()
X = X.to_numpy()
num_clusters = 3
clusters = hierarchical_clustering(X, num_clusters)

for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}:")
    for sample in cluster:
        print(sample)
    print()


color = ["#" + ''.join([random.choice('0123456789ABCDEF') for i in range(6)])
         for j in range(num_clusters)]

for data in X:
    for i, cluster in enumerate(clusters):
        for sample in cluster:
            print(data)
            print(sample)
            if data[0] == sample[0] and data[1] == sample[1]:
                plt.plot(sample[0], sample[1], marker="o", color=color[i])

legend_group = []
for x in range(num_clusters):
    legend_group.append(mpatches.Patch(color=color[x], label='cluster ' + str(x)))

plt.legend(
    loc=1, prop={'size': 6},
    handles=legend_group)
plt.show()
