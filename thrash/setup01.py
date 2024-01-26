# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# np.random.seed(45)
# # Create a grid
# x, y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
#
# # Create a 2D Gaussian distribution for the heatmap
# sigma = 2.0
# gaussian = np.exp(-(x**2 + y**2) / (2 * 2*sigma**2))
#
# # Add 6 epicenters
# epicenters = [(24, 24), (60, 40), (80, 80), (20, 84), (40, 64)]
#
# ALL_cluster_points=[]
# # Add little points around each epicenter
# for epicenter in epicenters:
#     cluster_points = np.random.normal(loc=epicenter, scale=6, size=(8, 2))
#     for point in cluster_points:
#         distance = np.sqrt((x - point[0])**2 + (y - point[1])**2)
#         gaussian += np.exp(-distance**2)
#     ALL_cluster_points.append(cluster_points)
#
# plt.figure(figsize=(4, 4))
# # Plot the heatmap
# sns.heatmap(gaussian, cmap='inferno', center=np.min(gaussian), annot=False, alpha=.7, cbar=False, square=True, xticklabels=False, yticklabels=False)
#
# # Plot epicenters as red dots
# epicenters_x, epicenters_y = zip(*epicenters)
#
# plt.scatter(epicenters_x, epicenters_y, color='red', marker='o', label='Epicenters')
# i = 0
# for a in ALL_cluster_points:
#     po_x, po_y = zip(*a)
#     i += 1
#     st = 'IPs' if i == 1 else None
#     plt.scatter(po_x, po_y, color='blue', marker='x', label=st)
#
# # Plot points around each epicenter as crosses
# # for epicenter in epicenters:
# #     plt.plot(epicenter[0], epicenter[1], color='white', marker='x')
#
# # Show the legend
# plt.legend(fontsize=12)
#
# # Show the plot
# plt.tight_layout()
# plt.savefig("../data/imgs/epicenters1.pdf")
# plt.show()


import matplotlib.pyplot as plt

# Function segments
X = [1, 2, 3, 4, 100, 201]
Y = [1, 1, 1, 1, 1, 1]

# Plot the function segments
plt.plot(X, Y, marker='o', linestyle='-', color='blue')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Function Defined in Segments')

# Show the plot
plt.show()