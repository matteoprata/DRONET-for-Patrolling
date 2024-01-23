import numpy as np
import matplotlib.pyplot as plt

def generate_nodes_on_path(coordinates, N):
    if len(coordinates) < 2 or N < 2:
        raise ValueError("Invalid input: At least two coordinates and N >= 2 are required.")

    # Make the path a closed loop by connecting the last point to the first point
    coordinates.append(coordinates[0])

    path_length = 0
    for i in range(len(coordinates) - 1):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[i + 1]
        path_length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    distance_between_nodes = path_length / (N - 1)
    current_distance = 0
    path_coordinates = [coordinates[0]]
    associated_indices = []  # List to store the index of the next node in the path

    for i in range(len(coordinates) - 1):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[i + 1]
        segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        while current_distance + distance_between_nodes < segment_length:
            ratio = (current_distance + distance_between_nodes) / segment_length
            x = x1 + ratio * (x2 - x1)
            y = y1 + ratio * (y2 - y1)
            path_coordinates.append((x, y))
            associated_indices.append((i + 1) % len(coordinates))  # Store the index of the next node in the loop
            current_distance += distance_between_nodes

        current_distance -= segment_length

    return path_coordinates, associated_indices

# Example usage: Create a circle as the input path
theta = np.linspace(0, 2 * np.pi, 100)
circle_coordinates = [(np.cos(t), np.sin(t)) for t in theta]

number_of_nodes = 20  # Replace with the desired number of nodes

result_nodes, next_node_indices = generate_nodes_on_path(circle_coordinates, number_of_nodes)

# Extract x and y coordinates for plotting
x_circle, y_circle = zip(*circle_coordinates)
x_nodes, y_nodes = zip(*result_nodes)

# Plotting
plt.figure(figsize=(8, 8))
plt.plot(x_circle + (x_circle[0],), y_circle + (y_circle[0],), linestyle='-', color='blue', label='Original Circle')
plt.scatter(x_nodes, y_nodes, color='red', label='Generated Nodes')

# Annotate the indices of the next node in the path
for i, index in enumerate(next_node_indices):
    plt.annotate(index, (x_nodes[i], y_nodes[i]), textcoords="offset points", xytext=(0,5), ha='center')

# Adding labels and legend
plt.title('Nodes along the Circular Path with Associated Indices')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Show the plot
plt.show()
