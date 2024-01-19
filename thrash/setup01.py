import numpy as np
import matplotlib.pyplot as plt


def generate_2d_smile_gaussian(width, epicenter_position, sigma):
    x, y = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, width-1, width))
    d = np.sqrt((x - epicenter_position[0])**2 + (y - epicenter_position[1])**2)
    values = np.exp(-(d**2) / (2 * sigma**2))
    return values

# Set parameters
WIDTH = 100
EPICENTER_POSITION = (50, 50)  # Center of the area
SIGMA = 20  # Adjust this parameter to control the spread of the values

# Generate 2D smile Gaussian distribution
smile_values = generate_2d_smile_gaussian(WIDTH, EPICENTER_POSITION, SIGMA)

# Access values using values[x][y]
for i in range(0, 40):
    x_coord = 50+i
    y_coord = 50
    print(f"Value at ({x_coord}, {y_coord}): {smile_values[x_coord][y_coord]}")

# Plot the result
plt.imshow(smile_values, cmap='viridis', extent=[0, WIDTH-1, 0, WIDTH-1], origin='lower')
plt.title('2D Smile Gaussian Distribution')
plt.colorbar(label='Values')
plt.scatter(*EPICENTER_POSITION, color='red', marker='x', label='Epicenter (Peak)')
plt.legend()
plt.show()
