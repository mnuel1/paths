import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from queue import PriorityQueue

def generate_heightmap(image_path, scale_factor=10):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Get dimensions of the image
    height, width, _ = img.shape

    # Scale the z-coordinate based on image intensity
    z = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255 * scale_factor

    return z, img

def is_user(color, threshold=0.1, red=0.9):  
    r, g, b = color    
    return r > red and g < threshold and b < threshold

def is_door(color, threshold=0.1, green=0.9):  
    r, g, b = color    
    return r < threshold and g > green and b < threshold

def is_obstacles(color, threshold=0.1,blue=0.9): 
    r, g, b = color
    return r < threshold and g < threshold and b > blue

def is_path(color, threshold=0.1): 
    r, g, b = color
    return r < threshold and g < threshold and b < threshold
def is_corner(color, threshold=0.1, green=0.8,blue=0.9):   
    r, g, b = color
    return r < threshold and g > green and b > blue
# Perform Dijkstra's algorithm to find the shortest path
def dijkstra(graph, start, end):
    queue = PriorityQueue()
    queue.put((0, start))
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    path = {}

    while not queue.empty():
        current_distance, current_node = queue.get()

        if current_node == end:
            break

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                queue.put((distance, neighbor))
                path[neighbor] = current_node

    shortest_path = []
    current_node = end
    while current_node != start:
        shortest_path.append(current_node)
        current_node = path[current_node]
    shortest_path.append(start)
    shortest_path.reverse()

    return shortest_path

# Example usage
image_path = 'final.png'
# original_image ='final.png'
z, elementImage = generate_heightmap(image_path)
# original_img = cv2.imread(original_image, cv2.IMREAD_COLOR)

# path_color = np.array([0, 0, 0])  # Black color for path
# bg_color = np.array([1, 1, 1])
graph = {}

# Create a point cloud
points = []
colors = []

# Trace the color and create points and colors
for i in range(z.shape[0]):
    for j in range(z.shape[1]):
        # for path and door 
        elementsColor = elementImage[i, j] / 255
        # for users and general
        # color = original_img[i, j] / 255

        if is_path(elementsColor):
            points.append([i, j, z[i, j]])
            colors.append(elementsColor)            
        elif is_obstacles(elementsColor):      
            points.append([i, j, z[i, j]])
            colors.append(elementsColor)       
        elif is_corner(elementsColor):
            points.append([i, j, z[i, j]])
            colors.append(elementsColor)  
        elif is_door(elementsColor):
            points.append([i, j, z[i, j]])
            colors.append(elementsColor)
        elif is_user(elementsColor): 
            points.append([i, j, z[i, j]])
            colors.append(elementsColor)
        else: 
            points.append([i, j, z[i, j]])
            colors.append(elementsColor)

# Convert lists to NumPy arrays
points = np.array(points)
colors = np.array(colors)

# Create point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Show the point cloud and start interactive editing
o3d.visualization.draw_geometries_with_editing([point_cloud])   
{(120, 113): [((1023, 118), 2)], (113, 120): [((1023, 1023), 2)], (118, 1023): [((1023, 1023), 90), ((1023, 1023), 5)], (1023, 118): [((1928, 1928), 51)], (1023, 1023): [((1928, 1023), 9), ((1928, 1928), 2)], (1023, 1928): [((1928, 1928), 5), ((1932, 1928), 52)], (118, 1928): [((1023, 1928), 88)]}