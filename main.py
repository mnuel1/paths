import numpy as np
import cv2
import open3d as o3d
import base64
import matplotlib.pyplot as plt
import heapq

def generate_heightmap(image_path, scale_factor=10, downsample_factor=4):
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

def is_obstacles(color, threshold=0.1, blue=0.9): 
    r, g, b = color
    return r < threshold and g < threshold and b > blue

def is_path(color, threshold=0.1): 
    r, g, b = color
    return r < threshold and g < threshold and b < threshold

def is_corner(color, threshold=0.1, green=0.7, blue=0.8):   
    r, g, b = color
    return r < threshold and g > green and b > blue
def dijkstra(graph, start, end):
    distances = {node: float('inf') for node in graph}
    path_distances = [65.76, 46.5, 46.5, 65.76, 46.5, 46.5, 65.76, 46.5, 46.5, 46.5, 46.5, 46.5, 46.5]
    distances[start] = 0
    priority_queue = [(0, start)]
    previous_nodes = {}
    edge_weights = {}  # Store edge weights in the shortest path

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_node == end:
            path = []
            path_weights = []  # Store weights of edges in the path
            while current_node != start:
                path.append(current_node)
                parent_node = previous_nodes[current_node]
                edge_weight = graph[parent_node][current_node]
                path_weights.append(edge_weight)
                current_node = parent_node
            path.append(start)
            path.reverse()
            path_weights.reverse()
            return distances[end], path, path_weights

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + path_distances[weight]
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                previous_nodes[neighbor] = current_node
                edge_weights[(current_node, neighbor)] = weight

    return float('inf'), None, None


def main(start_node, end_node):
    # image path
    image_path = 'final2.png'

    # z for the  3d dimensions
    # elementImage the image read by cv2
    z, elementImage = generate_heightmap(image_path)

    # Initialize lists to store vertices and edges
    vertices = []
    edges = []

    # Trace the color and identify vertices, edges and obstacles
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            elementsColor = elementImage[i, j] / 255

            if is_path(elementsColor):
                # This is an edge
                edges.append((i, j))
            
            elif is_obstacles(elementsColor):      
                # This is an obstacle
                pass        
            elif is_corner(elementsColor):
                # This is a vertex        
                edges.append((i, j))
                vertices.append((i,j))         
            else: 
                # This is not part of any special element
                pass
    
    # sort the coordinates vertices for easier grouping vertex
    sorted_vertices = sorted(vertices, key=lambda vertex: (vertex[0]))

    # list for the vertices (Group A, B, C, D and so on)
    grouped_vertices = []
    # Group vertices
    current_group = []

    # loop to group the coordinates to get the vertices
    for vertex in sorted_vertices:
        if not current_group:
            current_group.append(vertex)
        else :
        
            prev_vertex = current_group[-1]     
            if vertex[0] == prev_vertex[0]:           
                if vertex[1] == prev_vertex[1] + 1:                     
                    current_group.append(vertex)
                else:                                
                    for group in grouped_vertices:
                        group_prev_vertex = group[-1]                                                 
                        if abs(vertex[0] - group_prev_vertex[0]) <= 1:                                                 
                            if vertex[1] != group_prev_vertex[1]:                                                                                    
                                if vertex[1] - group_prev_vertex[1] <= 400:                                                                
                                    group.append(vertex)
                                
                    grouped_vertices.append(current_group)            
                    current_group = [vertex]                                    
            elif vertex[0] != prev_vertex[0] and abs(vertex[0] - prev_vertex[0] <= 1):
                if abs(vertex[1] - prev_vertex[1]) > 400:
                    
                    for group in grouped_vertices:
                        group_prev_vertex = group[-1]                                                 
                        if abs(vertex[0] - group_prev_vertex[0]) <= 1:                                                 
                            if vertex[1] != group_prev_vertex[1]:                                                                                    
                                if abs(vertex[1] - group_prev_vertex[1]) <= 400:                                                                
                                    group.append(vertex)
                                    break
                else:
                    current_group.append(vertex)
            else:
                grouped_vertices.append(current_group)            
                current_group = [vertex]                                                                           
    # Append the last group
    grouped_vertices.append(current_group)

    # get the average to display
    average_x = []
    average_y = []
    group_number = []   

    # Iterate through grouped vertices and calculate average coordinates
    for idx, group in enumerate(grouped_vertices):
        x_values = [vertex[0] for vertex in group]
        y_values = [vertex[1] for vertex in group]
        avg_x = sum(x_values) / len(group)
        avg_y = sum(y_values) / len(group)
        average_x.append(avg_x)
        average_y.append(avg_y)
        group_number.append(idx)

    
    average_x = list(map(int, average_x))
    average_y = list(map(int, average_y))
    group_number = list(map(int, group_number))

    for x, y, group in zip(average_x, average_y, group_number):
        plt.text(y, x, f"{group}", color='red', fontsize=10)

    # find connections between vertics
    
    def find_connections(average_x, average_y):
        letters = [chr(ord('A') + i) for i in range(26)]
        connections = []
        graph = {letters[i]: [] for i in range(len(average_x))}    
        flag = 0
        index = 0
        for i in range(0, len(average_x)):
            connection = []  

            # checks right diagonal
            if letters[i] == 'A' or letters[i] == 'B':
                diagonal_y = average_y[i]
                for j in range(average_x[i] + 1, z.shape[1]):
                    connection.append([diagonal_y, j])            
                    for number in range(len(average_x)):                                
                        if average_x[number] == j and abs(diagonal_y - average_y[number]) <= 400:                                                                                
                                dist = index
                                index += 1
                                graph[letters[i]].append({letters[number]: dist})
                                graph[letters[number]].append({letters[i]: dist})
                                flag = 1       
                        if flag : break
                    if flag : break 
                    diagonal_y += 1
                if flag == 0 : 
                    connection = [] 
            if connection : 
                connections.append(connection)            
            connection = []  
            flag = 0

            # checks left diagonal
            if letters[i] == 'D':        
                diagonal_y = average_y[i]
                for j in range(average_x[i] - 1, 60, -1):
                    connection.append([j, diagonal_y])                     
                    for number in range(len(average_x)):   
                        if average_y[number] == j and abs(diagonal_y - average_x[number]) <= 400:
                            if letters[i] == 'D':                                                
                                dist = index
                                index += 1
                                graph[letters[i]].append({letters[number]: dist})
                                graph[letters[number]].append({letters[i]: dist})
                                flag = 1
                        
                        if flag : break
                    if flag : break 
                    diagonal_y += 1
            if flag == 0 : 
                connection = [] 
            
            if connection : 
                connections.append(connection) 
            connection = []  
            flag = 0

            # checks right 
            for j in range(average_y[i] + 1, z.shape[0]):              
                connection.append([j, average_x[i]])            
                for number in range(len(average_x)):                                
                    if average_y[number] == j and abs(average_x[i] - average_x[number]) <= 10:                        
                        if letters[i] != 'C' and letters[number] != 'D':                                                
                            dist = index
                            index += 1
                            graph[letters[i]].append({letters[number]: dist})
                            graph[letters[number]].append({letters[i]: dist})
                            flag = 1
                        else:
                            connection = []
                    if flag : break
                if flag : break 
            if flag == 0 : 
                connection = [] 
            
            if connection : 
                connections.append(connection)             
            connection = []  
            flag = 0

            # check vertical connect
            for j in range(average_x[i] + 1, z.shape[1]):            
                for number in range(len(average_x)):
                    connection.append([average_y[i], j])        
                    if average_x[number] == j and abs(average_y[i] - average_y[number]) <= 10:
                       
                        dist = index
                        index += 1
                        graph[letters[i]].append({letters[number]: dist})
                        graph[letters[number]].append({letters[i]: dist})
                                        
                        flag = 1
                    if flag : break
                if flag : break               
            if flag == 0 : 
                connection = [] 
                        
            if connection : 
                connections.append(connection) 
            connection = [] 
            flag = 0       
        return connections, graph

    connections, graph = find_connections(average_x, average_y)
   
    # separate it properly
    new_data = {}
    for key, value in graph.items():
        new_value = {}
        for item in value:
            new_value.update(item)
        new_data[key] = new_value

    # print(new_data)
    shortest_distance, shortest_path, distances = dijkstra(new_data, start_node, end_node)
    # print(distances)
    # if shortest_path:
    #     print(f"Shortest distance from node {start_node} to node {end_node}: {shortest_distance}")
    #     print("Shortest path:", shortest_path)
        
    # else:
    #     print(f"There is no path from node {start_node} to node {end_node}")
  
    # Loop through indices specified in the distances list
    for idx in distances:
        print(idx)
        connected_groups = connections[idx]  # Assuming connections is defined somewhere
        for connected_group in connected_groups:
            elementImage[connected_group[1], connected_group[0]] = [255, 0, 0]  # Red color for vertices
   
    # # Create a window and display the image in full screen mode
    # cv2.namedWindow('Image with vertices highlighted', cv2.WINDOW_NORMAL)
 
    # # Display the image with vertices highlighted
    # cv2.imshow('Image with vertices highlighted', elementImage)
    # cv2.waitKey(0)  # Wait for any key press
    # cv2.destroyAllWindows()
    # Get the encoded image
            
    # Your existing code to generate the encoded image...
    _, buffer = cv2.imencode('.png', elementImage)
    encoded_image_bytes = base64.b64encode(buffer)
    return shortest_path, shortest_distance, encoded_image_bytes
    # return None
# main(start_node='A',end_node='B')
    #  conn_average_x = []
    # conn_average_y = []
  
    # start_node = 'A'
    # end_node = 'B'
    # for idx, group in enumerate(connections):
    #     x_values = [vertex[0] for vertex in group]
    #     y_values = [vertex[1] for vertex in group]
        
    #     if len(group) != 0:
    #         avg_x = sum(x_values) / len(group)
    #         avg_y = sum(y_values) / len(group)
    #         conn_average_x.append(avg_x)
    #         conn_average_y.append(avg_y)
    #         group_number.append(idx)

    # conn_average_x = list(map(int, conn_average_x))
    # conn_average_y = list(map(int, conn_average_y))
    # group_number = list(map(int, group_number))
    # distances = [65.76, 46.5, 46.5, 65.76, 46.5, 46.5, 65.76, 46.5, 46.5, 46.5, 46.5, 46.5, 46.5]
    # index = 0
    # # 11
    # for x, y, group in zip(conn_average_x, conn_average_y, group_number):
    #     plt.text(x, y, f"{distances[index]}", color='red', fontsize=10)
    #     index += 1



    # Set vertices
    # for vertex in vertices:   
    #     elementImage[vertex] = [255, 0, 0]  # Red color for vertices    
        
    # # Set edges
    # for edge in edges:
    #     elementImage[edge] = [0, 255, 0]  # Green color for edges

    # Set start node
    # for start in start_node:
    #     image_with_elements[start] = [0, 0, 255]  # Blue color for start   node
    # 
    # # Set destination node
    # for destination in destination_node:
    #     image_with_elements[destination] = [255, 0, 255]  # Magenta color for destination node
    #
    # Visualize the result
    # plt.imshow(elementImage)
    # plt.show()


