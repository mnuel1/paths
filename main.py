import numpy as np
import cv2
import base64
import matplotlib.pyplot as plt
import heapq

def cv2_process_image(image_path, binary_data):
    # Load the image
    img = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    obs = cv2.imdecode(np.frombuffer(binary_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Convert BGR to RGB
    obs_rgb = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB) 

    height, width, _ = image_rgb.shape
    return height, width, image_rgb, obs_rgb

def is_obstacles(color, threshold=0.4, red=0.7): 
    r, g, b = color
    return r > red and g < threshold and b < threshold

def is_corner(color, threshold=0.1, green=0.8, red=0.8):   
    r, g, b = color
    return r > red and g > green and b < threshold

def dijkstra(graph, start, end):
    distances = {node: float('inf') for node in graph}
    path_distances = [32.88, 23.25, 23.25, 23.25, 32.88, 23.25, 23.25, 32.88, 23.25, 32.88, 23.25, 32.88, 23.25, 
                      23.25, 23.25, 23.25, 23.25, 32.88, 23.25, 23.25, 23.25, 23.25, 23.25, 23.25]
    distances[start] = 0
    priority_queue = [(0, start)]
    previous_nodes = {}

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_node == end:
            path = []
            while current_node != start:
                path.append(current_node)
                current_node = previous_nodes[current_node]
            path.append(start)
            path.reverse()
            return distances[end], path

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + path_distances[weight]
            print(distance)
            if distance < distances[neighbor]:
                print(weight)
                distances[neighbor] = distance                
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    return float('inf'), None

def check_right_diagonal(i, average_coordinate, letters, graph, index, length, size, obstacles, obs_array) :
    diagonal_y = average_coordinate[i][1]    
    connection = []  
    for j in range(average_coordinate[i][0] + 1, size):
        if (j, diagonal_y) in obstacles:    
            obs_array.append(index)
            index += 1
            break 
        connection.append([diagonal_y, j])
        for number in range(length):
            if average_coordinate[number][0] == j and abs(diagonal_y - average_coordinate[number][1]) <= 100:
                graph[letters[i]][letters[number]] = index
                graph[letters[number]][letters[i]] = index  
                index += 1
                return connection if connection else [], index
        diagonal_y += 1

    return None, index
def check_left_diagonal(i, average_coordinate, letters, graph, index, length, obstacles, obs_array) :
    diagonal_y = average_coordinate[i][0]    
    connection = [] 
    for j in range(average_coordinate[i][1] - 1, 20, -1):
        if (diagonal_y, j) in obstacles:
            obs_array.append(index)
            index += 1
            break                   
        connection.append([j, diagonal_y]) 
        for number in range(length):   
            if average_coordinate[number][1] == j and abs(diagonal_y - average_coordinate[number][0]) <= 100:                                                        
                graph[letters[i]][letters[number]] = index
                graph[letters[number]][letters[i]] = index                       
                index += 1
                return connection if connection else [], index
        diagonal_y += 1

    return None, index
def check_right(i, average_coordinate, letters, graph, index, length, size, obstacles, obs_array) :    
    connection = []
    for j in range(average_coordinate[i][1] + 1, size):
        if (average_coordinate[i][0], j) in obstacles:
            obs_array.append(index)
            index += 1           
            break     
        connection.append([j, average_coordinate[i][0]]) 
        for number in range(length):                                
            if average_coordinate[number][1] == j and abs(average_coordinate[i][0] - average_coordinate[number][0]) <= 100:                                                        
                graph[letters[i]][letters[number]] = index
                graph[letters[number]][letters[i]] = index    
                index += 1
                return connection if connection else [], index
    return None, index
def check_down(i, average_coordinate, letters, graph, index, length, size, obstacles, obs_array) :
    connection = [] 
    for j in range(average_coordinate[i][0] + 1, size):
        if (j, average_coordinate[i][1]) in obstacles:
            obs_array.append(index)
            index += 1
            break 
        for number in range(length):
            connection.append([average_coordinate[i][1], j]) 
            if average_coordinate[number][0] == j and abs(average_coordinate[i][1] - average_coordinate[number][1]) <= 100:                            
                graph[letters[i]][letters[number]] = index
                graph[letters[number]][letters[i]] = index 
                index += 1
                return connection if connection else [], index
    return None, index
        
# find connections between vertics    
def find_connections(average_coordinate, letters, obstacles, size):
    
    connections = []
    obs_array = []
    ave_coor_length = len(average_coordinate)
    graph = {letters[i]: {} for i in range(ave_coor_length)}
    
    index = 0
    for i in range(0, ave_coor_length):
        
        # checks right diagonal
        if letters[i] in {'A', 'C', 'E', 'G'}:
            conn, index = check_right_diagonal(i, average_coordinate, letters, graph, index, ave_coor_length, size, obstacles, obs_array)
        if conn : connections.append(conn)

        # checks left diagonal
        if letters[i] in {'I', 'M'}:
            conn, index  = check_left_diagonal(i, average_coordinate, letters, graph, index, ave_coor_length, obstacles, obs_array)
        if conn : connections.append(conn)

        # checks right
        if letters[i] not in {'D', 'E', 'F', 'H', 'L', 'M', 'N'}:
            conn, index  = check_right(i, average_coordinate, letters, graph, index, ave_coor_length, size, obstacles, obs_array)
        if conn : connections.append(conn)

        # checks down
        if letters[i] not in {'B', 'E', 'G', 'J', 'M'}:
            conn, index  = check_down(i, average_coordinate, letters, graph, index, ave_coor_length, size , obstacles, obs_array)
        if conn : connections.append(conn)
        
    # print(graph)
    return graph, obs_array

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        # Read the image file
        image_data = image_file.read()
        # Encode the image data as Base64
        base64_encoded = base64.b64encode(image_data).decode("utf-8")
    return base64_encoded

def get_graph(obs_base64):
    
    image_path = 'final.png'
    binary_data = base64.b64decode(obs_base64)

    height, width, map_semantics, obs_element = cv2_process_image(image_path, binary_data)
 
    vertices = []    
    obstacles = []

    # Trace the color and identify vertices and obstacles
    for i in range(width):
        for j in range(height):
            semantics = map_semantics[i, j] / 255
            obs = obs_element[i, j] / 255
            if is_obstacles(obs):
                # This is an obstacle
                obstacles.append((i, j))
            elif is_corner(semantics):
                # This is a vertex     
                vertices.append((i, j))

    # sort the coordinates vertices for easier grouping vertex
    sorted_vertices = sorted(vertices, key=lambda vertex: (vertex[0]))

    grouped_vertices = []    
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
                        if abs(vertex[0] - group_prev_vertex[0]) <= 1 and vertex[1] != group_prev_vertex[1] and vertex[1] - group_prev_vertex[1] <= 60 :                         
                            group.append(vertex)
                    grouped_vertices.append(current_group)
                    current_group = [vertex]
            elif vertex[0] != prev_vertex[0] and abs(vertex[0] - prev_vertex[0] <= 1):
                if abs(vertex[1] - prev_vertex[1]) > 60:
                    for group in grouped_vertices:
                        group_prev_vertex = group[-1]
                        if abs(vertex[0] - group_prev_vertex[0]) <= 1 and vertex[1] != group_prev_vertex[1] and abs(vertex[1] - group_prev_vertex[1]) <= 60:
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
    average_coordinate = []    

    # Iterate through grouped vertices and calculate average coordinates
    for idx, group in enumerate(grouped_vertices):
        x_values = [vertex[0] for vertex in group]
        y_values = [vertex[1] for vertex in group]
        avg_x = int(sum(x_values) / len(group))
        avg_y = int(sum(y_values) / len(group))
        average_coordinate.append((avg_x, avg_y))
 
    letters = [chr(ord('A') + i) for i in range(26)]

    # for index, (x, y) in enumerate(average_coordinate):
    #     plt.text(y, x, f"{letters[index]}", color='red', fontsize=10)
        
    graph, obs_array = find_connections(average_coordinate, letters, obstacles, size=height)
    # print(graph, obs_array)
    # plt.imshow(obs_base64)
    # plt.show()
    return graph, obs_array
# get_graph()

def get_shortest_path_dijkstra(start_node, end_node, graph):
    # image_path = 'final.png'
    # obs_base64 = image_to_base64('obs.png')
    # binary_data = base64.b64decode(obs_base64)
    # height, width, elementImage, obs_element = cv2_process_image(image_path, binary_data)
    shortest_distance, shortest_path = dijkstra(graph, start_node, end_node)
    
    # print(shortest_distance, shortest_path)

    # plt.imshow(obs_element)
    # plt.show()
    return shortest_path, shortest_distance

