sorted_vertices = sorted(vertices, key=lambda vertex: (vertex[0]))

# grouped_vertices = []

# # Group vertices
# current_group = []

# for vertex in sorted_vertices:
#     if not current_group:
#         current_group.append(vertex)
#     else :
    
#         prev_vertex = current_group[-1]     
#         if vertex[0] == prev_vertex[0]:           
#             if vertex[1] == prev_vertex[1] + 1:                     
#                 current_group.append(vertex)
#             else:                                
#                 for group in grouped_vertices:
#                     group_prev_vertex = group[-1]                                                 
#                     if abs(vertex[0] - group_prev_vertex[0]) <= 1:                                                 
#                         if vertex[1] != group_prev_vertex[1]:                                                                                    
#                             if vertex[1] - group_prev_vertex[1] <= 400:                                                                
#                                 group.append(vertex)
                            
#                 grouped_vertices.append(current_group)            
#                 current_group = [vertex]                                    
#         elif vertex[0] != prev_vertex[0] and abs(vertex[0] - prev_vertex[0] <= 1):
#             if abs(vertex[1] - prev_vertex[1]) > 400:
                
#                 for group in grouped_vertices:
#                     group_prev_vertex = group[-1]                                                 
#                     if abs(vertex[0] - group_prev_vertex[0]) <= 1:                                                 
#                         if vertex[1] != group_prev_vertex[1]:                                                                                    
#                             if abs(vertex[1] - group_prev_vertex[1]) <= 400:                                                                
#                                 group.append(vertex)
#                                 break
#             else:
#                 current_group.append(vertex)
#         else:
#             grouped_vertices.append(current_group)            
#             current_group = [vertex]                                                                           
# # Append the last group
# grouped_vertices.append(current_group)



# average_x = []
# average_y = []
# group_number = []   


# # Iterate through grouped vertices and calculate average coordinates
# for idx, group in enumerate(grouped_vertices):
#     x_values = [vertex[0] for vertex in group]
#     y_values = [vertex[1] for vertex in group]
#     avg_x = sum(x_values) / len(group)
#     avg_y = sum(y_values) / len(group)
#     average_x.append(avg_x)
#     average_y.append(avg_y)
#     group_number.append(idx)

# # print(average_x, average_y)
# average_x = list(map(int, average_x))
# average_y = list(map(int, average_y))
# group_number = list(map(int, group_number))
# for x, y, group in zip(average_x, average_y, group_number):
#     plt.text(y, x, f"{group}", color='red', fontsize=10)

# def find_connections(average_x, average_y, group_number):

#     # group_letter = [A, B, C, D]
#     letters = [chr(ord('A') + i) for i in range(26)]
#     connections = []
#     graph = {letters[i]: [] for i in range(len(average_x))}
#     letters = [chr(ord('A') + i) for i in range(26)]
#     distances = [2, 5, 1, 6, 9, 21, 51, 52, 90, 88]
#     flag = 0
#     start_index = next((i for i, edge in enumerate(edges) if edge[0] == 160), None)
#     for i in range(0, len(average_x)):
#         connection = []        
#         # check horizontal connect
#         # Iterate from that index onwards
#         # if start_index is not None:
#         for edge in edges[start_index:]:            
#             connection.append([edge[0], average_x[i]])        
#             for number in range(len(average_x)):                        
#                 # print(average_y[number],edge[1])                            
#                 if average_y[number] == edge[0] and abs(average_x[i] - average_x[number]) <= 10:
#                     # print(average_y[i],average_x[i])                         
#                     dist = random.choice(distances)
#                     graph[letters[i]].append({letters[number]: dist})
#                     graph[letters[number]].append({letters[i]: dist})
#                     # print(graph, number)
#                     flag = 1
#                 if flag : break
#             if flag : break 
#         if flag == 0 : 
#             connection = []
#         # for j in range(average_y[i] + 1, z.shape[0]):
          
#         #     connection.append([j, average_x[i]])
#         #     for number in range(len(average_x)):                                
#         #         if average_y[number] == j and abs(average_x[i] - average_x[number]) <= 10:
#         #             # print(average_y[i],average_x[i])     
#         #             dist = random.choice(distances)
#         #             graph[letters[i]].append({letters[number]: dist})
#         #             graph[letters[number]].append({letters[i]: dist})
#         #             # print(graph, number)
#         #             flag = 1
                
#         #         if flag : break
#         #     if flag : break 
#         # if flag == 0 : 
#         #     connection = [] 
        
#         connections.append(connection)
#         # print(connections)
#         connection = []  
#         flag = 0
#         for conn in connection:
#             # print(conn)
#             # input('a')
#             elementImage[conn[1], conn[0]] = [255, 0, 0]  
#             # plt.text(conn[0], conn[1], f"-", color='red', fontsize=5)
#             with open("connections.txt", "w") as file:
#                 for conn in connection:
#                     file.write(str(conn) + "\n")

#         # check vertical connect
#         # for j in range(average_x[i] + 1, z.shape[1]):            
#         #     for number in range(len(average_x)):
#         #         connection.append([average_y[i], j])        
#         #         if average_x[number] == j and abs(average_y[i] - average_y[number]) <= 10:
#         #             # print(average_x[i],average_y[i])
#         #             dist = random.choice(distances)
#         #             graph[letters[i]].append({letters[number]: dist})
#         #             graph[letters[number]].append({letters[i]: dist})
#         #             # print(graph, number)                    
#         #             # connected_group.append([[average_y[i], j], average_y[i] , average_x[number]])                      
#         #             flag = 1
#         #         if flag : break
#         #     if flag : break               
#         # if flag == 0 : 
#         #     connection = [] 
                    
#         # connections.append(connection)
#         # connection = [] 
#         # flag = 0
#         # with open("search_coordinates.txt", "w") as file:
#         #     for x, y in zip(range(average_x[i] + 1, z.shape[1]), range(average_y[i] + 1, z.shape[0])):
#         #         for number in range(len(average_x)):
#         #             file.write(f"{average_x[i]} {x} {average_y[number]} {average_x[number]}\n")
#         #             file.write(f"{average_y[i]} {y} {average_y[number]} {average_x[number]}\n")
#         #             if abs(average_x[number] - x) <= 2000 and abs(average_y[i] - average_y[number]) <= 2000 \
#         #                 and abs(average_y[number] - y) <= 2000 \
#         #                 and abs(average_x[i] - average_x[number]) <= 2000 :
#         #                 if group_number[number] != group_number[i] and \
#         #                     group_number[number] != 1 and group_number[number] != 2:
#         #                     print(f"Group {group_number[i]} = Group {group_number[number]}")
#         #                     flag = 1
#         #                 # print(f"Coordinate [{j}, {average_y[i]}] found at index {average_y[number]} , {average_x[number]}")
#         #             if flag : break
#         #         if flag : break
#         # break
        
#     return connections, graph

# connections, graph = find_connections(average_x, average_y, group_number)

# print(graph)
# new_data = {}
# for key, value in graph.items():
#     new_value = {}
#     for item in value:
#         new_value.update(item)
#     new_data[key] = new_value



# conn_average_x = []
# conn_average_y = []
# for connected_groups in connections:  
#     for connected_group in connected_groups:                    
#         elementImage[connected_group[1], connected_group[0]] = [255, 0, 0]  # Red color for vertices
   


# # Build graph from connections and distances


# start_node = 'A'
# end_node = 'B'
# shortest_distance, shortest_path = dijkstra(new_data, start_node, end_node)
# if shortest_path:
#     print(f"Shortest distance from node {start_node} to node {end_node}: {shortest_distance}")
#     print("Shortest path:", shortest_path)
# else:
#     print(f"There is no path from node {start_node} to node {end_node}")

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
# distances = [2, 5, 1, 6, 9, 21, 51, 52, 90, 88]
# for x, y, group in zip(conn_average_x, conn_average_y, group_number):
#     plt.text(x, y, f"{random.choice(distances)}", color='red', fontsize=10)



# # Set vertices
# # for vertex in vertices:   
# #     elementImage[vertex] = [255, 0, 0]  # Red color for vertices    
    
# # # Set edges
# # for edge in edges:
# #     elementImage[edge] = [0, 255, 0]  # Green color for edges

# # Set start node
# # for start in start_node:
# #     image_with_elements[start] = [0, 0, 255]  # Blue color for start   node
# # 
# # # Set destination node
# # for destination in destination_node:
# #     image_with_elements[destination] = [255, 0, 255]  # Magenta color for destination node
# #
# # Visualize the result
# plt.imshow(elementImage)
# plt.show()

# # import heapq



# # # Example usage:
# # graph = {(0, 1): [88], (0, 2): [52], (1, 3): [21], (2, 3): [52], (2, 5): [52], (3, 4): [51], (3, 6): [1], (4, 7): [21], (5, 6): [2], (6, 7): [2]}
# # start_node = (0, 1)
# # end_node = (6, 7)
# # shortest_distance, shortest_path = dijkstra(graph, start_node, end_node)
# # print("Shortest distance from node", start_node, "to node", end_node, ":", shortest_distance)
# # print("Shortest path:", shortest_path)


