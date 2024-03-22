def dijkstra(vertices, start):
    distances = {vertex: float('inf') for vertex in vertices}
    distances[start] = 0
    previous_vertices = {vertex: None for vertex in vertices}
    visited = set()

    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        if current_vertex in visited:
            continue

        visited.add(current_vertex)

        for neighbor, distance in vertices[current_vertex].items():
            if neighbor in visited:
                continue
            new_distance = current_distance + distance
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous_vertices[neighbor] = current_vertex
                heapq.heappush(priority_queue, (new_distance, neighbor))

    return distances, previous_vertices

def shortest_path(vertices, start, end):
    distances, previous_vertices = dijkstra(vertices, start)
    path = []
    current_vertex = end
    while current_vertex is not None:
        path.append(current_vertex)
        current_vertex = previous_vertices[current_vertex]
    path.reverse()
    return distances[end], path


def save_vertices_to_file(vertices, filename):
    vertices_str_keys = {}
    for key, value in vertices.items():
        neighbor_dict = {}
        for neighbor, distance in value.items():
            neighbor_str = str(neighbor)
            neighbor_dict[neighbor_str] = distance
        key_str = str(key)
        vertices_str_keys[key_str] = neighbor_dict
    
    with open(filename, 'w') as file:
        json.dump(vertices_str_keys, file, indent=4)
def connect_edges_to_neighbors(height, width, elementImage):
    vertices = {}
    
    # Trace the color and identify vertices, edges and obstacles
    for i in range(width):
        for j in range(height):
            elementsColor = elementImage[i, j] / 255

            if is_path(elementsColor):
                # This is an edge
                neighbors = []
                # Check 8-connected neighbors
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        x, y = i + dx, j + dy
                        if 0 <= x < width and 0 <= y < height:
                            if is_path(elementImage[x, y] / 255):
                                neighbors.append((x, y))
                vertices[(i, j)] = {neighbor: 1 for neighbor in neighbors}
    
    return vertices


# get graph
 # def save_vertices_to_txt(vertices, filename):
    #     with open(filename, 'w') as file:
    #         for key, value in vertices.items():
    #             file.write(f"Vertex: {key}\n")
    #             file.write("Neighbors:\n")
    #             for neighbor, distance in value.items():
    #                 file.write(f"    {neighbor}: {distance}\n")
    #             file.write("\n")

    # Example usage:
    vertices = connect_edges_to_neighbors(height, width, elementImage)
    vertices_str_keys = {}
    for key, value in vertices.items():
        neighbor_dict = {}
        for neighbor, distance in value.items():
            neighbor_str = str(neighbor)
            neighbor_dict[neighbor_str] = distance
        key_str = str(key)
        vertices_str_keys[key_str] = neighbor_dict
    # save_vertices_to_txt(vertices, 'vertices.txt')
    start_vertex = "(15, 15)"
    end_vertex = "(421, 255)"
    shortest_distance, path = shortest_path(vertices_str_keys, start_vertex, end_vertex)
    print("Shortest distance:", shortest_distance)
    print("Shortest path:", path)
    shortest_path_coordinates = [(int(vertex.strip('()').split(', ')[0]), int(vertex.strip('()').split(', ')[1])) for vertex in path]    
    for path in shortest_path_coordinates:
        elementImage[path] = [0, 255, 0]  # Green color for edges 
    # # Save vertices to a JSON file
    # save_vertices_to_file(vertices, 'vertices.json')
    # # Visualize the result
    plt.imshow(elementImage)
    plt.show()