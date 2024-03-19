# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
# import base64
from main import get_shortest_path_dijkstra
from main import get_graph
app = Flask(__name__)
CORS(app, origins='http://localhost:5173')

@app.route('/api/data')
def get_data():
    # Your code to process data and return a response
    return {'data': 'Your data here'}

@app.route('/api/get/map_graph')
def map_graph():
    graph, connections = get_graph()
    response = {
        'graph': graph,  # Providing a key 'graph' for your graph data
        'paths': connections
    }
    return jsonify(message='Map Initialization Complete', data=response), 200


@app.route('/api/get/shortest/path', methods=['GET'])
def get_shortest_path():
    data = request.get_json()  # Get the JSON data from the request body
    
    shortest_path, shortest_distance, path = get_shortest_path_dijkstra(data.start_node, data.end_node, 
                                                                        data.graph, data.paths, data.obstacle)

#     shortest_path, shortest_distance, encoded_image_bytes  = main(start_node, end_node)  # Assuming main() takes start_node and end_node as arguments
#     # if shortest_path:
#     #     print(f"Shortest distance from node {start_node} to node {end_node}: {shortest_distance}")
#     #     print("Shortest path:", shortest_path)
#     # else:
#     #     print(f"There is no path from node {start_node} to node {end_node}")
#    # Encode the bytes to base64
#     encoded_image_base64 = encoded_image_bytes.decode('utf-8')

#     # Construct the response JSON
    response = {
        'data': {
            'shortest_path': shortest_path,
            'shortest_distance': shortest_distance,
            'path': path
            # 'encoded_image': encoded_image_base64
        }
    }

    return jsonify(message='Success', data=response), 200
if __name__ == '__main__':
    app.run(host='localhost', port=5000)
