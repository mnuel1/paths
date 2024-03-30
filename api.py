# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from main import get_shortest_path_dijkstra
from main import get_graph

app = Flask(__name__)
CORS(app, origins='http://localhost:5173')

@app.route('/api/data')
def get_data():
    # Your code to process data and return a response
    return {'data': 'Your data here'}

@app.route('/api/get/map_graph', methods=['POST'])
def map_graph():
    # USE THIS FOR INITIALIZING THE MAP, AFTER IMPORTING
    # CALL THIS
    # STORE THE GRAPH AND CONNECTIONS TO STATE
    data = request.get_json()
    if 'base64EncodedMap' in data:
        base64_image = data['base64EncodedMap']

        binary_data = base64.b64decode(base64_image)
        # print(binary_data)
        graph, obs_array = get_graph(binary_data)
        response = {
            'graph': graph,
            'blockedEdges': obs_array
        }
        return jsonify(message='Map Initialization Complete', data=response), 200
    else:
        return jsonify(message='Base64 image data not found in request'), 400


@app.route('/api/get/shortest/path', methods=['GET'])
def get_shortest_path():
    data = request.get_json()  # Get the JSON data from the request body
    shortest_path, shortest_distance = get_shortest_path_dijkstra(data['startNode'], data['targetNode'], 
                                                                data['graph'])    
    if not shortest_path:
        return jsonify(message='No path found!'), 200
    
    # Construct the response JSON
    response = {        
        'path': shortest_path,
        'safestAndShortestPathDistance': shortest_distance,            
    }

    return jsonify(message='Success', data=response), 200

if __name__ == '__main__':
    app.run(host='localhost', port=5000)


