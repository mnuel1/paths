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
    # USE THIS FOR INITIALIZING THE MAP, AFTER IMPORTING
    # CALL THIS
    # STORE THE GRAPH AND CONNECTIONS TO STATE
    graph, connections = get_graph()
    response = {
        'graph': graph,  
        'paths': connections
    }
    return jsonify(message='Map Initialization Complete', data=response), 200


@app.route('/api/get/shortest/path', methods=['GET'])
def get_shortest_path():
    data = request.get_json()  # Get the JSON data from the request body
    shortest_path, shortest_distance, path = get_shortest_path_dijkstra(data['start_node'], data['end_node'], 
                                                                        data['graph'], data['paths'])
    # Construct the response JSON
    response = {
        'data': {
            'shortest_path': shortest_path,
            'shortest_distance': shortest_distance,
            'path': path
        }
    }

    return jsonify(message='Success', data=response), 200
if __name__ == '__main__':
    app.run(host='localhost', port=5000)
