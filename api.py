# app.py
from flask import Flask, jsonify
from flask_cors import CORS
import base64
from main import main
app = Flask(__name__)
CORS(app, origins='http://localhost:5173')
@app.route('/api/data')
def get_data():
    # Your code to process data and return a response
    return {'data': 'Your data here'}

@app.route('/api/get/shortest/path/<start_node>/<end_node>')
def get_shortest_path(start_node, end_node):
    shortest_path, shortest_distance, encoded_image_bytes  = main(start_node, end_node)  # Assuming main() takes start_node and end_node as arguments
    # if shortest_path:
    #     print(f"Shortest distance from node {start_node} to node {end_node}: {shortest_distance}")
    #     print("Shortest path:", shortest_path)
    # else:
    #     print(f"There is no path from node {start_node} to node {end_node}")
   # Encode the bytes to base64
    encoded_image_base64 = encoded_image_bytes.decode('utf-8')

    # Construct the response JSON
    response = {
        'data': {
            'shortest_path': shortest_path,
            'shortest_distance': shortest_distance,
            'encoded_image': encoded_image_base64
        }
    }

    return jsonify(response)
    # return {'data': {'shortest_path': shortest_path, 'shortest_distance': shortest_distance, 'encoded_image': encoded_image}}
if __name__ == '__main__':
    app.run(host='localhost', port=5000)
