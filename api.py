# app.py
from flask import Flask
from flask_cors import CORS
from main import main
app = Flask(__name__)
CORS(app, origins='http://localhost:5173')
@app.route('/api/data')
def get_data():
    # Your code to process data and return a response
    return {'data': 'Your data here'}

@app.route('/api/get/shortest/path/<start_node>/<end_node>')
def get_shortest_path(start_node, end_node):
    shortest_path, shortest_distance = main(start_node, end_node)  # Assuming main() takes start_node and end_node as arguments
    if shortest_path:
        print(f"Shortest distance from node {start_node} to node {end_node}: {shortest_distance}")
        print("Shortest path:", shortest_path)
    else:
        print(f"There is no path from node {start_node} to node {end_node}")

    return {'data': {'shortest_path': shortest_path, 'shortest_distance': shortest_distance}}
if __name__ == '__main__':
    app.run(host='localhost', port=5000)
