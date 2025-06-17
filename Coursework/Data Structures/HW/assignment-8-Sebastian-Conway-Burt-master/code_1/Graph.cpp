#include "Graph.hpp"
#include <vector>
#include <queue>
#include <iostream>
#include <limits>


int inf = std::numeric_limits<int>::max();





using namespace std;


void Graph::addEdge(string v1, string v2) {
    vertex* v1Ptr = nullptr;
    vertex* v2Ptr = nullptr;

    for (auto it = vertices.begin(); it != vertices.end(); ++it) {
        if ((*it)->name == v1) {
            v1Ptr = *it;
        } else if ((*it)->name == v2) {
            v2Ptr = *it;
        }

        // Break early if both vertices are found
        if (v1Ptr && v2Ptr) break;
    }

    if (v1Ptr && v2Ptr) {
        adjVertex adjV1 {v1Ptr};
        adjVertex adjV2 {v2Ptr};

        v2Ptr->adj.push_back(adjV1);
        v1Ptr->adj.push_back(adjV2);
    } else {
        if (!v1Ptr) {
            cout << "Vertex " << v1 << " does not exist." << endl;
        }
        if (!v2Ptr) {
            cout << "Vertex " << v2 << " does not exist." << endl;
        }
    }
}




void Graph::addVertex(string name) {
    bool found = false;
    for (auto it = vertices.begin(); it != vertices.end(); ++it) {
        if ((*it)->name == name) {
            found = true;
            cout << (*it)->name << " found." << endl;
            break; // Stop the search once we find the vertex
        }
    }
    if (!found) {
        vertex* v = new vertex;
        v->name = name;
        vertices.push_back(v);
    }
}





void Graph::displayEdges() {
    for (auto &vertex : vertices) {
        cout << vertex->name << " --> ";
        for (auto &adjacent : vertex->adj) {
            cout << adjacent.v->name << " ";
        }
        cout << endl;
    }
}

void Graph::breadthFirstTraverse(string sourceVertex) {
    for (auto &v : vertices) {
        v->visited = false;
        v->distance = std::numeric_limits<int>::max(); 
    }

    vertex* startVertex = nullptr;
    for (auto &v : vertices) {
        if (v->name == sourceVertex) {
            startVertex = v;
            break;
        }
    }

    if (!startVertex) {
        cout << "Start vertex not found." << endl;
        return;
    }

    startVertex->visited = true;
    startVertex->distance = 0; 
    queue<vertex*> q;
    q.push(startVertex);

    cout << "Starting vertex (root): " << startVertex->name << "-> ";

    while (!q.empty()) {
        vertex* current = q.front();
        q.pop();

        if (current != startVertex) {
            cout << current->name << "(" << current->distance << ")" << " ";
        }

       
        for (auto &adj : current->adj) {
            if (!adj.v->visited) {
                adj.v->visited = true;
                adj.v->distance = current->distance + 1; 
                q.push(adj.v);
            }
        }
    }

    cout << "\n\n";
}





vector<string> Graph::findReachableCitiesWithinDistanceK(string source, int k) {
    vector<string> reachableCities;

    
    for (auto &v : vertices) {
        v->visited = false;
        v->distance = inf; 
    }

    // Find the starting vertex
    vertex *startVertex = nullptr;
    for (auto &v : vertices) {
        if (v->name == source) {
            startVertex = v;
            break;
        }
    }

    if (!startVertex) {
        cout << "Starting city " << source << " not found." << endl;
        return reachableCities; 
    }

    queue<vertex*> q;
    startVertex->visited = true;
    startVertex->distance = 0;
    q.push(startVertex);

    // Begin BFS
    while (!q.empty()) {
        vertex *current = q.front();
        q.pop();

        if (current->distance == k) {
            reachableCities.push_back(current->name);
            continue; 
        }

        if (current->distance < k) {
            for (auto &adj : current->adj) {
                vertex *adjVertex = adj.v;
                if (!adjVertex->visited) {
                    adjVertex->visited = true;
                    adjVertex->distance = current->distance + 1;
                    q.push(adjVertex);
                }
            }
        }
    }

    return reachableCities;
}
