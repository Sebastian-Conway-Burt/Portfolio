# Amazing Maze Dijkstra Solver

## Project Overview

This project implements a custom maze solver that navigates complex mazes represented as image files. The core of the solution lies in a modified **Dijkstra's algorithm**, tailored to operate on pixel data rather than a pre-constructed adjacency list, optimizing for memory efficiency with large image inputs. The "weight" of traversing between pixels is determined by the color difference between adjacent pixels, allowing the algorithm to find paths that minimize visual changes (e.g., sticking to white paths in a black-and-white maze).

This solver was developed as part of a university algorithms course, focusing on graph theory, shortest path algorithms, and on-the-fly graph construction for memory-constrained problems.

## Key Concepts & Algorithms

* **Dijkstra's Algorithm:** A classic shortest path algorithm used to find the shortest (lowest-cost) path between a source and a destination in a graph with non-negative edge weights.
* **On-the-Fly Graph Construction:** Instead of building a full adjacency list for the entire image (which would be very memory intensive for large images), the algorithm dynamically identifies and evaluates neighboring pixels as needed during the search.
* **Weighted Edges based on Pixel Color:** Edges between adjacent pixels are assigned weights based on the squared Euclidean distance of their RGB color values. A small constant is added to ensure all weights are positive, a requirement for Dijkstra's. Black pixels (representing walls) are treated as impassable.
* **Priority Queue:** A min-priority queue (implemented using Python's `heapq` module) is used to efficiently select the next unvisited pixel with the smallest known distance from the source.
* **Path Reconstruction:** Once the destination is reached, the path is reconstructed by backtracking through `predecessor` pointers stored during the Dijkstra's execution.

## Code Structure

The project's logic is encapsulated within a single Jupyter Notebook for ease of demonstration and execution:

* **`Maze_Dijkstra_Solver.ipynb`**:
    * **Helper Functions:** Includes `fix_pixel_values` (converts RGB to float to prevent overflow), `get_edge_weight` (calculates pixel-based edge costs), and `draw_path` (visualizes the solution on the image).
    * **`Vertex` Class:** A custom class to represent pixels as graph vertices, storing distance, processed status, and predecessor information for Dijkstra's.
    * **`PriorityQueue` Class:** A basic wrapper around `heapq` to manage vertex priorities.
    * **`computeShortestPath` Function:** The main function that orchestrates the Dijkstra's algorithm, handling image loading, distance/predecessor tracking, and on-the-fly neighbor generation.
    * **Example Usage:** Demonstrates the solver's application on several different maze images (`maze.png`, `maze2.JPG`, `maze3.JPG`) with varying complexities.

## How to Run

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-github-repo-name.git](https://github.com/your-username/your-github-repo-name.git)
    cd your-github-repo-name/projects/maze_dijkstra_solver
    ```
2.  **Install Dependencies:**
    ```bash
    pip install opencv-python numpy matplotlib
    ```
3.  **Download Maze Images:** Ensure `maze.png`, `maze2.JPG`, and `maze3.JPG` are in the same directory as the notebook (these were part of your original uploaded files).
4.  **Open Jupyter Notebook:**
    ```bash
    jupyter notebook Maze_Dijkstra_Solver.ipynb
    ```
5.  **Run Cells:** Execute the cells sequentially within the notebook to see the solver in action and visualize the results.

## Project Files

* **[View the Jupyter Notebook](./Maze_Dijkstra_Solver.ipynb)**
* **(Optional) Add links to maze image files if you upload them to the repo:**
    * `./maze.png`
    * `./maze2.JPG`
    * `./maze3.JPG`

---