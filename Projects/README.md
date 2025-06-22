# My Projects Portfolio

This section showcases a selection of my key projects, demonstrating my skills in computational mathematics, AI, data structures, and algorithms. Each project represents a significant learning experience or a practical application of advanced concepts.

## Projects

### 1. P-Spline Volatility Surface Modeling (Numerical Analysis Capstone)

**Description:** This project involved developing a robust multi-stage numerical pipeline in Python to construct and analyze implied volatility surfaces from real-world market option data. It addresses the complexities of financial option valuation and risk management, where raw market data is often noisy and sparse.

**Advanced Aspects Highlighted:**
* **Implied Volatility Extraction:** Utilized the Newton-Raphson method to numerically solve the non-linear Black-Scholes pricing equation, handling issues like near-zero Vega for stability.
* **Yield Curve Interpolation:** Employed 1D natural cubic splines to create a smooth, $C^2$ continuous risk-free interest rate curve from discrete market yield data.
* **2D Surface Fitting:** Implemented 2D Penalized B-Splines (P-splines) with a tensor-product basis for fitting a smooth implied volatility surface to scattered data points, incorporating a tunable smoothing parameter ($\lambda$) to balance fidelity and smoothness.
* **Large Scale System Solving:** Applied the LSQR algorithm to efficiently and stably solve large, sparse linear least-squares problems arising from the P-spline formulation, avoiding the ill-conditioning issues of normal equations.
* **Data Robustness:** Focused on extensive data preprocessing and filtering techniques to ensure numerical stability and accuracy.

**[View Project Details](./numerical_analysis_project/README.md)**

---

### 2. EDU-Copilot (Entrepreneurship Capstone)

**Description:** As CEO and Project Lead, I directed a team in developing a comprehensive business venture plan for an AI-driven academic advising platform. This project emphasized strategic design, market validation, and the articulation of a scalable solution to address inefficiencies in higher education advising.

**Advanced Aspects Highlighted:**
* **AI-Driven Personalization:** Conceptualized an AI core capable of parsing unstructured student documents (transcripts, requirements) and leveraging graph-based data models to provide personalized academic and career pathways.
* **Business Model Innovation:** Developed a multi-stream revenue model, including a direct-to-student freemium offering and B2B university licenses, informed by extensive market analysis and competitive positioning.
* **MVP Development Strategy:** The project included the design of a Minimum Viable Product (MVP) website as a demonstration tool for potential investors, showcasing key AI functionalities and user experience.
* **Market & Competitive Analysis:** Conducted in-depth market research, including student and industry interviews, to validate the market need and differentiate EDU-Copilot within the EdTech landscape.
* **Operational & Financial Planning:** Outlined scalable technology infrastructure, AI model development and optimization strategies, and detailed financial projections to achieve profitability.

**[View Project Details](./edu_copilot_project/README.md)**

---

### 3. Maze Solver using Dijkstra's Algorithm

**Description:** This project implements a Python-based maze solver that finds the shortest path through mazes represented as images. It showcases an adapted Dijkstra's algorithm that efficiently handles large graph structures by generating vertices and edges "on-the-fly" based on pixel color differences.

**Advanced Aspects Highlighted:**
* **Graph Representation for Image Data:** Each pixel is treated as a vertex, and edges are dynamically created between neighbors.
* **On-the-Fly Dijkstra's:** The algorithm avoids memory-intensive pre-computation of an adjacency list, crucial for large image dimensions.
* **Weighted Edges:** Edge weights are derived from the squared Euclidean distance of RGB pixel values, enabling the algorithm to find paths that minimize changes in color (e.g., sticking to non-wall areas).
* **Priority Queue Optimization:** Utilizes `heapq` for efficient vertex selection in Dijkstra's, ensuring optimal performance for shortest path discovery.

**[View Project Details](./maze_dijkstra_solver/README.md)**

---

Now, let's create the `README.md` for your Algorithms course. I will refer to the "cspb" files with their new suggested names (e.g., `Algorithms_PA2_Merge_Sort_and_Fixed_Points.ipynb`) to highlight the advanced aspects.