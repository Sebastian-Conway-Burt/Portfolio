# Algorithms Coursework (CSPB 3104)

This directory contains programming assignments from my Algorithms course (CSPB 3104). The course provided a comprehensive foundation in fundamental algorithm design techniques, complexity analysis, and various data structures.

## Key Learnings from the Course

* **Algorithmic Paradigms:** Gained hands-on experience with Divide and Conquer, Dynamic Programming, Greedy Algorithms, and Graph Algorithms.
* **Complexity Analysis:** Developed proficiency in analyzing the time and space complexity of algorithms using Big O notation.
* **Sorting Algorithms:** Implemented and analyzed various sorting techniques, understanding their trade-offs in different scenarios.
* **Graph Traversal & Analysis:** Explored Breadth-First Search (BFS) and Depth-First Search (DFS) and their applications in cycle detection.
* **Optimization Problems:** Applied dynamic programming and greedy approaches to solve optimization challenges.

## Assignments

The assignments are listed from the latest (most advanced) to the earliest, reflecting the progression of complexity and concepts learned throughout the course.

### 1. Programming Assignment 9: Kruskal's, Prim's, and Mutation Tree Analysis

**Description:** This assignment involved implementing two fundamental Minimum Spanning Tree (MST) algorithms, Kruskal's and Prim's, and applying MST concepts to a bioinformatics problem of finding the most likely mutation tree for RNA fragments.
**Advanced Aspects:** Graph representation, greedy algorithms for MST, transformation of a maximization problem (most likely mutation tree) into a minimization problem (shortest MST) using logarithmic weights.
**[View Code](./PA9-Kruskal-Prim-MST/Algo_PA9_Kruskal_Prim_Mutation_Tree.ipynb)**

### 2. Programming Assignment 8: BFS, DFS, and Cycle Detection

**Description:** Implemented core graph traversal algorithms, Breadth-First Search (BFS) and Depth-First Search (DFS), and applied them to detect cycles within a graph.
**Advanced Aspects:** Adjacency list representation, explicit tracking of discovery and finishing times in DFS, using recursion stacks and visited sets for cycle detection in directed graphs.
**[View Code](./PA8-BFS-DFS-CycleDetection/Algo_PA8_BFS_DFS_Cycle_Detection.ipynb)**

### 3. Programming Assignment 6/7: Dynamic Programming (Magic Bean Stalks)

**Description:** Explored dynamic programming techniques to solve variations of the "Magic Bean Stalks" problem, including finding the minimum number of fertilizer drops to reach a target length with constraints (forbidden lengths, budget limitations).
**Advanced Aspects:** Formulating recurrences for dynamic programming, implementing memoization (top-down DP), and developing bottom-up DP solutions with path reconstruction for optimal sequences.
**[View Code](./PA6-7-DynamicProgramming/Algo_PA6-7_DynamicProgramming.ipynb)**

### 4. Programming Assignment 5: Binary Search Trees (Preorder/Postorder Traversals)

**Description:** Focused on Binary Search Trees (BSTs), including reconstructing a BST from its preorder traversal and converting a preorder traversal directly to a postorder traversal without explicitly building the tree.
**Advanced Aspects:** Recursive tree construction, efficient traversal conversions based on BST properties, and understanding tree invariants.
**[View Code](./PA5-BST-Traversals/Algo_PA5_BST_Preorder_Postorder_Traversal.ipynb)**

### 5. Programming Assignment 4: Heap Implementation and Heapsort

**Description:** Implemented a max-heap (or min-heap, for heapsort typically max-heap is used) and the Heapsort algorithm. This involved understanding and implementing `heapify`, `bubble_up`, `bubble_down`, and `extract_min` (or `extract_max`) operations.
**Advanced Aspects:** In-place heap construction (`heapify`), maintaining heap properties efficiently during insertions and deletions, and implementing a complete sorting algorithm based on heap operations.
**[View Code](./PA4-Heap-Heapsort/Algo_PA4_Heap_Heapsort.ipynb)**

### 6. Programming Assignment 3: Quicksort and Quickselect (Median-of-7)

**Description:** Implemented Quicksort, a classic divide-and-conquer sorting algorithm, and Quickselect for finding the k-th largest element. A more advanced pivot selection strategy, "Median-of-7-Medians," was used for Quickselect to improve worst-case performance.
**Advanced Aspects:** In-place partitioning, recursive sorting, robust pivot selection strategies (Median-of-Medians for better guarantees on pivot quality), and understanding average vs. worst-case complexities.
**[View Code](./PA3-Quicksort-Quickselect/Algo_PA3_Quicksort_Quickselect.ipynb)**

### 7. Programming Assignment 2: Mergesort and Fixed Points

**Description:** Implemented Mergesort, another divide-and-conquer sorting algorithm, and a binary search-based function to find fixed points in sorted arrays.
**Advanced Aspects:** Recursive mergesort with comparison counting, efficient merging of sorted sub-arrays, and logarithmic-time fixed point detection using binary search.
**[View Code](./PA2-MergeSort-FixedPoints/Algo_PA2_Merge_Sort_FixedPoints.ipynb)**

### 8. Programming Assignment 1: Insertion Sort Fundamentals

**Description:** Implemented fundamental array manipulation techniques (`swap` and `insert_into`) as a basis for understanding insertion-like sorting algorithms.
**Advanced Aspects:** Direct in-place array manipulation, efficiency of element insertion into a sorted segment, and foundational building blocks for more complex sorting algorithms.
**[View Code](./PA1-InsertionSort/Algo_PA1_InsertionSort.ipynb)**