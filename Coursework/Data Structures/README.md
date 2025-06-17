# Data Structures Coursework (CSCI 2270)

This directory contains programming assignments from my Data Structures course (CSCI 2270). This course provided a foundational understanding of abstract data types, their implementations, and their applications in solving various computational problems efficiently.

## Key Learnings from the Course

* **Core Data Structures:** Gained expertise in implementing and utilizing arrays, linked lists (singly and doubly), stacks, queues, trees (Binary Search Trees), hash tables, and heaps.
* **Performance Analysis:** Focused on understanding the time and space complexity of different data structure operations.
* **Memory Management:** Developed skills in dynamic memory allocation and deallocation in C++ to prevent memory leaks.
* **Object-Oriented Programming (OOP):** Applied OOP principles to design modular and reusable data structure components.
* **Problem-Solving:** Utilized appropriate data structures to efficiently solve problems related to data organization, searching, and prioritization.

## Assignments

The assignments are listed from the latest (most comprehensive/complex) to the earliest, reflecting the progression of concepts covered in the course.

### 1. Project: Restaurant Review System (Hash Table & Priority Queue)

**Description:** Implemented a restaurant review system that efficiently stores, retrieves, and manipulates reviews. The system leverages a **Hash Table with chaining** (using linked lists for collision resolution) to store distinct restaurants, and each restaurant node contains a **Max-Heap-based Priority Queue** to manage its reviews by recency (higher `time` values indicate higher priority).
**Advanced Aspects:** Combination of multiple complex data structures (Hash Table and Heap-based Priority Queue), custom collision handling logic, dynamic memory management in C++, and adherence to specific performance criteria.
**[View Code](./Project-RestaurantReviews/)**

### 2. Programming Assignment 9: Graph Algorithms (Dijkstra's & MST)

**Description:** This assignment likely involved implementing graph algorithms like Dijkstra's for shortest paths and/or Minimum Spanning Tree (MST) algorithms such as Kruskal's or Prim's. (Based on typical course progression and similar Algorithms PA9).
**Advanced Aspects:** Graph representation (adjacency lists/matrices), efficient shortest path or MST computation, handling weighted graphs.
**[View Code](./PA9-Graph-Dijkstra-MST/)**

### 3. Programming Assignment 8: Graph Traversal (BFS & DFS)

**Description:** Implemented fundamental graph traversal algorithms: Breadth-First Search (BFS) and Depth-First Search (DFS). This assignment likely focused on the mechanics of these traversals and their basic applications within graph theory.
**Advanced Aspects:** Managing visited nodes, handling graph cycles (if applicable to the assignment), and recursive vs. iterative approaches to traversal.
**[View Code](./PA8-Graph-BFS-DFS/)**

### 4. Programming Assignment 7: Show Catalog (Hash Table)

**Description:** Developed a system for managing a catalog of shows, likely using a **Hash Table** for efficient storage and retrieval of show information. This built upon earlier concepts of hash tables, possibly introducing more complex collision resolution strategies or hashing functions.
**Advanced Aspects:** Implementing hash table functionalities (insertion, search, deletion) for custom data types, managing collisions, and performance analysis of hash table operations.
**[View Code](./PA7-ShowCatalog-HashTable/)**

### 5. Programming Assignment 6: Show Catalog (BST)

**Description:** Created a show catalog management system using a **Binary Search Tree (BST)**. This assignment focused on the properties of BSTs for ordered data storage and efficient searching.
**Advanced Aspects:** Implementing BST operations (insertion, search, traversal, deletion) and understanding how BSTs maintain sorted order.
**[View Code](./PA6-ShowCatalog-BST/)**

### 6. Programming Assignment 5: Pancake House (Queues & Stacks)

**Description:** Implemented a simulation of a pancake house, likely using **Queues** to manage customer lines and **Stacks** for various operational tasks within the shop.
**Advanced Aspects:** Applying FIFO (Queue) and LIFO (Stack) principles to model real-world processes, understanding the differences between array-based and linked-list-based implementations of queues/stacks.
**[View Code](./PA5-PancakeHouse-QueuesStacks/)**

### 7. Programming Assignment 4: Browser History (Doubly Linked Lists)

**Description:** Developed a browser history feature using a **Doubly Linked List**. This assignment emphasized the advantages of DLLs for efficient forward and backward navigation.
**Advanced Aspects:** Implementing node insertion, deletion, and traversal in both directions, managing head and tail pointers, and handling edge cases in DLL operations.
**[View Code](./PA4-BrowserHistory-DLL/)**

### 8. Programming Assignment 3: Browser History (Singly Linked Lists)

**Description:** Implemented a basic browser history using a **Singly Linked List**. This served as an introduction to linked list concepts and their limitations compared to arrays.
**Advanced Aspects:** Fundamental linked list operations (insertion, traversal, deletion), understanding pointer manipulation, and identifying the `O(n)` cost of backward traversal in a singly linked list.
**[View Code](./PA3-BrowserHistory-LinkedLists/)**

### 9. Programming Assignment 2: Array Doubling

**Description:** This assignment focused on dynamic array resizing techniques, specifically implementing **array doubling** (or resizing) to handle collections that grow beyond their initial capacity.
**Advanced Aspects:** Manual memory management for arrays, `new` and `delete` operators in C++, and understanding the amortized `O(1)` complexity of array doubling for insertions.
**[View Code](./PA2-Array-Doubling/)**

### 10. Programming Assignment 1: Fundamentals (File I/O & Structs)

**Description:** This introductory C++ assignment covered fundamental programming concepts such as file input/output (I/O) and working with structs (custom data types). It laid the groundwork for managing structured data from external files.
**Advanced Aspects:** Basic file parsing, error handling for file operations, and defining custom data structures using C++ structs.
**[View Code](./PA1-Fundamentals/)**

---