# Data Structures Project: Restaurant Review System

## Project Overview

This project implements a comprehensive system for storing, retrieving, and manipulating restaurant reviews, developed as a capstone for a university Data Structures course. The primary objective was to design and build a robust data management solution leveraging two fundamental data structures: a **Hash Table for efficient restaurant lookup** and a **Priority Queue (implemented as a Max-Heap) for managing reviews by recency**.

A significant portion of this project involved **extensive research and planning** around how to effectively combine these data structures to meet specific operational requirements. The emphasis was on theoretical understanding, proper implementation of core data structure operations, and strategic design for data flow within the system.

## Learning Objectives & Business Context

This project served as a practical application of core data structures concepts, simulating a real-world scenario where efficient data organization is paramount. As a capstone to a business minor, it particularly highlighted:

* **Strategic Data Modeling:** How fundamental data structures like Hash Tables and Heaps can be combined to solve practical business problems, such as managing customer feedback.
* **Efficiency Considerations:** Understanding the performance implications (e.g., `O(1)` average-case lookup in hash tables, `O(log n)` operations in heaps) for real-time review management.
* **Business Requirements to Technical Design:** Translating business rules (like "most recent review has highest priority" or "only count collisions for new restaurants") into specific data structure implementations and algorithms.
* **Software Architecture (MVP Perspective):** While not a full product, the system functions as a Minimal Viable Product (MVP) backend, demonstrating core capabilities to potential stakeholders or for further development. This involved careful planning of interfaces and interactions between modules.

## System Architecture & Data Structures

The system is designed with a layered approach:

1.  **Hash Table (`HashTable` class):**
    * **Purpose:** Stores restaurant information, enabling quick access by `restaurantName`.
    * **Collision Resolution:** Implemented using **chaining**, where each hash table "bucket" points to a linked list of `node`s.
    * **`node` structure:** Each `node` in the linked list represents a distinct `restaurantName` and contains its own `PriorityQ` instance.
    * **Hash Function:** A simple sum of ASCII values modulo `tableSize` is used.
    * **Collision Tracking:** A `numCollision` counter tracks new restaurant insertions that result in a hash collision (i.e., mapping to an already occupied bucket). Adding multiple reviews for the *same* restaurant does not increment the collision count.

2.  **Priority Queue (`PriorityQ` class):**
    * **Purpose:** Manages `ReviewInfo` objects for a specific restaurant, ordered by `time` (higher `time` means higher priority).
    * **Implementation:** An **array-based Max-Heap** is used, ensuring that the review with the highest `time` value is always at the root.
    * **`ReviewInfo` struct:** Defines the structure of individual reviews, including `restaurantName`, `review` text, `customer` name, and `time` (24-hour format).

## Core Functionality

The system provides a menu-driven interface to perform the following operations:

* **Build Data Structure:** Populates the hash table from an external semicolon-separated file (e.g., `test.txt`). This operation can only be executed once per program run.
* **Add a Review:** Prompts the user for `restaurantName`, `customer`, `review`, and `time`, then inserts this `ReviewInfo` into the appropriate restaurant's priority queue within the hash table.
* **Retrieve Most Recent Review:** Searches for a restaurant and, if found, displays the highest priority (most recent) review using the `peek` operation on its priority queue.
* **Pop Most Recent Review:** Removes the highest priority review for a specified restaurant using the `pop` operation.
* **Print All Reviews:** Displays all reviews for a given restaurant from its associated priority queue.
* **Display Number of Collisions:** Shows the total count of hash collisions (for distinct restaurants).
* **Display Table Results:** Prints the structure of the hash table, showing which restaurants are in which chains/buckets.

## How to Compile and Run

This project is written in C++ and uses a `Makefile` (or `CMakeLists.txt`) for compilation.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-github-repo-name.git](https://github.com/your-username/your-github-repo-name.git)
    cd your-github-repo-name/coursework/data_structures/Project-RestaurantReviews
    ```
2.  **Compile the Code:**
    * Assuming you have `g++` installed and a `Makefile`:
        ```bash
        make
        ```
    * If using `CMakeLists.txt`:
        ```bash
        mkdir build
        cd build
        cmake ..
        make
        cd ..
        ```
3.  **Run the Executable:**
    The program requires two command-line arguments: the input file (e.g., `test.txt`) and the desired hash table size.

    ```bash
    ./your_executable_name test.txt 10  # Replace 'your_executable_name' with the actual name (e.g., 'driver') and '10' with your desired table size
    ```
    * *(Note: The `executableName.txt` file in your original upload suggests the executable might simply be `executableName`. Verify this in your `CMakeLists.txt` or `Makefile`.)*

4.  **Interact with the Menu:** Follow the on-screen menu prompts to test various functionalities.

## Project Files

* **`app/main.cpp`**: Contains the `main` function and the menu-driven interface (`driver.cpp` in your original structure).
* **`code/hash.hpp`**: Header file for the `HashTable` class and `node` struct.
* **`code/hash.cpp`**: Implementation file for the `HashTable` class.
* **`code/PriorityQueue.hpp`**: Header file for the `PriorityQ` class and `ReviewInfo` struct.
* **`code/PriorityQueue.cpp`**: Implementation file for the `PriorityQ` class.
* **`data/test.txt`**: Sample input file containing restaurant reviews.
* **`docs/CSCI2270_Spring24_Project.pdf`**: The original project specification (PDF).
* **`CMakeLists.txt`** (and `CMakeLists.txt.in` if applicable): Build configuration files.
* **`tests/`** (and its contents): Unit tests for the data structures.
* **`executableName.txt`** (optional: a file indicating the compiled executable name).

---