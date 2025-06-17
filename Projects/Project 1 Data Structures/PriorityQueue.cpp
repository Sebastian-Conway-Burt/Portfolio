#include "PriorityQueue.hpp"
#include <iostream> // For std::cout, std::endl
#include <limits.h> // For INT_MIN (though not strictly used after initial cleanup)

// Helper function to swap two ReviewInfo objects
// Used internally by heap operations
void swap(ReviewInfo *a, ReviewInfo *b) {
    ReviewInfo temp = *a;
    *a = *b;
    *b = temp;
}

// Default constructor for PriorityQ, initializes with a default capacity
PriorityQ::PriorityQ() : capacity(0), currentSize(0), heapArr(nullptr) {
    // You might want to set a default capacity like 50 if this is expected to be used
    // without a specific capacity. For now, it initializes to an empty state.
    // If a default capacity is desired:
    // capacity = 50;
    // heapArr = new ReviewInfo[capacity];
    // if (!heapArr) {
    //     std::cout << "Failed to allocate memory for heapArr in default constructor." << std::endl;
    // }
}


// Parameterized constructor for PriorityQ
// Initializes the heap with a given capacity
PriorityQ::PriorityQ(int cap) {
    capacity = cap;
    currentSize = 0;
    heapArr = new ReviewInfo[capacity]; // Dynamically allocate array for the heap
    if (!heapArr) {
        std::cout << "Failed to allocate memory for heapArr." << std::endl;
        // Consider handling this error more robustly, e.g., throw an exception
    }
}

// Destructor for PriorityQ
// Frees the dynamically allocated heap array to prevent memory leaks
PriorityQ::~PriorityQ() {
    delete[] heapArr; // Delete the array of ReviewInfo objects
    heapArr = nullptr; // Set pointer to null after deletion
}

// Calculates the index of the parent of a node
// For a 0-indexed array where children are at 2*i + 1 and 2*i + 2
int PriorityQ::parent(int index) {
    if (index == 0) return -1; // Root has no parent
    return (index - 1) / 2;
}

// Returns the index of the left child of a node
int PriorityQ::leftChild(int index) {
    return (2 * index) + 1;
}

// Returns the index of the right child of a node
int PriorityQ::rightChild(int index) {
    return (2 * index) + 2;
}

// Peeks at (returns/prints) the highest priority element (root of the max-heap)
void PriorityQ::peek() {
    if (isEmpty()) {
        std::cout << "no record found" << std::endl;
    } else {
        const ReviewInfo& top = heapArr[0]; // Access the root element
        std::cout << "Restaurant: " << top.restaurantName << std::endl;
        std::cout << "Customer: " << top.customer << std::endl;
        std::cout << "Review: " << top.review << std::endl;
        std::cout << "Time: " << top.time << std::endl;
    }
}

// Maintains the max-heap property by sifting down an element at a given index
void PriorityQ::heapify(int index) {
    int largest = index; // Assume current node is the largest
    int left = leftChild(index);
    int right = rightChild(index);

    // Check if left child exists and has a higher priority (time)
    if (left < currentSize && heapArr[left].time > heapArr[largest].time) {
        largest = left;
    }

    // Check if right child exists and has a higher priority (time)
    if (right < currentSize && heapArr[right].time > heapArr[largest].time) {
        largest = right;
    }

    // If the largest is not the current index, swap them and recursively heapify
    if (largest != index) {
        swap(&heapArr[index], &heapArr[largest]); // Swap current with largest child
        heapify(largest); // Recursively heapify the affected sub-tree
    }
}

// Inserts a new ReviewInfo element into the priority queue
void PriorityQ::insertElement(ReviewInfo value) {
    if (currentSize == capacity) {
        std::cout << "Maximum heap size reached. Cannot insert anymore." << std::endl;
        return;
    }

    // Insert the new element at the end of the heap array
    heapArr[currentSize] = value;
    int current = currentSize; // Index of the newly inserted element
    currentSize++;             // Increment current size

    // Sift up the new element to maintain the max-heap property
    // Continue swapping with parent as long as current element has higher priority
    while (current != 0 && heapArr[parent(current)].time < heapArr[current].time) {
        swap(&heapArr[parent(current)], &heapArr[current]);
        current = parent(current); // Move up to the parent's position
    }

    // The provided driver output expects this print after every insert.
    // For a cleaner output, you might remove this or make it conditional.
    // std::cout << "Review inserted and heap property maintained. Current size: " << currentSize << std::endl;
}

// Removes the highest priority element (root) from the priority queue
void PriorityQ::pop() {
    if (isEmpty()) {
        std::cout << "Priority queue is empty, nothing to pop." << std::endl;
        return;
    }

    // Move the last element to the root position
    heapArr[0] = heapArr[currentSize - 1];
    currentSize--; // Decrease the current size

    // Restore the heap property by calling heapify on the new root
    heapify(0);

    // The provided driver output expects this print after every pop.
    // For a cleaner output, you might remove this or make it conditional.
    // std::cout << "Top element popped and heap property restored." << std::endl;
}

// Prints the elements currently stored in the heap array
// This is not necessarily in priority order as it traverses the underlying array.
void PriorityQ::print() {
    if (isEmpty()) {
        std::cout << "No reviews in the priority queue." << std::endl;
        return;
    }

    std::cout << "Printing all reviews in the priority queue:" << std::endl;
    for (int i = 0; i < currentSize; i++) {
        std::cout << "\tRestaurant: " << heapArr[i].restaurantName << std::endl; // Added restaurant name for clarity
        std::cout << "\tCustomer: " << heapArr[i].customer << std::endl;
        std::cout << "\tReview: " << heapArr[i].review << std::endl;
        std::cout << "\tTime: " << heapArr[i].time << std::endl;
        std::cout << "\t=====" << std::endl;
    }
}