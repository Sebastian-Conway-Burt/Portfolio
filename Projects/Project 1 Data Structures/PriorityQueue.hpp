// PriorityQueue.hpp
#ifndef PRIORITYQUEUE_HPP
#define PRIORITYQUEUE_HPP

#include <string>
#include <iostream>

using namespace std;

struct ReviewInfo {
   string restaurantName;
   string review;
   string customer;
   int time;

   // Default constructor
   ReviewInfo() : restaurantName(""), review(""), customer(""), time(0) {}

   // Parameterized constructor: ADD THIS
   ReviewInfo(string rName, string rev, string cust, int t) :
       restaurantName(rName), review(rev), customer(cust), time(t) {}

   // Overload assignment operator (already there)
   ReviewInfo& operator=(const ReviewInfo& other) {
       if (this != &other) {
           restaurantName = other.restaurantName;
           review = other.review;
           customer = other.customer;
           time = other.time;
       }
       return *this;
   }
};

// Priority Queue class implemented using a max-heap (array-based)
// Reviews are prioritized by 'time' (higher time = higher priority)
class PriorityQ {
private:
    ReviewInfo* heapArr; // Pointer to array of elements in heap
    int capacity;        // Maximum possible size of the heap
    int currentSize;     // Current number of elements in the heap

public:
    // Default constructor (if no capacity is specified during object creation)
    PriorityQ();

    // Parameterized constructor
    PriorityQ(int cap);

    // Destructor to free dynamically allocated memory
    ~PriorityQ();

    // Returns the index of the parent of a node
    int parent(int index);

    // Returns the index of the left child of a node
    int leftChild(int index);

    // Returns the index of the right child of a node
    int rightChild(int index);

    // Returns the current number of elements in the heap
    int getCurrentSize() { return currentSize; }

    // Prints the information of the highest priority item (root)
    void peek();

    // Maintains the max-heap property from a given index downwards
    void heapify(int index);

    // Removes the highest priority element from the priority queue
    void pop();

    // Inserts a new element into the priority queue
    void insertElement(ReviewInfo value);

    // Prints the contents of the entire heap array
    void print();

    // Checks if the priority queue is empty
    bool isEmpty() { return currentSize == 0; }
};

#endif // PRIORITYQUEUE_HPP