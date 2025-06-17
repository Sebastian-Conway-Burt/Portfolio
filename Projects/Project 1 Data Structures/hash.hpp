#ifndef HASH_HPP
#define HASH_HPP

#include <string>
#include <fstream>
#include "PriorityQueue.hpp" // Include for PriorityQ and ReviewInfo structures

// Structure to represent a node in the hash table's linked list chain
// Each node represents a unique restaurant
struct node {
    std::string restaurantName; // Unique identifier for the restaurant
    PriorityQ pq;               // Priority Queue storing reviews for this restaurant
    struct node* next;          // Pointer to the next node in the chain (for collision resolution)

    // Default constructor for node
    // Ensures pq is correctly initialized when a node is created
    node() : restaurantName(""), next(nullptr) {}

    // Destructor for node
    // The PriorityQ destructor will be automatically called,
    // handling its dynamically allocated memory.
    ~node() {
        // std::cout << "Deleting node for restaurant: " << restaurantName << std::endl;
        // No explicit deletion of 'pq' here, as it's a member object
        // The destructor of PriorityQ will be called when 'pq' goes out of scope.
    }
};

// HashTable class
// Implements hashing with chaining for collision resolution
class HashTable {
private:
    int tableSize;           // Number of buckets (linked lists) in the hash table
    node** table;            // Pointer to an array of node pointers (heads of linked lists)
    int numCollision;        // Counter for hash collisions

    // Private helper function to create a new node
    // This function is responsible for initializing a node and its associated Priority Queue
    node* createNode(std::string restaurant_name, node* next_node);

public:
    // Constructor: Initializes the hash table with a given size
    HashTable(int bsize);

    // Destructor: Frees all dynamically allocated memory (hash table array and all nodes/PQs)
    ~HashTable();

    // Sets up the hash table by populating it with data from a specified file
    void setup(std::string filename);

    // Inserts a ReviewInfo item into the hash table
    void insertItem(ReviewInfo restaurant_review);

    // Calculates the hash value for a given restaurant name
    unsigned int hashFunction(std::string restaurant_name);

    // Returns the current number of collisions encountered
    int getNumCollision() { return numCollision; }

    // Searches for a restaurant node in the hash table
    // Returns the node pointer if found, otherwise nullptr
    node* searchItem(std::string restaurant_name);

    // Displays the structure of the hash table (chains of restaurant names)
    void displayTable();
};

#endif // HASH_HPP