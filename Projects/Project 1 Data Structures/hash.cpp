#include <iostream>
#include <sstream> // For std::stringstream to parse file lines
#include <fstream> // For std::ifstream to read files
#include "hash.hpp" // Include the HashTable class definition

// Constructor for node. This is a private helper function for HashTable.
// It creates a new node for a restaurant and initializes its PriorityQueue.
node* HashTable::createNode(std::string restaurant_name, node* next_node) {
    node* new_node = new node; // Dynamically allocate memory for the new node
    new_node->restaurantName = restaurant_name; // Set the restaurant name
    new_node->next = next_node; // Set the next pointer in the linked list chain
    
    // Initialize the PriorityQueue for this node.
    // The capacity (50) is hardcoded as per project specifications.
    new_node->pq = PriorityQ(50); // Calls the parameterized constructor for PriorityQ
    
    // std::cout << "New node created for " << restaurant_name << std::endl; // For debugging
    return new_node; // Return the newly created node
}

// Constructor for HashTable
// Initializes the hash table array with nullptrs and sets its size and collision count.
HashTable::HashTable(int bsize) {
    tableSize = bsize; // Set the size of the hash table (number of buckets)
    numCollision = 0;  // Initialize collision counter
    
    // Dynamically allocate an array of node pointers (buckets)
    table = new node*[tableSize];
    
    // Initialize each bucket head to nullptr (empty linked list)
    for (int i = 0; i < tableSize; i++) {
        table[i] = nullptr;
    }
    // std::cout << "HashTable created with size: " << tableSize << std::endl; // For debugging
}

// Destructor for HashTable
// Frees all dynamically allocated memory associated with the hash table:
// 1. All nodes within each linked list chain.
// 2. The array of pointers (buckets) itself.
HashTable::~HashTable() {
    // Iterate through each bucket in the hash table
    for (int i = 0; i < tableSize; i++) {
        node* current = table[i]; // Get the head of the current chain
        // Traverse the linked list chain and delete each node
        while (current != nullptr) {
            node* to_delete = current; // Store the current node to be deleted
            current = current->next;   // Move to the next node before deleting
            delete to_delete;          // Delete the node (its PriorityQ destructor is called automatically)
        }
        table[i] = nullptr; // Set the bucket head to nullptr after emptying the chain
    }
    delete[] table; // Delete the array of pointers
    table = nullptr; // Set the table pointer to null after deletion
    // std::cout << "HashTable destructor called. All memory freed." << std::endl; // For debugging
}

// Displays the contents of the hash table's chains.
// It iterates through each bucket and prints the restaurant names in that chain.
// It does NOT print the Priority Queue details within each node.
void HashTable::displayTable() {
    std::cout << "Displaying Hash Table:" << std::endl;
    for (int i = 0; i < tableSize; i++) {
        std::cout << i << "|"; // Print the bucket index
        node* current_node = table[i]; // Get the head of the current chain
        // Traverse the linked list chain
        while (current_node != nullptr) {
            std::cout << current_node->restaurantName << "-->"; // Print the restaurant name
            current_node = current_node->next; // Move to the next node
        }
        std::cout << "NULL" << std::endl; // Indicate the end of the chain
    }
}

// Calculates the hash value for a given restaurant name.
// Sums up the ASCII values of all characters in the string and
// then applies the modulo operator with respect to the tableSize.
unsigned int HashTable::hashFunction(std::string restaurant_name) {
    unsigned int ascii_sum = 0;
    // Iterate through each character of the restaurant name
    for (char c : restaurant_name) {
        ascii_sum += static_cast<unsigned int>(c); // Sum ASCII values
    }
    return ascii_sum % tableSize; // Return the bucket index
}

// Searches for a restaurant node in the hash table.
// Returns a pointer to the node if found, otherwise returns nullptr.
node* HashTable::searchItem(std::string restaurant_name) {
    unsigned int key = hashFunction(restaurant_name); // Calculate the hash key
    node* current = table[key]; // Get the head of the corresponding chain

    // Traverse the linked list chain to find the restaurant name
    while (current != nullptr) {
        if (current->restaurantName == restaurant_name) {
            return current; // Node found
        }
        current = current->next; // Move to the next node
    }
    return nullptr; // Node not found
}

// Inserts a ReviewInfo item into the hash table.
// It first searches for an existing node for the restaurant.
// If found, the review is added to that node's Priority Queue.
// If not found, a new node is created and added to the hash table,
// and the review is added to its Priority Queue.
void HashTable::insertItem(ReviewInfo restaurant_review) {
    std::string name = restaurant_review.restaurantName;
    unsigned int key = hashFunction(name); // Calculate the hash key for the restaurant name
    
    node* existing_node = searchItem(name); // Search for an existing node for this restaurant

    if (existing_node == nullptr) {
        // No existing node for this restaurant, so create a new one.
        // The new node is inserted at the head of the chain.
        node* new_node = createNode(name, table[key]);
        table[key] = new_node; // Update the bucket head to point to the new node

        // Add the review to the newly created node's Priority Queue
        new_node->pq.insertElement(restaurant_review);

        // If this is not the first node in the bucket, it's a collision
        if (new_node->next != nullptr) {
            numCollision++; // Increment collision count
            // The project description states: "Beyond this, any other insertion of
            // 'McDonlad's' or 'Chipotle' will not change numCollision."
            // This logic assumes a collision only counts when a *new restaurant* hashes to an *occupied bucket*.
            // If the restaurant already exists in the table, it's not a new collision.
        }

        // std::cout << "Insert successful. Restaurant: " << restaurant_review.restaurantName << std::endl; // For debugging
    } else {
        // Node for the restaurant already exists.
        // Insert the review into its existing Priority Queue.
        existing_node->pq.insertElement(restaurant_review);
        // The project description explicitly states: "adding multiple reviews into
        // the priority queue for the same restaurant node doesn't count as a collision."
        // So, numCollision is NOT incremented here.
        // std::cout << "Review inserted into existing Priority Queue. Restaurant: " << restaurant_review.restaurantName << std::endl; // For debugging
    }
}


// Populates the hash table by reading data from a specified file.
// Each line in the file represents a review and is parsed into a ReviewInfo structure.
void HashTable::setup(std::string filename) {
    std::ifstream file(filename); // Open the input file
    
    // Check if the file was successfully opened
    if (!file.is_open()) {
        std::cout << "Error: Could not open file '" << filename << "'." << std::endl;
        return; // Exit if file cannot be opened
    } else {
        std::cout << "File opened successfully: " << filename << std::endl;
    }

    std::string line;
    int line_count = 0;
    // Read the file line by line
    while (std::getline(file, line)) {
        line_count++;
        // std::cout << "Processing line " << line_count << ": " << line << std::endl; // For debugging

        std::stringstream ss(line); // Use stringstream to parse the line
        std::string restaurant_name_str, review_str, customer_str, time_str;
        int time_val;

        // Parse fields separated by semicolons
        std::getline(ss, restaurant_name_str, ';');
        std::getline(ss, review_str, ';');
        std::getline(ss, customer_str, ';');
        std::getline(ss, time_str); // Read until the end of the line for time

        // Convert time string to integer
        try {
            time_val = std::stoi(time_str);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Invalid time format in line " << line_count << ": " << time_str << std::endl;
            continue; // Skip this line and continue with the next
        } catch (const std::out_of_range& e) {
            std::cerr << "Time value out of range in line " << line_count << ": " << time_str << std::endl;
            continue; // Skip this line
        }

        // Create a ReviewInfo instance
        ReviewInfo review_info = {restaurant_name_str, review_str, customer_str, time_val};
        
        // Insert the review into the hash table
        insertItem(review_info);

        // std::cout << "Insert successful. Current table size: " << tableSize << std::endl; // For debugging
    }
    file.close(); // Close the file
    std::cout << "Finished setting up hash table from file. Processed " << line_count << " lines." << std::endl;
}