#include "ShowCatalog.hpp"
#include <iostream>
#include <string>
#include <algorithm>

using namespace std;

// Constructor
ShowCatalog::ShowCatalog() {    
    root = nullptr;
}

// Destructor
ShowCatalog::~ShowCatalog() {
    clear(root);
}

// Helper to recursively delete all nodes
void ShowCatalog::clear(ShowItem* node) {
    if (node != nullptr) {
        clear(node->left);
        clear(node->right);
        delete node;
    }
}

// Add a show item to the catalog
void ShowCatalog::addShowItem(string title, int year, string showRating, float userRating) {
    root = insert(root, title, year, showRating, userRating);
}

// Helper to insert a new show item into the BST
ShowItem* ShowCatalog::insert(ShowItem* node, string title, int year, string showRating, float userRating) {
    if (node == nullptr) {
        return new ShowItem(title, year, showRating, userRating);
    } else if (title < node->title) {
        node->left = insert(node->left, title, year, showRating, userRating);
    } else {
        node->right = insert(node->right, title, year, showRating, userRating);
    }
    return node;
}

// Print the show catalog
void ShowCatalog::printShowCatalog() {
    if (root == nullptr) {
        cout << "Tree is Empty. Cannot print" << endl;
    } else {
        printPreOrder(root);
    }
}

// Helper for preorder traversal
void ShowCatalog::printPreOrder(ShowItem* node) {
    if (node != nullptr) {
        cout << "Show: " << node->title << " "  << node->userRating << endl;
        printPreOrder(node->left);
        printPreOrder(node->right);
    }
}

// Get and print information about a show given its title
void ShowCatalog::getShow(string title) {
    ShowItem* found = find(root, title);
    if (found == nullptr) {
        cout << "Show not found." << endl;
    } else {
        cout << "Show Info:" << endl;
        cout << "==================" << endl;
        cout << "Title :" << found->title << endl;
        cout << "Year :" << found->year << endl;
        cout << "Show Rating :" << found->showRating << endl;
        cout << "User Rating :" << found->userRating << endl;
    }
}

// Helper to find a show by title
ShowItem* ShowCatalog::find(ShowItem* node, const string& title) {
    if (node == nullptr || node->title == title) {
        return node;
    } else if (title < node->title) {
        return find(node->left, title);
    } else {
        return find(node->right, title);
    }
}

// Search shows starting with a given character
void ShowCatalog::searchShows(char titleChar) {
    bool found = false;
    if(existsShowWithChar(root, titleChar)){
        cout<< "Shows that starts with " << titleChar << ":"<<endl;
    }
    searchByChar(root, titleChar, found);
    if (!found) {
        cout << "No show starts with " << titleChar << "." << endl;
    }
}

// Helper to search shows by starting character
void ShowCatalog::searchByChar(ShowItem* node, char titleChar, bool& found) {
    if (node != nullptr) {
        if (toupper(node->title[0]) == toupper(titleChar)) {
            cout << node->title << "(" << node->year << ") "<< node->userRating<<endl;
            found = true;
        }
        searchByChar(node->left, titleChar, found);
        searchByChar(node->right, titleChar, found);
    }
}

// Display the number of shows with a specific rating
void ShowCatalog::displayNumShowRating(int& count, string showRating) {
    countShowsWithRating(root, count, showRating);
}

// Helper to count shows with specific rating
void ShowCatalog::countShowsWithRating(ShowItem* node, int& count, const string& showRating) {
    if (node != nullptr) {
        if (node->showRating == showRating) {
            ++count;
        }
        countShowsWithRating(node->left, count, showRating);
        countShowsWithRating(node->right, count, showRating);
    }
}

// Print all leaf nodes
void ShowCatalog::printLeafNodes() {
    if (root == nullptr) {
        return;
    } else {
        printLeaves(root);
    }
}

// Helper to print leaf nodes
void ShowCatalog::printLeaves(ShowItem* node) {
    if (node != nullptr) {
        if (node->left == nullptr && node->right == nullptr) {
            cout << node->title << endl;
        } else {
            printLeaves(node->left);
            printLeaves(node->right);
        }
    }
}

bool ShowCatalog::existsShowWithChar(ShowItem* node, char titleChar) {
    if (node == nullptr) {
        return false;
    }
    if (toupper(node->title[0]) == toupper(titleChar)) {
        return true;
    }
    // Traverse the left subtree
    if (existsShowWithChar(node->left, titleChar)) {
        return true;
    }
    // If not found in the left subtree, traverse the right subtree
    return existsShowWithChar(node->right, titleChar);
}

