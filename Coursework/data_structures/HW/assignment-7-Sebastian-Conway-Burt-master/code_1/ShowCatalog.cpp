#include "ShowCatalog.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

/* Completed functions. DO NOT MODIFY*/
ShowCatalog::ShowCatalog()
{
    root = nullptr;
}

/* Completed functions. DO NOT MODIFY*/
void destroyNode(ShowItem *current)
{
    if (current != nullptr)
    {
        destroyNode(current->left);
        destroyNode(current->right);

        delete current;
        current = nullptr;
    }
}

/* Completed functions. DO NOT MODIFY*/
ShowCatalog::~ShowCatalog()
{
    destroyNode(root);
}

/* Completed functions. DO NOT MODIFY*/
void printShowHelper(ShowItem *m)
{
    if (m != nullptr)
    {
        cout << "Show: " << m->title << " " << m->userRating << endl;
        printShowHelper(m->left);
        printShowHelper(m->right);
    }
}

/* Completed functions. DO NOT MODIFY*/
void ShowCatalog::printShowCatalog()
{
    if (root == nullptr)
    {
        cout << "Tree is Empty. Cannot print" << endl;
        return;
    }
    printShowHelper(root);
}

/* Completed functions. DO NOT MODIFY*/
ShowItem *getShowHelper(ShowItem *current, string title)
{
    if (current == NULL)
        return NULL;

    if (current->title == title)
        return current;

    if (current->title > title)
        return getShowHelper(current->left, title);

    return getShowHelper(current->right, title);
}

/* Completed functions. DO NOT MODIFY*/
void ShowCatalog::getShow(string title)
{
    ShowItem *node = getShowHelper(root, title);
    if (node != nullptr)
    {
        cout << "Show Info:" << endl;
        cout << "==================" << endl;
        cout << "Title :" << node->title << endl;
        cout << "Year :" << node->year << endl;
        cout << "Show Rating :" << node->showRating << endl;
        cout << "User Rating :" << node->userRating << endl;
        return;
    }

    cout << "Show not found." << endl;
}

/* Completed functions. DO NOT MODIFY*/
ShowItem *addNodeHelper(ShowItem *current, ShowItem *newNode)
{
    if (current == nullptr)
    {
        return newNode;
    }

    if (current->title > newNode->title)
    {
        current->left = addNodeHelper(current->left, newNode);
    }
    else
    {
        current->right = addNodeHelper(current->right, newNode);
    }

    return current;
}

/* Completed functions. DO NOT MODIFY*/
void ShowCatalog::addShowItem(string title, int year, string showRating, float userRating)
{
    if (root == nullptr)
    {
        root = new ShowItem(title, year, showRating, userRating);
        return;
    }

    root = addNodeHelper(root, new ShowItem(title, year, showRating, userRating));
}

/* TODO */
void ShowCatalog::removeShow(std::string title)
{
    root = removeShowHelper(root, title);
}

ShowItem* ShowCatalog::removeShowHelper(ShowItem* current, std::string title)
{
    if (current == nullptr)
    {
        return nullptr;
    }
    else if (title < current->title)
    {
        current->left = removeShowHelper(current->left, title);
    }
    else if (title > current->title)
    {
        current->right = removeShowHelper(current->right, title);
    }
    else
    {
        
        if (current->left == nullptr)
        {
            ShowItem* temp = current->right;
            delete current;
            return temp;
        }
        else if (current->right == nullptr)
        {
            ShowItem* temp = current->left;
            delete current;
            return temp;
        }

       
        
        ShowItem* temp = findMin(current->right);

        
        current->title = temp->title;
        current->year = temp->year;
        current->showRating = temp->showRating;
        current->userRating = temp->userRating;
        current->right = removeShowHelper(current->right, temp->title);
    }
    return current;
}

ShowItem* ShowCatalog::findMin(ShowItem* node)
{
    while (node->left != nullptr)
    {
        node = node->left;
    }
    return node;
}

/* TODO */
void ShowCatalog::rightRotate(std::string title)
{
    
    root = rightRotateHelper(root, title, nullptr);
}

ShowItem* ShowCatalog::rightRotateHelper(ShowItem* current, std::string title, ShowItem* parent)
{
    if (current == nullptr) {
        std::cout << "Show not found." <<endl;
        return nullptr;
    }

    if (current->title == title) {
        ShowItem* y = current->left;
        if (y == nullptr) {
            cout << "Cannot perform right rotation. Left child is null." << endl;
            return current; 
        }
        current->left = y->right;
        y->right = current;
        if (parent != nullptr) {
            if (current == parent->left) {
                parent->left = y;
            } else {
                parent->right = y;
            }
        } else {
            root = y;
        }
        return y;
    }

    if (title < current->title) {
        current->left = rightRotateHelper(current->left, title, current);
    } else {
        current->right = rightRotateHelper(current->right, title, current);
    }

    return current;
}



void ShowCatalog::leftRotate(std::string title)
{
    root = leftRotateHelper(root, title, nullptr);
}

ShowItem* ShowCatalog::leftRotateHelper(ShowItem* current, std::string title, ShowItem* parent)
{
    if (current == nullptr) {
        
            cout << "Show not found." << std::endl;
        return nullptr;
    }

    if (current->title == title) {
        ShowItem* x = current;
        ShowItem* y = x->right;
            if (y == nullptr) {
                cout << "Cannot perform left rotation. Right child is null." << std::endl;
            return current; 
                }
        x->right = y->left;
        y->left = x;
        if (parent != nullptr) {
            if (x == parent->left) {
                parent->left = y;
            } else {
                parent->right = y;
            }
        } else {
            root = y; 
        }
        return y; 
    }

    
    if (title < current->title) {
        current->left = leftRotateHelper(current->left, title, current);
    } else {
        current->right = leftRotateHelper(current->right, title, current);
    }

    return current;
}
