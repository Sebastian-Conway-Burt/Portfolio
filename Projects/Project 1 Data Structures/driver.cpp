#include <iostream>
#include "hash.hpp"
#include "PriorityQueue.hpp"
#include <vector>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;
using namespace std::chrono;

void displayMenu() // do not modify
{
    cout << "------------------" << endl;
    cout << "1: Build the data structure (execute this option one time)" << endl;
    cout << "2: Add a review" << endl;
    cout << "3: Retrieve most recent review for a restaurant" << endl;
    cout << "4: Pop most recent review for a restaurant" << endl;
    cout << "5: Print reviews for a restaurant" << endl;
    cout << "6: Display number of collisions" << endl;
    cout << "7: Display table results" << endl;
    cout << "8: Exit" << endl;
    cout << "------------------" << endl;
}

int main(int argc, char* argv[])
{
    if (argc <3)
    {
        cout<<"need correct number of arguments"<<endl;
    }
	
    string fname = argv[1];
    int tableSize = stoi(argv[2]);
    int ch = 0;
    string chs;
    HashTable ht(5);
	bool built = false;
    string Resturaunt_Name;
    while(ch!=8)
    {
        displayMenu();
        cout << "Enter your choice >>";
        getline(cin, chs);
        ch = stoi(chs);
        switch (ch)
        {
            case 1:
			{
				if (!built) {
                    ht.setup(fname);
                    built = true;
                    cout << "Data structure built." << endl;
                } else {
                    cout << "Data structure has already been built." << endl;
                }
				break;
            }
            case 2:
			{
                {
                    cout << "Enter restaurant name, customer, review, and time (comma separated): ";
                    string restName, customer, review, timeStr;
                    getline(cin, restName, ',');
                    getline(cin, customer, ',');
                    getline(cin, review, ',');
                    getline(cin, timeStr);
                    int time = stoi(timeStr);
                    ReviewInfo newReview = {restName, review, customer, time};
                    ht.insertItem(newReview);
                    cout << "Review added." << endl;
                }
				break;
            }
            case 3:
			{
                cout << "Enter restaurant name: ";
                getline(cin, Resturaunt_Name);
                {
                    node* foundNode = ht.searchItem(Resturaunt_Name);
                    if (foundNode) {
                        foundNode->pq.peek();
                    } else {
                        cout << "No record found." << endl;
                    }
                }
                cin.clear();
				break;
            }
            case 4:
			{
                cout << "Enter restaurant name to pop the most recent review: ";
                getline(cin, Resturaunt_Name);
                {
                    node* foundNode = ht.searchItem(Resturaunt_Name);
                    if (foundNode) {
                        foundNode->pq.pop();
                        cout << "Most recent review removed." << endl;
                    } else {
                        cout << "No record found." << endl;
                    }
                }
                cin.clear();
				break;
            }
            case 5:
                cout << "Enter restaurant name to print reviews: ";
                getline(cin, Resturaunt_Name);
                {
                    node* foundNode = ht.searchItem(Resturaunt_Name);
                    if (foundNode) {
                        foundNode->pq.print();
                    } else {
                        cout << "No record found." << endl;
                    }
                }
                break;
            case 6:
                cout << "Number of collisions: " << ht.getNumCollision() << endl;
                break;
            case 7:
                ht.displayTable();
                break;
            case 8:
                cout << "Exiting program." << endl;
                return 0;
                cout << "Enter a valid option." << endl;
                break;
        }
    }
}
