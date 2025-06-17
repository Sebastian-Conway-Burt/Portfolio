#include<iostream>
#include <fstream>
#include<cmath>
#include<iomanip>
#include "../code_1/PancakeHouse.hpp"


using namespace std;

int main(int argc, char* argv[]) {
    // DO NOT MODIFY THIS.
    // This is used to read a simulted input from a file. Used in test cases.
    if(argc>1) 
    {
        freopen(argv[1],"r",stdin);
    }
    // DO NOT MODIFY ABOVE.

    PancakeHouse * pancakeHouse = new PancakeHouse;

    while (true) {
        cout << "*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*" << endl;
        cout << "WELCOME TO ASA'S PANCAKE HOUSE!\n";
        cout << "*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*" << endl;
        cout << "============= MENU =============" << endl;
        cout << "1. Make an order" << endl;
        cout << "2. Cook pancakes" << endl;
        cout << "3. Strike order" << endl;
        cout << "4. Refund order" << endl;
        cout << "5. Display orders" << endl;
        cout << "6. Display order total" << endl;
        cout << "7. Close shop" << endl;
        cout << "8. Exit" << endl;
        cout << "================================" << endl;

        int choice;
        cout << "Enter your choice: ";
        cin >> choice;

        switch (choice) {
            case 1: {
                string name;
                int numberOfPancake;
                int typeOfPancake;

                cout << "Enter the name of the customer: ";
                cin.ignore();
                getline(cin, name);

                cout << "Enter the number of pancakes: ";
                cin >> numberOfPancake;

                cout << "Enter the type of pancakes: " << endl;
                cin >> typeOfPancake;

                pancakeHouse->addOrder(name, numberOfPancake, typeOfPancake);
                break;
            }    
            case 2: {
                pancakeHouse->cookPancakes();
                
                break;
            }

            case 3: {
                pancakeHouse->strikeOrder();
                break;
            }
            case 4:{
                pancakeHouse->refundOrder();
                break;
            }

            case 5: {
                cout << "Orders for the pancake house: " << endl;
                pancakeHouse->printOrders();
                break;
            }

            case 6: {
                pancakeHouse->updateTotalProfit();
                cout << "Profit since last check: " << pancakeHouse->getProfit() << endl;
                break;
            }

            case 7: {
                pancakeHouse->closeShop();
                break;
            }

            case 8: {
                cout << "Exiting program.\n";
                return 0;
            }

            default:
                cout << "Invalid choice. Please try again.\n";
            }
        }
        return 0;
    }

