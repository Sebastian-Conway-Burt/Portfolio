#include "../code_1/ShowCatalog.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;
traceNode *traceHead = NULL;
bool verbose = false;
int numAllocs = 0;

void displayMenu() {
    cout << "======Main Menu======" << endl;
    cout << "1. Get a show" << endl;
    cout << "2. Search shows"<< endl;
    cout << "3. Print show catalog"<< endl;
    cout << "4. Display number of show ratings"<< endl;
    cout << "5. Print Leaves" << endl;
    cout << "6. Quit"<< endl;
    cout << "> ";
}

int main(int argc, char** argv) {
    
    // DO NOT MODIFY THIS.
    if(argc>2) 
     {
         freopen(argv[2],"r",stdin);  
     }
    
    int input;
    ShowCatalog* tree = new ShowCatalog();

    // PROCESS THE FILE
    ifstream iFile(argv[1]);
    string line;

    string title;
    int year;
    string showRating;
    float userRating;
    
    while(getline(iFile, line)) {
        stringstream ss(line);
        string field;

        int counter = 0;
        while(getline(ss, field, ',')) {
            if (counter == 0)
                title = field;
            else if (counter == 1)
                year = stoi(field);
            else if (counter == 2)
                showRating = field;
            else if (counter == 3)
                userRating = stof(field);
            counter++;
        }
        tree->addShowItem(title,year,showRating,userRating);
    }

    // PRESENT MENU
    displayMenu();
    cin >> input;

    while (1) {
        if (input == 1) {
            string movie_title;
            cout << "Enter title:" << endl;
            cin.ignore();
            getline(cin, movie_title);
            tree->getShow(movie_title);
        }
        else if (input == 2){
            char titleChar;
            cout << "Enter the title character:" << endl;
            cin.ignore();
            cin >> titleChar;
            tree->searchShows(toupper(titleChar));
        }
        else if (input == 3){
            tree->printShowCatalog();
        }
        else if (input == 4){
            string rating;
            cout << "Enter the show rating to count:" << endl;
            cin.ignore();
            cin >> rating;
            int count = 0;
            tree->displayNumShowRating(count, rating);
            cout << "Number of " << rating << ": " << count << endl;
        }
        else if (input == 5){
            tree->printLeafNodes();
        }
        else if (input == 6){
            cout << "Goodbye!" << endl;
            break;
        }
        displayMenu();
        cin >> input;
    }
    
    
    // MEMCHECK!
    delete tree;
    if (traceHead != nullptr){
        cout << endl << "[ERROR] The program leaked memory. The following nodes are still alive:" << endl;
        traceNode *hold = traceHead;
        while (hold != nullptr){
            printNode(hold);
            hold = hold->next;
        }
    }
    else {
        cout << endl << "[GRADER] No leaked memory (Good)." << endl;
    }
    
    return 0;
}
