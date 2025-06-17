#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include "../code_1/browserHistory.hpp"

using namespace std;

void displayMenu();

int main(int argc, char* argv[]) {
    if (argc > 1) {
        freopen(argv[1], "r", stdin);
    }

    int input = 0;
    BrowserHistory myBrowserHistory;
    while (input != 6) {
        displayMenu();
        cin >> input;
        cin.ignore(); // Ignore the newline left in the input buffer by cin

        switch (input) {
            case 1: {
                    myBrowserHistory.buildBrowserHistory();
                    myBrowserHistory.displayHistory();
                }
                break;
            case 2:
                myBrowserHistory.displayHistory();
                break;
case 3: {
    string newUrl, prevUrl;
    int newId;

    cout << "Enter the new web page's url:\n";
    getline(cin, newUrl);

    // Loop to check for duplicate ID and take valid input
    do {
        cout << "Enter the new web page's id:\n";
        cin >> newId;
        cin.ignore(); // To ignore the newline left in the input buffer.

        // Check if ID exists
        if(myBrowserHistory.searchPageByID(newId) != nullptr)
        {
            cout << "This ID already exists. Try again.\n";
        }
    } while (myBrowserHistory.searchPageByID(newId) != nullptr); // Continue until a unique ID is entered

    WebPage* previousPage = nullptr;
    do {
        cout << "Enter the previous page's url (or First):\n";
        getline(cin, prevUrl);
        previousPage = myBrowserHistory.searchPageByURL(prevUrl);
        if (previousPage == nullptr && prevUrl != "First") {
            cout << "INVALID(previous page url)... Please enter a VALID previous page url!" << endl;
        }
    } while (previousPage == nullptr && prevUrl != "First");

    // Create the new page and add it to the history.
    WebPage* newPage = new WebPage{newId, 0, newUrl, "", nullptr};
    myBrowserHistory.addWebPage(previousPage, newPage);

    break;
}


            case 4: {
                string url, ownerName;
                WebPage* foundPage = nullptr;
                while(foundPage == nullptr)
                {
                cout << "Enter url of the web page to add the owner:\n";
                getline(cin, url);
                foundPage = myBrowserHistory.searchPageByURL(url);
                if (foundPage != nullptr) {
                    cout << "Enter the owner:";
                    getline(cin, ownerName);
                    myBrowserHistory.addOwner(url, ownerName);
                } 
                }
                break;
            }
            case 5: {
    string url;
    WebPage* pagePtr = nullptr;
    bool pagefound = false;

    while (pagefound == false) {
        cout << "Enter url of the web page to check the view count: \n";
        getline(cin, url);

        pagePtr = myBrowserHistory.searchPageByURL(url);
        if (pagePtr == nullptr) {
            cout << "Page not found. Try again.\n";
        } else {
            pagefound = true;
        }
    }

    cout << "View count for URL - " << url << " is " << pagePtr->views << "\n"; 

    break;
}


            case 6:
                cout << "Quitting...Goodbye!\n";
                return 0;
        }
    }

}

void displayMenu() {
    cout << "Select a numerical option:" << endl;
    cout << "+=====Main Menu=========+" << endl;
    cout << " 1. Build history " << endl;
    cout << " 2. Display history " << endl;
    cout << " 3. Add web page " << endl;
    cout << " 4. Add owner" << endl;
    cout << " 5. View count for a web page" << endl;
    cout << " 6. Quit " << endl;
    cout << "+-----------------------+" << endl;
    cout << "#> ";
}
