/*************************************************************/
/*                BrowserHistory Definition                  */
/*************************************************************/
/* TODO: Implement the member functions of BrowserHistory    */
/*     This class uses a linked-list of WebPage structs to   */
/*     represent the schedule of web pages                   */
/*************************************************************/

#include "browserHistory.hpp"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>

using namespace std;

// Constructor for empty linked list
BrowserHistory::BrowserHistory() {
    head = nullptr;
}

// Check if list is empty
bool BrowserHistory::isEmpty() {
    return (head == NULL);
}

// Prints the current list of pages in the given format
void BrowserHistory::displayHistory() {
    WebPage* temp = head;

    if (temp == nullptr) {
        cout << "== CURRENT BROWSER HISTORY ==" << endl;
        cout << "Empty History" << endl;
        cout << "NULL" << endl;
        cout << "===" << endl;
        return;
    }

    cout << "== CURRENT BROWSER HISTORY ==" << endl;

    while (temp != nullptr) {
        cout << "[ID:: " << temp->id << "]-" << "(URL::" << temp->url << ") -> ";
        temp = temp->next;
    }

    cout << "NULL" << endl;
    cout << "===" << endl;
}

// Add a new webpage to the browser history LL
void BrowserHistory::addWebPage(WebPage* previousPage, WebPage* newPage) {
    if (previousPage == nullptr) {
        newPage->next = head;
        head = newPage;
        cout  <<  "adding: "  <<  "["  <<  newPage->id  <<  "]-"  <<  newPage->url  <<  " (HEAD)\n";
    } else {
        WebPage* tmp = head;
        while (tmp != nullptr && tmp != previousPage) {
            tmp = tmp->next;
        }
        if(tmp != nullptr){
        newPage->next = tmp->next;
        tmp->next = newPage;
        cout << "adding: [" << newPage->id << "]-" << newPage->url << " (prev: [" << previousPage->id << "])\n";
    }
    }
}

// Populates the BrowserHistory with the predetermined pages
void BrowserHistory::buildBrowserHistory() {
    WebPage* colorado = new WebPage{10, 0, "https://www.colorado.edu/", "", nullptr};
    WebPage* wikipedia = new WebPage{11, 0, "https://www.wikipedia.org/", "", nullptr};
    WebPage* brilliant = new WebPage{12, 0, "https://brilliant.org/", "", nullptr};
    WebPage* khanAcademy = new WebPage{13, 0, "https://www.khanacademy.org/", "", nullptr};
    WebPage* numberphile = new WebPage{14, 0, "https://www.numberphile.com/", "", nullptr};

    addWebPage(nullptr, colorado);
    addWebPage(colorado, wikipedia);
    addWebPage(wikipedia, brilliant);
    addWebPage(brilliant, khanAcademy);
    addWebPage(khanAcademy, numberphile);
}

// Search the BrowserHistory for the specified web page by ID
WebPage* BrowserHistory::searchPageByID(int id) {
    WebPage* current = head;
    while (current != nullptr) {
        if (current->id == id) {
            return current;
        }
        current = current->next;
    }
    return nullptr;
}

// Search the BrowserHistory for the specified web page by the URL
WebPage* BrowserHistory::searchPageByURL(std::string url) {
    WebPage* current = head;
    while (current != nullptr) {
        if (current->url == url) {
            return current;
        }
        current = current->next;
    }
    return nullptr;
}

// Give an owner to a web page
void BrowserHistory::addOwner(string url, string owner) {
    WebPage* Temp = searchPageByURL(url);
    if (Temp != nullptr) {
        Temp->owner = owner;
        cout << "The owner (" << Temp->owner << ") has been added for the ID - "<< Temp->id << "\n";
    } 
    else{
        cout << "Page not found\n";
    }

}



void BrowserHistory::updateViews(string url) {
   /* WebPage* current = head;
    int pageviews = 0;
    while(current != nullptr){
        if(current->url == url){
        pageviews ++;
        }
        current = current->next;
    }
    current = head; // This reset is unnecessary if you break after finding the URL
    while(current!= nullptr){
        if(current->url == url){
            current->views = pageviews;
            cout << "View count for URL - " << url << " is " << current->views << endl;
            break; // Should break after updating the views
        }
        current = current->next;
    }*/

    WebPage *tmp = head;
    while(tmp != nullptr){
    if(tmp->url == url){
    tmp->views++;
}
    tmp = tmp->next;
    }
}

