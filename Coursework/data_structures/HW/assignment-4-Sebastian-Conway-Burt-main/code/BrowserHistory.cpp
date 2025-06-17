
#include "BrowserHistory.hpp"

BrowserHistory::BrowserHistory()
{
    // No changes needed
}

BrowserHistory::~BrowserHistory()
{
    // No changes needed
}

/*
 * Purpose: Has to detect if a defect is present in the linkedlist pointed by head
 * @param none
 * @return integer length of defect if one exists. If defect not present return -1
 */
int BrowserHistory::findDefectLength(){

    WebPage* fast = head;
    WebPage* slow = head;
    bool isloop = false;

    while(fast != nullptr && fast -> next != nullptr)
    {   
        slow = slow-> next;
        fast = fast->next->next;
        if(slow == fast)
        {
            isloop = true;
            break;
        }
    }
    if (isloop) {
        int length = 0;
        do {
            slow = slow->next;
            length++;
        } while (slow != fast);
        return length;
}
return -1;
}
/*
 * Purpose: Has to remove all the WebPage nodes from [start, end] inclusive.
 * Has to print appropriate messages on cout like below if the linkedlist is empty
 * or if the values of start/ end are improper
 * follow the same order for couts in the writeup - check empty list first, then check wrong start/end values
 * @param integers start and end (1 based indexing not 0 based)
 * @return none
 */
void BrowserHistory::removeWebPages(int start, int end){
    if (head == nullptr) {
        cout << "Browsing history is Empty" << endl;
        return;
    }

    // Check if the start and end values are valid
    int length = 0;
    WebPage* temp = head;
    while (temp != nullptr) {
        length++;
        temp = temp->next;
    }
    if (start < 1 || end > length || start > end) {
        cout << "Invalid start or end values" << endl;
        return;
    }

    WebPage* current = head;
    WebPage* prev = nullptr;
    int index = 1;

    // Traverse to the start position
    while (current != nullptr && index < start) {
        prev = current;
        current = current->next;
        index++;
    }

    // Adjust nextNode to point to the node right after the end of the deletion range
    WebPage* nextNode = current;
    for (int i = start; i <= end && nextNode != nullptr; i++) {
        nextNode = nextNode->next;
    }

    // Delete the nodes from start to end
    while (current != nullptr && index <= end) {
        WebPage* deleteNode = current;
        current = current->next;
        delete deleteNode;
        index++;
    }

    // Re-link the list
    if (prev != nullptr) {
        prev->next = nextNode;
    } else {
        head = nextNode;
    }
}


/*
 * Purpose: Interweave the webpages alternatively into a new linkedlist 
 * starting with the first webpage in the list one
 * Assign the head of the new interweaved list to the head of this BrowserHistory
 * DO NOT create new nodes and copy the data, just use the same nodes from one and two and change pointers
 * If one of them runs out of length append the remaining of the other one at end
 * @param two linkedlist heads one and two
 * @return none
 */
void BrowserHistory::mergeTwoHistories(WebPage *headOne, WebPage * headTwo){
    if(headOne == nullptr){
        head = headTwo;
        return;
    }
    if(headTwo == nullptr){
        head = headOne;
        return;
    }
    WebPage* pointerOne = headOne;
    WebPage* pointerTwo = headTwo;
    WebPage* nextOne = nullptr;
    WebPage* nextTwo = nullptr;
    
    while(pointerOne != nullptr && pointerTwo != nullptr)
    {
        nextOne = pointerOne->next;
        nextTwo = pointerTwo->next;
        pointerOne->next = pointerTwo;
        if (nextOne != nullptr) {
            pointerTwo->next = nextOne;
        }

        // Move the pointers forward
        pointerOne = nextOne;
        pointerTwo = nextTwo;
    }

    // Update the head of the original list
    head = headOne;
       

    
return;

    // TODO END ==================================================
}
