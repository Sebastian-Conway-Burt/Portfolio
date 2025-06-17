#include "ShopQueue.hpp"
#include <iostream>

using namespace std;

ShopQueue::ShopQueue() {
queueFront = nullptr;
queueEnd = nullptr;
}

ShopQueue::~ShopQueue() {

}

/**
 * Checks if the shopqueue is empty or not
 * @returns Whether its empty or not
 */ 
bool ShopQueue::isEmpty() {
   // TODO
   if(queueFront != nullptr)
   {
      return false;
   }
   return true;
}

/**
 * Looks at the shopqueue and returns the most 'urgent' order on the queue. No elements should be removed.
 * @return A customer order
 */
CustomerOrder* ShopQueue::peek() {
  bool empty = isEmpty();
   if(empty == false){
      return queueFront;
   } else{
   cout << "Queue empty, cannot peek!\n";
   
   return nullptr;
   }
}

/**
 * Adds the customers to the queue.
 * @param name The name of the customer to add.
 * @param num_pancakes The number of pancakes to add
 * @param type_of_pancake The type of pancake to add. 
 * {1, 2, 5, 3, 7}
 */
void ShopQueue::enqueue(string name, int num_pancakes, int type_of_pancake) {
    CustomerOrder* newOrder = new CustomerOrder; 
    
    newOrder->name = name;
    newOrder->number_of_pancakes = num_pancakes;
    newOrder->type_of_pancakes = type_of_pancake;
    newOrder->next = nullptr; 

    if (queueEnd != nullptr) {
        queueEnd->next = newOrder;
    } else {
        queueFront = newOrder;
    }

    // Update the end of the queue to be the new order
    queueEnd = newOrder;
}


void ShopQueue::dequeue() {
    if (isEmpty()) {
        cout << "Queue empty, cannot dequeue!" << endl;
        return;
    } else {
        CustomerOrder* tempOrder = queueFront;
        queueFront = queueFront->next;
        if (queueFront == nullptr) {
            queueEnd = nullptr;
        }

        
        delete tempOrder;
    }
}

/**
 * Should return the total number of customers in the queue. 
 * @returns The total number of elements
 */
int ShopQueue::queueSize() {
    CustomerOrder* current = queueFront;
    int size = 0;
    
    while (current != nullptr) {
        size++;
        current = current->next;
    }
    return size; 
}



/**
 * Just returns the end of the queue. Used for testing. Do not touch! :)
 */ 
CustomerOrder* ShopQueue::getQueueEnd(){
   return queueEnd;
}