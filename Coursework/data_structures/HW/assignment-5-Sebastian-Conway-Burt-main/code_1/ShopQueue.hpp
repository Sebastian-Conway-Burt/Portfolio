/****************************************************************/
/*                    Queue Class                      */
/****************************************************************/
/* LEAVE THIS FILE AS IS! DO NOT MODIFY ANYTHING! =]            */
/****************************************************************/
#pragma once

#define SHOPQUEUE_HPP__

#include <string>
struct CustomerOrder {
    std::string name;
    int number_of_pancakes;
    int type_of_pancakes;
    CustomerOrder *next;
};

class ShopQueue {
  private:
    CustomerOrder* queueFront;
    CustomerOrder* queueEnd;
  public:
    ShopQueue();
    ~ShopQueue();
    bool isEmpty();
    void enqueue(std::string name, int num_pancakes, int type_of_pancake);
    void dequeue();
    int queueSize();

    CustomerOrder* peek();
    CustomerOrder* getQueueEnd();
    ShopQueue* getQueue();

};