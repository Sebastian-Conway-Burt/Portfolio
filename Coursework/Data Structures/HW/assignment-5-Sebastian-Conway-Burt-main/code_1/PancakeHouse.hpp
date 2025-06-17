/****************************************************************/
/*                     Pancake House Definition                     */
/****************************************************************/
/* LEAVE THIS FILE AS IS! DO NOT MODIFY ANYTHING! =]            */
/****************************************************************/

#pragma once
#ifndef PANCAKE_HOUSE_HPP__
#define PANCAKE_HOUSE_HPP__
#include <iostream>
#include "Register.hpp"
#include "ShopQueue.hpp"

using namespace std;
const int MAX_CUSTOMERS = 20;

class PancakeHouse {
    private:
        int pancakesCooked;
        int profit;
        static int const pancake_types[];

        Register* cashRegister;
        ShopQueue* shopQueue;

    public:
        PancakeHouse();
        ~PancakeHouse();

        ShopQueue* getQueue();
        Register* getStack();
        int getProfit();
        
        void cookPancakes();
        void addOrder(string name, int number_of_pancakes, int type_of_pancakes);
        void updateTotalProfit();
        void printOrders();
        void strikeOrder();
        void refundOrder();
        void closeShop();
};


#endif //PANCAKE_HOUSE_HPP__
