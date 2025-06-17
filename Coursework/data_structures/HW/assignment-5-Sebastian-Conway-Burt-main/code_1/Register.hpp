/****************************************************************/
/*                    Stack Class                      */
/****************************************************************/
/* LEAVE THIS FILE AS IS! DO NOT MODIFY ANYTHING! =]            */
/****************************************************************/
#pragma once

#define PANCAKESTACK_HPP__

#include <iostream>

const int SIZE = 20;

class Register{
   private:
      int top; // Index for top element and total count
      int a[SIZE]; // Stack array
   
   public:
      Register();

      bool isEmpty();
      bool isFull();
      void push( int price);
      int pop();
      void disp();
};

