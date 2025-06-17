#include <cstdlib>
#include <iostream>
#include "Register.hpp"


using namespace std;

Register::Register(){
top = -1;
}

/*
 * If the register is full or not
 * @returns 
 */
bool Register::isFull(){
   if(top == SIZE - 1){
      return true;
   }
   
   return false;
}

/*
 * Adds money to stack.
 * @param money: the type of pancake's price times the quantity
 */
void Register::push( int money ){
   // TODO
  
    if (!isFull()) { // Check if the stack has space
        a[++top] = money; // Increment top and add money to the stack
    } else {
        cout << "Stack overflow: " << endl;
    }


}


/*
 * Checks if stack is full. 
 * @returns a bool
 */
bool Register::isEmpty(){
   return (top == -1);
}

/*
 * Looping through the stack array to display amounts
 */
void Register::disp(){
   // TODO
   int tempTop = 0;
   cout<< "top = [" << a[top] << "]" <<endl;
   while(a[tempTop] > 0){
      cout << a[top- 1]<<endl;
      top --;
   }
}

int Register::pop(){
    if (isEmpty()) {
        cout << "Stack empty, cannot pop an item!" << endl;
        return -1; // Indicating the stack is empty
    } else {
        int poppedValue = a[top]; // Retrieve the top value
        top--; 
        return poppedValue; // Return the popped value
    }
}

