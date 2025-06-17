# CSCI 2270 – Data Structures - Assignment 5 - Stacks and Queues

## Objectives

1. Implement a queue using a linked list via ShopQueue
2. Implement a stack using an array via Register
3. Implement PancakeHouse

## Instructions to run the program

Write code to complete Parts 1, 2, and 3. To receive credit for your code, you will need to pass the necessary test cases. Use the following steps to test your code as you work on the assignment:

 1. Open up your Linux terminal, navigate to the build directory of this assignment (e.g. `cd build`).
 2. Run the `cmake ..` command.
 3. Run the `make` command.
 4. If there are no compilation errors, executables will be generated within the build directory: `run_app` and `run_tests`.
 5. If you would like to run your program including your app implementation in the `main` function, execute `run_app` from the terminal by typing `./run_app`.
 6. To run the grading tests, execute `run_tests` from the terminal by typing `./run_tests`. 

 If you would like to quit the program, hit `ctrl+c` in your terminal. 

## Instructions for the problem

Build a queue for the people that are in line and a stack for the register tickets. Keep track of the profits Asa has generated from his pancake house ventures. 

## Background

A queue is a container of objects (a linear collection) that are inserted and removed according to the first-in first-out (FIFO) principle. On the other hand, a stack is a container of objects that are inserted and removed according to the last-in first-out (LIFO) principle. Both of them can be implemented using arrays or linked lists. In this assignment, queues need to be implemented using linked lists, and stacks should be implemented using arrays.

## Problem
### Overview:
There are a line of people that Asa has to serve at his pancake restaurant. You're helping keep track of the queue of people with their orders and the stack of register tickets. The following files, `ShopQueue.cpp`, `Register.cpp`, and `PancakeHouse.cpp` need to be implemented. 

### Specification:
The PancakeHouse is the main class which integrates the stack and queue that you need to implement. The following methods are provided in `PancakeHouse.cpp`:

`ShopQueue* getQueue()` \
➜ Retrieves the queue of people.

`Register* getStack()`\
➜ Retrieves the stack of register tickets.  

`int getProfit()` \
➜ Retrieves the profit.

You will also have to implement the following functions in `PancakeHouse.cpp` (more details in Part 3):

`void cookPancakes()` \
`void addOrder(string name, int number_of_pancakes, int type_of_pancakes)` \
`void updateTotalProfit()` \
`void printOrders()` \
`void strikeOrder()` \
`void refundOrder()` \
`void closeShop()` 

In the PancakeHouse, the queue and the stack are being used with the required functionality:

Queue: \
	Constructor: Sets the initial values of the queue \
	Destructor: Frees dynamically allocated memory \
	`isEmpty`: Check if the queue is empty \
	`peek`: See who's in the front (to be dequeued) \
	`enqueue`: Add a new person to the back of the queue \
	`dequeue`: Remove a person from the queue’s front \
	`queueSize`: Return the size of the queue 

Stack: \
	Constructor: Sets the initial values of the stack \
	`isFull` : Check if the stack is full \
	`push`: Add a new node to the top of the stack \
	`isEmpty`: Check if the stack is empty \
	`disp`: Shows all the nodes in the stack \
	`pop`: Remove node from the stack 

#### Part 1 - ShopQueue Class (ShopQueue.cpp):
---
This implementation of a queue involves using a linked list. There are two variables you can use for various functions, where the return type is a struct containing the name of the customer, how many pancakes they want, the type of pancakes they want, and a pointer to the next customer in line:
1. `queueFront`: the customer order in ShopQueue that will be dequeued next
2. `queueEnd`: the customer order in ShopQueue that was most recently enqueued

You should work with these variables to implement the following functions:

`ShopQueue()` - Constructor\
➜ Set `queueFront` and `queueEnd` to null.

`~ShopQueue` - Destructor\
➜ While there are customer orders in the queue, dequeue them.

`bool isEmpty()`\
➜ Return true if the queue is empty, false otherwise.

`CustomerOrder* peek()`\
➜ Returns the next customer in line. If the queue is empty, print:
```
Queue empty, cannot peek!
```
followed by a line break and return null. 

`void enqueue(string name, int num_pancakes, int type_of_pancake)`\
➜ Add a customer and their order information to the queue. Adjust the ShopQueue variables accordingly. 

`void dequeue()`\
➜ Remove a customer order from the queue and free memory. Adjust the ShopQueue variables accordingly. If the queue is empty, print:

```
Queue empty, cannot dequeue!
```

followed by a line break.

`int queueSize()`\
➜ Return the size of the queue. Utilize the ShopQueue variables and traverse through the list. 

#### Part 2 - Register Class (Register.cpp):
---
We will use an array to implement the stack. We are keeping track of:
1. `top`: the index corresponding to the top ticket on the register
2. `a[SIZE]`: the actual array containing the money associated with each ticket, where `SIZE` is `20`

`Register()` - Constructor\
➜ Set the `top` of the stack to `0`.

`bool isFull()`\
➜ Run a check to see if the stack is full. 

`push(int money)`\
➜ If there is space in the stack, add the ticket containing the price of the order to the next available slot in the stack. Otherwise, print:
```
Stack overflow: 
```
followed by a line break.

`bool isEmpty()`\
➜ Run a check to see if the stack is empty. 

`void disp()`\
➜ For the top element, print:
```
top = [top]
```
followed by a line break. We then print each ticket in the stack array followed by a line break.

`int pop()`\
➜ Retrieve, remove, and return the top of the stack. If there are no tickets in the register, print:
```
Stack empty, cannot pop an item!
```
followed by a line break. Return `-1` in this case.

#### Part 3 - PancakeHouse Class (PancakeHouse.cpp)
---
Here, you will see how stacks and queues may be utilized within a system. The following variables will be used in the restaurant:
1. `pancakesCooked`: keeping track of all the pancakes which have been cooked by the restaurant
2. `profit`: the total profit thus far of the pancake house
3. `pancake_types[]`: an array containing the different types of pancakes the store offers
4. `cashRegister`: the register of the pancake house
5. `shopQueue`: the queue of customers and their orders

`pancakeHouse()` - Constructor \
➜ Sets `profit` and `pancakesCooked` to `0`. We also want new instances for the `cashRegister` and `shopQueue`. 

`~pancakeHouse()` - Destructor\
➜ Remove the `cashRegister` and `shopQueue`. Handle dynamically allocated memory properly.

`void addOrder()`\
➜ Use the corresponding ShopQueue function to add a customer order to the queue.

`void cookPancakes()`\
➜ Take the next order in the queue. You want to increment the corresponding `PancakeHouse` variables given the amount of pancakes in the order to cook, and add a ticket to the register with the cost of the order. 

`void strikeOrder()`\
➜ Remove the next order in the queue.

`void refundOrder()`\
➜ Remove the top ticket in the register. If there isn't one, we want to print:
```
No money in the cash register!
```
and end the function. If there is, we print:
```
Money refunded = [refund]
```
followed by a line break.

`void closeShop()`\
➜ Traverse through the shop queue and the register by removing all the items in each.

`void updateTotalProfit()`\
➜ Take the tickets that are in the cash register and add the money associated with each ticket to the profit section. Once we add the money to our profit pool, we remove the ticket from the register.

`void printOrders()`\
➜ Traverse through the queue of customer orders and print out all the information in each order. We want to preserve the order in which the customers lined up. 

For each order, print:
```
Customer name: [name]
Number of pancakes ordered: [# of pancakes]
Type of pancakes: [type of pancakes]
```
followed by a line break for each piece of information. 

### Submitting your code:
Write your code and push the changes to your private GitHub repository. Find the associated Assignment 5 dropbox in Canvas. Paste your GitHub repository link and submit.
