# CSCI 2270 – Data Structures - Assignment 1: C++ Essentials

## Objectives

1. Read from and write to a file
2. Use command-line arguments
3. Utilize an array
4. Process a string
5. Utilize an array of structures

## Instructions

Write code to complete Parts A and B. To receive credit for your code, you will need to pass the necessary test cases. Use the following steps to test your code as you work on the assignment:

1. Open up your Linux terminal, navigate to the build directory of this assignment (e.g. `cd build`).
2. Run the `cmake ..` command.
3. Run the `make` command.
4. If there are no compilation errors, two executables will be generated within the build directory: `run_app` and `run_tests`.
5. If you would like to run your program including your app implementation in the `main` function, execute `run_app` from the terminal by typing `./run_app <path-to-input-stock-file> <name-and-path-of-output-file>`. For eg. `./run_app ../stock_2.csv ../output_2.csv`
6. To run the grading tests, execute `run_tests` from the terminal by typing `./run_tests`.

## Part A: Read Data Into an Array of Structs

### Overview

The first part requires that you **read data about stocks from a file**, **insert it into an array of structs**, and **manipulate the data** accordingly. It is mandatory that you use the [`stringstream` class](https://cplusplus.com/reference/sstream/stringstream/stringstream/) when parsing the respective data file.

You will find the function prototypes in `code\fundamentals.hpp`. Following the guidlines in the specifications below, implement the required functions in `code\fundamentals.cpp`. You are required to read from the stock file using the `main()` function in `app/main.cpp` and manipulate the data using `insertStockInfo()`.

### Specifications:

Your program will accomplish the following:
-   Read a `.csv` file containing up to 6 stock records. Each record within the file consists of 6 columns containing data on 5 months of stock prices for various companies in 2022.
    > `csv` stands for comma separated values.
- There are three files you will need to test: `stock_1.csv`, `stock_2.csv` & `stock_3.csv`. You run the program on each file individually.
-   **Store the data** from a stock file and **average stock price** of each company in an _array-of-structs_. The struct is given in `code\fundamentals.hpp` as:
    ```
    struct  stockInfo {
    	string company_name;
    	double stock_prices_arr[5];
    	double average;
    };
    ```

    -   Implement a function named `insertStockInfo()` with the following declaration:

    ```
    void insertStockInfo(stockInfo stocks[], string company_name, double stock_prices_arr[], int index);
    ```

    > -   `stocks[]` is an array-of-structs of type `stockInfo`.
    > -   `company_name` is the name of the company.
    > -   `stock_prices_arr[]` is an array of 5 months of stock prices.
    > -   `index` is the position at which we're storing the item in the array `stocks[]`.

    -   Within the `insertStockInfo()` function, instantiate a local `stockInfo` struct and store the `company_name`, `stock_prices_arr[]` values.

    -   Compute the average of all the stock prices for each company across the 5 months and store it in the aforementioned struct.

    -   Add the struct to the stocks array.

## Part B:  Sorting and Storing Stock Structs

### Overview

In this part, you need to write code that **sorts the array of stock prices for each struct in ascending order** and **prints the updated structs to the standard output as well as the external file**.

Your task is to implement the other function, `displaySortedStocks()` in `code\fundamentals.cpp` with the associated function prototype located in `code\fundamentals.hpp`. The `main()` function should then call the completed `displaySortedStocks` function.

### Specifications:

Implement a function called `displaySortedStocks()` that does the following:

-   Takes in two arguments:
    -   `stock` : stockInfo instance whose stock array we are going to sort. 
    -   `file` : the external file (an `ofstream` object) where we are to store the updated `stock`.

-   Sorts the stock prices (`stock_prices_arr[]`) in _ascending_ order. 
    > **Note:** You are not permitted to use any library functions (i.e. `sort` from the `<algorithm>` library)
-   After sorting the stock prices array, print the stock to the standard output, and also to the input file. The ouput to the file and the standard output should be in the following order: `company_name`, `average` and then the `sorted stock prices`. There is an example later demonstrating the exact format to follow for output.

The function declaration should be as follows:

```
void displaySortedStocks(stockInfo stock, ofstream& file);
```


#### Sample Input and Output:
Assuming that an input file, `stock_1.csv` has the following content:
```
Activision Blizzard Inc,66.44,79.83,77.97,74.41,76.6
Adobe Inc,566.65,455.0,364.18,278.27,340.16
ASML Holding NV ADRs,75.0,150.0,50.0,125.0,100
Advanced Micro Devices Inc,145.14,110.48,75.19,64.46,66.0
```

Running your program using the following command:
```
./run_app stock_1.csv output.csv
```

Running `./run_app stock_1.csv output.csv` should produce the following file, `output.csv`:
```
Activision Blizzard Inc,75.05,66.44,74.41,76.6,77.97,79.83
Adobe Inc,400.852,278.27,340.16,364.18,455,566.65
ASML Holding NV ADRs,100,50,75,100,125,150
Advanced Micro Devices Inc,92.254,64.46,66,75.19,110.48,145.14
```

Follow the same process for `stock_2.csv` and `stock_3.csv`.
