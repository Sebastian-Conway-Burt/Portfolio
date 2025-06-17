#include <gtest/gtest.h>
#include "test_helpers.hpp"



std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

std::string readFileIntoString(std::string fileName){
	std::string line, out;
  	ifstream A;
  	A.open(fileName);
  	if(!A){throw std::runtime_error("could not open file: " + fileName);}
	while(getline(A,line)){
		out = out+line+"\n";
	}
	return out;
}

//////////////////////////////////////////////////
// Stacks, Queue and main driver class functions
//////////////////////////////////////////////////
CustomerOrder* createCustomerOrder(string name, int num_pancakes, int type_of_pancakes){
    return new CustomerOrder{name, num_pancakes, type_of_pancakes};
}

CustomerOrder** createSmallCustomerOrderArray(){
    string temp_names [] = {"Jon", "James", "Adam", "Preston"};
    int num_pancakes [] = {4, 10, 3, 1};
    int type_pancakes [] = {1, 4, 5, 3};
    CustomerOrder **orders = new CustomerOrder*[4];
    for(int x =0; x < 4; x++){
        orders[x] = createCustomerOrder(temp_names[x], num_pancakes[x], type_pancakes[x]);
    }
    return orders;
}

CustomerOrder** createLargeCustomerOrderArray(){
    string temp_names [] = {
        "Jon", "James", "Adam", "Preston", "Jack",
        "Aaron", "Ben", "Shiela", "Violet", "Felix",
        "August", "April", "May", "Ivy", "Willow",
        "Ezra", "Asher", "Charlotte", "Ambrose", "Graham",
        "Extra"
    };
    int num_pancakes [] = {
        4, 10, 3, 1, 5,
        10, 4, 2, 1, 9,
        0, 3, 2, 1, 5,
        10, 4, 2, 4, 1,
        11
    };
    int type_pancakes [] = {
        2, 2, 1, 2, 2,
        3, 2, 2, 1, 3,
        5, 11, 3, 3, 1,
        2, 1, 2, 2, 2,
        3
    };
    CustomerOrder **orders = new CustomerOrder*[21];
    for(int x =0; x < 21; x++){
        CustomerOrder* temp_order = new CustomerOrder;
        temp_order->name = temp_names[x]; 
        temp_order->number_of_pancakes = num_pancakes[x];
        temp_order->type_of_pancakes = type_pancakes[x];
        orders[x] = temp_order;
    }
    return orders;
}

string test_push(int arr[], int length)
{
    Register calc;
    testing::internal::CaptureStdout();
   
    for(int i=0;i<length;i++)
    {
        calc.push(arr[i]);
    }

    cout << "Print stack from top to bottom:" << endl;
    while(!calc.isEmpty()){
        int temp = calc.pop();
        cout << temp << endl;
    }
   
    string output = testing::internal::GetCapturedStdout();
    return output;
    
}

string test_pop(int arr[], int length)
{
    Register calc;
    testing::internal::CaptureStdout();
   
    for(int i=0;i<length;i++)
    {
        calc.push(arr[i]);
    }
    int flag = 0;
    cout << "Print stack from top to bottom:" << endl;
    if(!calc.isEmpty()){
        while(!calc.isEmpty()){
            int num = calc.pop();
            cout << num << endl;
            flag++;
            if(flag > 10000){
                break;
            }
        }
    }
   
    string output = testing::internal::GetCapturedStdout();
    return output;
    
}

string test_pop_no_elements(int arr[], int length){
    Register calc;
    testing::internal::CaptureStdout();
   
    for(int i=0;i<length;i++)
    {
        calc.push(arr[i]);
    }
    int flag = 0;
    cout << "Print stack from top to bottom:" << endl;
    if(!calc.isEmpty()){
        while(!calc.isEmpty()){
            int num = calc.pop();
            cout << num << endl;
            flag++;
            if(flag > 10000){
                break;
            }
        }
    }
    calc.pop();
    string output = testing::internal::GetCapturedStdout();
    return output;
}


string test_all(int arr[], int length)
{
    Register calc;
    testing::internal::CaptureStdout();
    
   
    for(int i=0;i<2;i++)
    {
        cout<<"push: "<<arr[i]<<endl;
        calc.push(arr[i]);
    }
    
    cout<<"poping thrice "<<endl;
    calc.pop();
    calc.pop();
    calc.pop();
    
    for(int i=2;i<length;i++)
    {
        cout<<"push: "<<arr[i]<<endl;
        calc.push(arr[i]);
    }
    
    cout << "Print stack from top to bottom:" << endl;
    if(!calc.isEmpty()){
        while(!calc.isEmpty()){
            int num = calc.pop();
            if(num != -1){
                cout << num << endl;
            }
            else{
                break;
            }
        }
    }
   
    string output = testing::internal::GetCapturedStdout();
    return output;
    
}

string test_consQ(){
    testing::internal::CaptureStdout();
    ShopQueue q;
    CustomerOrder* front = q.peek();
    CustomerOrder*  back = q.getQueueEnd();

    if(front == nullptr and back == nullptr) cout << "Correct!" << endl;
    else cout << "front and back are not -1, check constructor" << endl;
    string output = testing::internal::GetCapturedStdout();
    return output;
}


string test_enQ(CustomerOrder* customers[], int length){
    
    ShopQueue queue;
    testing::internal::CaptureStdout();
   
    for(int i = 0; i < length; i++)
    {
        queue.enqueue(customers[i]->name,customers[i]->number_of_pancakes, customers[i]->type_of_pancakes );
    }
    CustomerOrder* front = queue.peek();

    cout << "Print queue from front to back:" << endl;
    while(front && front->next){
        cout << front->name << " " << front->number_of_pancakes<< " " << front->type_of_pancakes << endl;
        queue.dequeue();
        front = queue.peek();
    }
    front = queue.peek();
    if(front)
        cout << front->name << " " << front->number_of_pancakes<< " " << front->type_of_pancakes << endl;
    queue.dequeue();

    string output = testing::internal::GetCapturedStdout();
    return output;
}


string test_deQ(CustomerOrder* customers[], int length, int dequeue)
{
    ShopQueue queue;
    testing::internal::CaptureStdout();
   
    for(int i = 0; i < length; i++)
    {
        queue.enqueue(customers[i]->name,customers[i]->number_of_pancakes, customers[i]->type_of_pancakes );
    }
    
    cout << "dequeuing "<< dequeue <<" times"<<endl;
    for(int i = 0; i < dequeue; i++)
    {
        queue.dequeue();
    }
    

    if(length > dequeue)
    {
        CustomerOrder* front = queue.peek();
        cout << "Print queue from front to back:" << endl;
        while(front && front->next){
            cout << front->name << " " << front->number_of_pancakes<< " " << front->type_of_pancakes << endl;
            queue.dequeue();
            front = queue.peek();
        }
        front = queue.peek();
        if(front)
            cout << front->name << " " << front->number_of_pancakes<< " " << front->type_of_pancakes << endl;
        queue.dequeue();

    }
    
    string output = testing::internal::GetCapturedStdout();
    return output;   
}

string test_peekQ(CustomerOrder* customers[], int length, int deQ)
{
    ShopQueue queue; //by default size is 5 so not passing length to constructor
    testing::internal::CaptureStdout();
     
    cout << "enqueuing " << length << " elements" << endl;
    for(int i = 0; i < length; i++)
    {
        queue.enqueue(customers[i]->name,customers[i]->number_of_pancakes, customers[i]->type_of_pancakes );
    }
    
    cout << "dequeuing " << deQ << " times" << endl;
    for(int i = 0; i < deQ; i++)
    {
        queue.dequeue();
    }
    
    cout << "peeking in the queue" << endl;
    CustomerOrder* it = queue.peek();
    
    if(it != nullptr)
    {
        cout << "Front of the queue: " << it->name << " " << it->number_of_pancakes<< " " << it->type_of_pancakes << endl;
    }
    string output = testing::internal::GetCapturedStdout();

    return output;
}
        
string test_allQ(CustomerOrder* customers[], int names_length, string* op, int length)
{
    ShopQueue q;
    testing::internal::CaptureStdout();

    int enq = 0;
    for(int i = 0; i < length; i++)
    {
        if(op[i] == "E")
        {
            CustomerOrder* curr_customer = customers[i%names_length];
            cout << "Enqueue(" << curr_customer->name << ")" << endl;
            q.enqueue(curr_customer->name, curr_customer->number_of_pancakes, curr_customer->type_of_pancakes );
            enq++;
        }
        else if(op[i] == "D")
        {
            cout << "Dequeue" << endl;
            q.dequeue();
        }
        else if(op[i] == "P")
        {
            cout << "Peek" << endl;
            CustomerOrder* it = q.peek();
            if(it != nullptr){
                cout << it << endl;
            }   
        }
    }    
    string empty = q.isEmpty() ? "true" : "false";
    cout << "Queue is empty? " << empty << endl;
    cout << "Queue size: " << q.queueSize() << endl;
    if(q.queueSize() > 0){
        CustomerOrder* front = q.peek();
        CustomerOrder* end = q.getQueueEnd();
        if(front)
            cout << "Queue Front: " << front->name << " " << front->number_of_pancakes << " " << front->type_of_pancakes << endl;
        if(end)
            cout << "Queue End: " << end->name << " " << end->number_of_pancakes << " " << end->type_of_pancakes << endl;
    }
    
    string output = testing::internal::GetCapturedStdout();
    return output;
}

string test_create_customers(CustomerOrder** orders, int length){
    PancakeHouse * ph = new PancakeHouse();
    testing::internal::CaptureStdout();
    for(int x=0; x < length; x++){
        ph->addOrder(orders[x]->name, orders[x]->number_of_pancakes, orders[x]->type_of_pancakes);
    }
    cout << "Size: "<< ph->getQueue()->queueSize() << endl;
    string output = testing::internal::GetCapturedStdout();
    return output;
}

string test_close_shop(CustomerOrder** orders, int length){
    PancakeHouse * ph = new PancakeHouse();
    testing::internal::CaptureStdout();
    for(int x=0; x < length; x++){
        ph->addOrder(orders[x]->name, orders[x]->number_of_pancakes, orders[x]->type_of_pancakes);
    }
    cout << "Size: " << ph->getQueue()->queueSize() << endl;
    //Cook half of the pancakes
    for(int x=0; x<length/2; x++){
        ph->cookPancakes();
    }
    ph->closeShop();
    cout << "Queue Size: "<< ph->getQueue()->queueSize() << endl;
    cout << "Is register empty?: " << (ph->getStack()->isEmpty() ? "True" : "False");
    string output = testing::internal::GetCapturedStdout();
    return output;
}

string test_cook_orders(int num_customers){
    PancakeHouse *p = new PancakeHouse();
    testing::internal::CaptureStdout();
    p->cookPancakes();
    p->updateTotalProfit();
    cout << p->getProfit() << "\n";
    if(num_customers==0){
        string output = testing::internal::GetCapturedStdout();
        delete p;
        return output;
    }
    CustomerOrder** orders = createSmallCustomerOrderArray();
    for(int i = 0; i < num_customers; i++){
        p->addOrder(orders[i]->name, orders[i]->number_of_pancakes, orders[i]->type_of_pancakes);
    }
    for(int i = 0; i < num_customers; i++){
        p->cookPancakes();
    }
    p->updateTotalProfit();
    cout << p->getProfit() << "\n";
    string output = testing::internal::GetCapturedStdout();
    delete p;
    return output;
    
}

string test_refund(int num_customers){
    PancakeHouse *p = new PancakeHouse();
    testing::internal::CaptureStdout();
    CustomerOrder** orders = createSmallCustomerOrderArray();
    for(int i = 0; i < num_customers; i++){
        p->addOrder(orders[i]->name, orders[i]->number_of_pancakes, orders[i]->type_of_pancakes);
    }
    p->cookPancakes();
    p->refundOrder();
    for(int i = 0; i < num_customers-1; i++){
        p->cookPancakes();
    }
    p->updateTotalProfit();
    cout << p->getProfit() << "\n";
    string output = testing::internal::GetCapturedStdout();
    delete p;
    return output;
    
}

void deleteCustomerOrders(CustomerOrder** orders, int length){
    for(int x = 0; x< length; x++){
        delete orders[x];
    }
    delete[] orders;
}