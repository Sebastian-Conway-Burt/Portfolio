#include <gtest/gtest.h>
#include "test_helpers.hpp"

#include "../code_1/ShowCatalog.hpp"


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


traceNode *traceHead = NULL;
bool verbose = false;
int numAllocs = 0;

string test_add(string* name,int* year, string* showRating, float* userRating, int len)
{
    ShowCatalog* tree = new ShowCatalog();
    testing::internal::CaptureStdout();
    
    for(int i=0;i<len;i++)
    {
        cout << "[GRADER] Adding:(" <<name[i]<<", "<<year[i]<<", "<<showRating[i]<<", "<<userRating[i] <<")"<< endl;
        tree->addShowItem(name[i],year[i],showRating[i],userRating[i]);
    }
    cout<<"[GRADER] Printing show inventory:"<<endl;
    tree->printShowCatalog();
    
     // MEMCHECK!
    delete tree;
    if (traceHead != nullptr){
        cout << endl << "[ERROR] The program leaked memory. The following nodes are still alive:" << endl;
        traceNode *hold = traceHead;
        while (hold != nullptr){
            printNode(hold);
            hold = hold->next;
        }
    }
    else {
        cout << endl << "[GRADER] No leaked memory (Good)." << endl;
    }
     
    string output = testing::internal::GetCapturedStdout();
    return output;
}


string test_find(string* title,string* showRating, int* year, float* userRating, int len, string key)
{
    ShowCatalog* tree = new ShowCatalog();
    testing::internal::CaptureStdout();
    
    for(int i=0;i<len;i++)
    {
        cout << "[GRADER] Adding:("<<title[i]<<", "<<year[i]<<", "<<userRating[i] <<", " << showRating[i] <<")"<< endl;
        tree->addShowItem(title[i],year[i],showRating[i],userRating[i]);
    }
    std::cout << "[GRADER] Finding: " <<key<< std::endl;
    tree->getShow(key);
    
      // MEMCHECK!
    delete tree;
    if (traceHead != nullptr){
        cout << endl << "[ERROR] The program leaked memory. The following nodes are still alive:" << endl;
        traceNode *hold = traceHead;
        while (hold != nullptr){
            printNode(hold);
            hold = hold->next;
        }
    }
    else {
        cout << endl << "[GRADER] No leaked memory (Good)." << endl;
    }
     
    string output = testing::internal::GetCapturedStdout();
    return output;
}

string test_query(string* title,string* showRating, int* year, float* userRating, char titleChar, int len)
{
    ShowCatalog* tree = new ShowCatalog();
    testing::internal::CaptureStdout();
    
    for(int i=0;i<len;i++)
    {
        cout << "[GRADER] Adding:("<<title[i]<<", "<<year[i]<<", "<<userRating[i] <<", " << showRating[i] <<")"<< endl;
        tree->addShowItem(title[i],year[i],showRating[i],userRating[i]);
    }
    cout << "[GRADER] Searching shows" << endl;
    tree->searchShows(titleChar);
    
      // MEMCHECK!
    delete tree;
    if (traceHead != nullptr){
        cout << endl << "[ERROR] The program leaked memory. The following nodes are still alive:" << endl;
        traceNode *hold = traceHead;
        while (hold != nullptr){
            printNode(hold);
            hold = hold->next;
        }
    }
    else {
        cout << endl << "[GRADER] No leaked memory (Good)." << endl;
    }
    
    string output = testing::internal::GetCapturedStdout();
    return output;
}


string test_displayNumShowRating(ShowCatalog* tree,  string showRating)
{
    testing::internal::CaptureStdout();

    int count = 0;

    tree->displayNumShowRating(count , showRating);

	cout << "Number of " << showRating << ": " << count << endl;
    
    string output = testing::internal::GetCapturedStdout();
    return output;
}


string test_printLeafNodes(string* name,int* year, string* showRating, float* userRating, int len)
{
    ShowCatalog* tree = new ShowCatalog();
    testing::internal::CaptureStdout();
    
    for(int i=0;i<len;i++)
    {
        tree->addShowItem(name[i],year[i],showRating[i],userRating[i]);
    }
    
    cout << "[GRADER] Result of movies.printLeafNodes is:" << endl;
    tree->printLeafNodes();
    cout << endl;
        
    // MEMCHECK!
    delete tree;
    if (traceHead != nullptr){
        cout << endl << "[ERROR] The program leaked memory. The following nodes are still alive:" << endl;
        traceNode *hold = traceHead;
        while (hold != nullptr){
            printNode(hold);
            hold = hold->next;
        }
    }
    else {
        cout << endl << "[GRADER] No leaked memory (Good)." << endl;
    }
    
    string output = testing::internal::GetCapturedStdout();
    return output;
}