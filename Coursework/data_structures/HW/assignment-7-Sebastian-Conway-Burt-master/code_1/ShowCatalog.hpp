#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <cstring>
#include <sstream>

using namespace std;


struct ShowItem{
    string title;
    int year;
    string showRating;
    float userRating;

    ShowItem* left = NULL;
    ShowItem* right = NULL;

    ShowItem(string t, int y, string sr, float ur) {
        title = t;
        year = y;
        showRating = sr;
        userRating = ur;
    }

    void *operator new(std::size_t bytes);
    void operator delete(void *);
};

class ShowCatalog{
    private:
        ShowItem* root;

    public:
        ShowCatalog();

        ~ShowCatalog();

        void printShowCatalog();

        void addShowItem(std::string title, int year, string showRating, float userRating);

        void getShow(std::string title);

        void removeShow(string title);

        ShowItem* removeShowHelper(ShowItem* current, std::string title);

        ShowItem* findMin(ShowItem* node);

        void rightRotate(string title) ;

        ShowItem* rightRotateHelper(ShowItem* current, string title, ShowItem* parent);

        void leftRotate(string title) ;

        ShowItem* leftRotateHelper(ShowItem* current, string title, ShowItem* parent);

};


//----------------- DONOT change the code given below --------------------------//
/**************** you can ignore this ******************/
/*     TraceNode memcheck definitions                  ***********/
/*     This will keep track of memory allocation and deallocation*/
/*     This is required for overloading new and delete **********/
/****************************************************************/

struct traceNode
{
  std::string pType;
  void *ptr;
  traceNode *next = NULL;
  traceNode(){}
  traceNode(std::string t, void *p, traceNode *n) : pType(t), ptr(p), next(n) {}
};

extern traceNode *traceHead;
extern bool verbose;
extern int numAllocs;

inline traceNode * addTraceNode(std::string pType, void *ptr, traceNode *traceHead)
{
  traceNode *nn = new traceNode(pType, ptr, traceHead);
  return nn;
}

inline traceNode * removeTraceNode(traceNode *traceHead, void *ptr)
{

  if (traceHead == nullptr){
    std::cerr << "[ERROR] Attempted to delete a node that was never allocated." << std::endl;
    exit(1);
  }
  else if (traceHead->ptr == ptr){
    traceNode *tmp = traceHead->next;
    delete traceHead;
    return tmp;
  }
  else{
    traceHead->next = removeTraceNode(traceHead->next, ptr);
    return traceHead;
  }
}
//----------------------Overloads tree nodes---------------------------
inline void* ShowItem::operator new(std::size_t bytes)
{
  numAllocs += 1;
  void * buf = malloc(bytes);
  memset(buf, 0x0, bytes);

  if(verbose){
    std::cout << "[GRADER] Created TreeNode with address: " << buf << std::endl;
  }

  traceHead = addTraceNode("BST", buf, traceHead);

  return buf;
}

inline void ShowItem::operator delete(void *ptr)
{
  numAllocs -= 1;

  if (verbose){
      std::cout << "[GRADER] Deleted TreeNode with address: " << ptr << std::endl;
  }

  traceHead = removeTraceNode(traceHead, ptr);

  free(ptr);
}

inline void printNode(traceNode * n)
{
  if(n->pType=="BST"){
    ShowItem *tn = (ShowItem*) n->ptr;
    cout << "    MovieNode <" << tn->title << ">" << endl;
  }
}
