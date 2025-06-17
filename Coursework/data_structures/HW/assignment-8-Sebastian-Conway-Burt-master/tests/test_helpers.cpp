#include <gtest/gtest.h>
#include "test_helpers.hpp"

#include "../code_1/Graph.hpp"


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


string test_addVertex(string* name,int len)
{
    Graph g;
    testing::internal::CaptureStdout();
    
   for(int i=0;i<len;i++)
    {
        cout << "[GRADER] Adding:" <<name[i]<< endl;
        g.addVertex(name[i]);
    }
    
    vector<vertex*> vertices = g.getVertices();
    cout<<"added vertices are:"<<endl;
    for(unsigned int i=0; i < vertices.size(); i++){
        cout<< vertices[i]->name << endl;
    }
    string output = testing::internal::GetCapturedStdout();
    return output;
}

string test_addShowEdge(string* name,int len, string* edge, int elen)
{
    Graph g;
    testing::internal::CaptureStdout();
    string delimiter = "->";
    for(int i=0;i<len;i++)
    {
        cout << "[GRADER] Adding Vertex:" <<name[i]<< endl;
        g.addVertex(name[i]);
    }
    
    for(int i=0;i<elen;i++)
    {
        int pos = edge[i].find(delimiter);
        string v1 = edge[i].substr(0, pos-1);
        string v2 = edge[i].substr(pos + delimiter.length(), edge[i].length());
        cout<< "[GRADER] Adding edge between: "<<v1<<"   "<<v2<<endl;
        g.addEdge(v1,v2);
    
    }
    g.displayEdges();
    string output = testing::internal::GetCapturedStdout();
    return output;
}

string test_bfs(string* name,int len, string* edge, int elen, string st)
{
    Graph g;
    testing::internal::CaptureStdout();
    string delimiter = "->";
    for(int i=0;i<len;i++)
    {
        cout << "[GRADER] Adding Vertex:" <<name[i]<< endl;
        g.addVertex(name[i]);
    }
    
    for(int i=0;i<elen;i++)
    {
        int pos = edge[i].find(delimiter);
        string v1 = edge[i].substr(0, pos-1);
        string v2 = edge[i].substr(pos + delimiter.length(), edge[i].length());
        cout<< "[GRADER] Adding edge between: "<<v1<<"   "<<v2<<endl;
        g.addEdge(v1,v2);
    
    }
    g.displayEdges();
    g.breadthFirstTraverse(st);
    string output = testing::internal::GetCapturedStdout();
    return output;
}


string test_findReachableCities(string* name, int len, string* edge, int elen, string src, int k)
{
    Graph g;
    testing::internal::CaptureStdout();
    string delimiter = "->";
    for(int i=0;i<len;i++)
    {
        cout << "[GRADER] Adding Vertex:" <<name[i]<< endl;
        g.addVertex(name[i]);
    }
    
    for(int i=0;i<elen;i++)
    {
        int pos = edge[i].find(delimiter);
        string v1 = edge[i].substr(0, pos-1);
        string v2 = edge[i].substr(pos + delimiter.length(), edge[i].length());
        cout<< "[GRADER] Adding edge between: "<<v1<<"   "<<v2<<endl;
        g.addEdge(v1,v2);
    
    }
    g.displayEdges();
    vector <string> cities = g.findReachableCitiesWithinDistanceK(src, k);

    int numCities = cities.size();
    if (numCities > 0) {
        cout << "[GRADER] The reachable cities are: ";
        for (int i = 0; i < numCities - 1; i++) cout << cities[i] << ",";
        cout << cities[numCities - 1] << "\n";
    } else {
        cout << "[GRADER] No reachable cities.\n";
    }
    string output = testing::internal::GetCapturedStdout();
    return output;
}

    