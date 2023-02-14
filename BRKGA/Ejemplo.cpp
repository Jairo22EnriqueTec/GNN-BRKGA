#include <iostream>
#include <fstream>
#include <string.h>
using namespace std;

string pi; //path to instances
string ps; //path to save

inline int stoi(string &s) {

  return atoi(s.c_str());
}

inline double stof(string &s) {

  return atof(s.c_str());
}


void read_parameters(int argc, char **argv) {

    int iarg = 1;

    while (iarg < argc) {
        if (strcmp(argv[iarg],"-pi")==0) pi = argv[++iarg];
        else if (strcmp(argv[iarg],"-ps")==0) ps = argv[++iarg];
        iarg++;
    }
}

  
int main(int argc, char** argv)
{
    read_parameters(argc,argv);
    
    string directory = "Models";
    string PATH_TO_SAVE = "../"+directory+"/results/scalefree_MDH_socialnetworks/Pruebas_GA/TwoLayers/Results_SMS.txt";
    ofstream myfile;
    myfile.open (PATH_TO_SAVE);

    //for (int i = 0; i < argc; ++i)
    myfile << "path to save: " << ps << "\n";
    myfile << "path to instances: " << pi << "\n";
    
    myfile.close();
}