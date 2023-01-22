#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <list>
#include <set>
#include <string>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <limits>
#include <random>
#include <chrono>
#include <fstream>
#include <dirent.h>
#include <time.h>

using namespace std;

struct Individual {
    vector<double> vec;
    set<int> target_set;
    int target_set_size;
};

struct Option {
    int node;
    double value;
};

string inputFile;
/*
vector<string> graphs = {"graph_football",
    "Amazon0302",
    "graph_jazz",
    "graph_karate",
    "gemsec_facebook_artist",
    "ego-facebook",
    "graph_actors_dat",
    "graph_dolphins",
    "graph_Email-Enron",
    "graph_ncstrlwg2",
    "soc-gplus",
    "socfb-Brandeis99",
    "socfb-Mich67",
    "socfb-nips-ego",
    "loc-gowalla_edges",
    "deezer_HR",
    "musae_git"};
    */


vector<string> graphs = {"graph_football",
    "graph_jazz",
    "graph_karate",
    "gemsec_facebook_artist",
    "ego-facebook",
    "graph_actors_dat",
    "graph_CA-AstroPh",
    "graph_CA-CondMat",
    "graph_CA-GrQc",
    "graph_CA-HepPh",
    "graph_CA-HepTh",
    "graph_dolphins",
    "graph_Email-Enron",
    "graph_ncstrlwg2",
    "soc-gplus",
    "socfb-Brandeis99",
    "socfb-Mich67",
    "socfb-nips-ego",
    "Amazon0302",
    "Amazon0312",
    "Amazon0505",
    "Amazon0601",
    "com-dblp.ungraph",
    "loc-gowalla_edges",
    "deezer_HR",
    "musae_git"};

/*

vector<string> graphs = {
 "ER_10000_10_0",
 "ER_10000_10_1",
 "ER_10000_15_0",
 "ER_10000_15_1",
 "ER_10000_20_0",
 "ER_10000_20_1",
 "ER_20000_10_0",
 "ER_20000_10_1",
 "ER_20000_15_0",
 "ER_20000_15_1",
 "ER_20000_20_0",
 "ER_20000_20_1",
 "ER_30000_10_0",
 "ER_30000_15_0",
 "ER_30000_20_0",
 "ER_50000_10_0",
 "ER_50000_15_0",
 "ER_50000_20_0"
 };


*/
// instance data
int n_of_vertices;
int n_of_arcs;
vector< set<int> > neigh;                                   //neighbors vector
vector<int> degree;
vector<int> required;

bool option_compare(const Option& o1, const Option& o2) {

    return (o1.value > o2.value);
}

bool diffusion(set<int>& input, vector<bool>& member, vector<int>& covered_by, int& node) {

    set<int> to_add;
    to_add.insert(node);

    int n_of_vertices_covered = int(input.size());
    while (int(to_add.size()) > 0) {
        set<int> new_to_add;
        // comienzas infectando el nuevo nodo y haces la difusión
        for (set<int>::iterator sit = to_add.begin(); sit != to_add.end(); ++sit) {
            // repasas cada vecino de ese nuevo nodo
            for (set<int>::iterator sit2 = neigh[*sit].begin(); sit2 != neigh[*sit].end(); ++sit2) {
                // si no está infectado
                if (not member[*sit2]) {
                    // se incrementa el número de vecinos infectados que tiene en 1 porque
                    // está conectado a sit
                    ++covered_by[*sit2];
                    // se revisa si cumple el requisito para infectarse
                    if (covered_by[*sit2] >= required[*sit2]) {
                        new_to_add.insert(*sit2);
                        input.insert(*sit2);
                        member[*sit2] = true;
                        ++n_of_vertices_covered;
                    }
                }
            }
        }
        to_add.clear();
        to_add = new_to_add;
    }
    return (n_of_vertices_covered == n_of_vertices);
}

int first_pos_not_member(vector<Option>& options, vector<bool>& member) {

    for (int i = 0; i < n_of_vertices; ++i) {
        if (not member[options[i].node]) return i;
    }
    return -1;
}


void evaluate(Individual& ind) {

    vector<Option> options(n_of_vertices);
    // se multiplica el valor del nodo por el grado del nodo
    // si es MDH, entonces todos los individuos tienen la misma prob
    for (int i = 0; i < n_of_vertices; ++i) {
        options[i].node = i;
        options[i].value = double(degree[i])*(ind.vec)[i];
        //options[i].value = (ind.vec)[i];
    }
    // se ordena la lista para recorrerse del mayor al menor
    sort(options.begin(), options.end(), option_compare);
    
    //
    vector<int> covered_by(n_of_vertices, 0);
    bool finished = false;
    set<int> input;
    (ind.target_set).clear();
    //los nodos infectados
    vector<bool> member(n_of_vertices, false);
    while (not finished) {
        // se extraé el nodo mayor que no esté infectado
        int pos = first_pos_not_member(options, member);
        // este forma parte de la solución final
        (ind.target_set).insert(options[pos].node);
        input.insert(options[pos].node);
        // este nodo se infecta
        member[options[pos].node] = true;
        // se llama la función para ver si se difunde a toda la red
        finished = diffusion(input, member, covered_by, options[pos].node);
    }
    
    //bool check = diffusion_check(target_set);
    //if (not check) cout << "OHOOHOHOOHOHOHOHOH" << endl;
    ind.target_set_size = int((ind.target_set).size());
}

int main() {

    vector<string> models = {
        //"GCN", "GAT", "GraphConv", "SAGE", "SGConv"
        //"SAGE75", "SAGE100"
        "SAGE30"
    };

    string directory = "Models";

    string model = "";
    string PATH_TO_SAVE = "";
    string pathprob = "";
    string pathinstance = "";

    for (int m = 0; m < models.size(); ++m) {
        model = models[m];
            //string PATH_TO_SAVE = "../FastCover/results/scalefree/justprob/FastCoverResults_scalefree.txt";
            PATH_TO_SAVE = "../"+directory+"/results/scalefree_MDH_socialnetworks/Pruebas_GA/TwoLayers/"+model+"Results_SMS.txt";
            pathprob = "../"+directory+"/probabilidades/scalefree_socialnetworks/Pruebas_GA/TwoLayers/"+model;
            pathinstance = "instances/socialnetworks/dimacs/";

            //PATH_TO_SAVE = "../"+directory+"/results/scalefree_MDH_Erdos/Pruebas_GA/MoreTrain/SinBC/"+model+"Results_SME.txt";
            //pathprob = "../"+directory+"/probabilidades/scalefree_Erdos/Pruebas_GA/MoreTrain/SinBC/"+model;
            //pathinstance = "instances/Erdos/test/dimacs/";

            vector<int> resultados (graphs.size());
            vector<int> graphsize (graphs.size());
            vector<double> ElapsedTime (graphs.size());

            for (int j = 0; j < graphs.size(); ++j) {

                //string inputFile = "instances/dimacs/"+graphs[j]+".dimacs";
                string inputFile = pathinstance + graphs[j]+".dimacs";

                // reading an instance
                cout << "\nCargando "+inputFile+" ..." << endl;

                ifstream indata;
                indata.open(inputFile.c_str());
                if(!indata) { // file couldn't be opened
                    cout << "Error: file could not be opened" << endl;
                    exit(1);
                }

                string s1, s2;
                indata >> s1 >> s2;
                indata >> n_of_vertices;
                indata >> n_of_arcs;
                neigh = vector< set<int> >(n_of_vertices);
                int u, v;
                while(indata >> s1 >> u >> v) {
                    neigh[u - 1].insert(v - 1);
                    neigh[v - 1].insert(u - 1);
                }

                indata.close();
                
                degree = vector<int>(n_of_vertices);
                required = vector<int>(n_of_vertices);
                for (int i = 0; i < n_of_vertices; i++) {
                    degree[i] = int(neigh[i].size());
                    required[i] = ceil((double)degree[i] / 2);
                }


                string vecfile = pathprob+"_"+graphs[j]+".txt";
                cout << "\nCargando vector de probabilidades "+vecfile+" ..." << endl;
                indata.open(vecfile.c_str());
                if(!indata) { // file couldn't be opened
                    cout << "Error: file could not be opened 2" << endl;
                    exit(1);
                }
                vector<double> vec = vector<double>(n_of_vertices, 0);
                float i;
                int counter = 0;
                while(indata >> i) {
                    vec[counter] = i;
                    ++counter;
                }

                Individual Prueba;
                Prueba.vec = vec;
                cout << "\nIniciando infección...\n" << endl;
                clock_t start = clock();
                evaluate(Prueba);
                clock_t end = clock();
                double elapsed = double(end - start)/CLOCKS_PER_SEC;

                std::cout << Prueba.target_set_size << endl;

                resultados[j] = Prueba.target_set_size;
                ElapsedTime[j] = elapsed;
                graphsize[j] = n_of_vertices;
            }

            ofstream myfile;
            myfile.open (PATH_TO_SAVE);
            for (int j = 0; j < graphs.size(); ++j) {
                myfile << graphs[j]+","+std::to_string(resultados[j])+
                ","+std::to_string(ElapsedTime[j])+","+std::to_string(graphsize[j])+"\n";
            }
            
            myfile.close();

        
    }
    return 0;

    }

