#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/betweenness_centrality.hpp>
#include <boost/graph/closeness_centrality.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <iostream>
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

typedef boost::adjacency_list<boost::vecS,
                              boost::vecS,
                              boost::undirectedS> Graph;
typedef boost::graph_traits<Graph>::edge_iterator edge_iterator;

typedef boost::property_map<Graph, boost::vertex_index_t>::type VertexIndexMap;
 
int main()
{
    Graph g;
    string pathinstance = "./instances/dimacs/";
    string graphs = "com-youtube.ungraph";
    //com-youtube.ungraph
    string inputFile = pathinstance + graphs+".dimacs";

    // reading an instance
    cout << "\nCargando "+inputFile+" ..." << endl;

    ifstream indata;
    indata.open(inputFile.c_str());
    if(!indata) { // file couldn't be opened
        cout << "Error: file could not be opened" << endl;
        exit(1);
    }


    string s1, s2;
    int n_of_vertices;
    int n_of_arcs;
    indata >> s1 >> s2;
    indata >> n_of_vertices;
    indata >> n_of_arcs;
    
    int u, v;
    while(indata >> s1 >> u >> v) {
        boost::add_edge (u-1, v-1, g);    
    }

    indata.close(); 
    
 
    std::pair<edge_iterator, edge_iterator> ei = edges(g);
 
    std::cout << "Number of edges = " << num_vertices(g) << "\n";
    std::cout << "Edge list:\n";
    
    double n = num_vertices(g);
    /*
    for (edge_iterator it = ei.first; it != ei.second; ++it )
    {
        std::cout << *it << std::endl;
    }
    */
 
    std::cout << std::endl;
    clock_t start = clock();
    
                
    std::vector<double> centrality(boost::num_vertices(g), 0.0);

        {
            VertexIndexMap v_index = get(boost::vertex_index, g);
            boost::iterator_property_map<std::vector<double>::iterator, VertexIndexMap>
                vertex_property_map = make_iterator_property_map(centrality.begin(), v_index);

            boost::brandes_betweenness_centrality(g, vertex_property_map);
        }
    
    clock_t end = clock();
    double elapsed = double(end - start)/CLOCKS_PER_SEC;
    std::cout << elapsed << endl;

    std::cout << "Edge list:\n" <<std::endl;

    namespace ba = boost::accumulators;
    namespace bt = ba::tag;
    ba::accumulator_set<double, ba::features<bt::mean, bt::variance> > acc;

    /*
    for(auto elem : centrality)
    {
        std::cout<<elem * (2 / ((n -1) * (n-2))) << ",";
    }
    std::cout<<std::endl;
    */
    ofstream myfile;
    myfile.open ("bc_youtube.txt");
    for(auto elem : centrality)
    {
        myfile << std::to_string(elem * (2 / ((n -1) * (n-2)))) + "\n";
    }
    /*
    for (int j = 0; j < graphs.size(); ++j) {
        myfile << graphs[j]+","+std::to_string(resultados[j])+
        ","+std::to_string(ElapsedTime[j])+","+std::to_string(graphsize[j])+"\n";
    }
    */
    myfile.close();
 
    return 0;
 }