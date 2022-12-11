/***************************************************************************
                       aco_hybrid.cpp  -  description
                             -------------------
    begin                : Fri Dec 2 2022
    copyright            : (C) 2022 by Christian Blum
    email                : christian.blum@iiia.csic.es
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <list>
#include <set>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <limits>
#include <random>
#include <chrono>

using namespace std;

struct Solution {
    set<int> vertices;
    int target_set_size;
};

struct Option {
    int node;
    double value;
};

string inputFile;
string probabilityFile;

// instance data
int n_of_vertices;
int n_of_arcs;
vector< set<int> > neigh;                                   //neighbors vector
vector<int> degree;
vector<int> required;

// Probability data
vector<double> probabilities;
bool probabilities_provided = false;

//ACO parameters
int n_of_ants = 10;
int candidate_list_size = 10;
double aco_determinism_rate = 0.7;
double tau_max = 0.999;
double tau_min = 0.001;
double learning_rate = 0.1;
bool tuning = false;

//general parameters
double computation_time_limit = 100.0;
bool time_limit_provided = false;
bool candidate_list_size_provided = false;
int trials = 1;

vector<string> graphs = {
"graph1__2.25_1000_5.txt",
"graph1__2.5_1000_5.txt",
"graph1__2.75_1000_5.txt",
"graph1__3_1000_5.txt",
"graph1__2_1000_5.txt"
};
/*
vector<string> graphs = {
"graph1__2.25_1000_5.txt",

// espacio
 "graph1__2.25_1000_10.txt",
 "graph1__2.25_1000_20.txt",
 "graph1__2.25_1000_30.txt",
 
 //"graph1__2.25_1000_5.txt", // ESTE FALTA

 "graph1__2.5_1000_10.txt",
 "graph1__2.5_1000_20.txt",
 "graph1__2.5_1000_30.txt",
 //"graph1__2.5_1000_5.txt", // ESTE FALTA

 "graph1__2.75_1000_10.txt",
 "graph1__2.75_1000_20.txt",
 "graph1__2.75_1000_30.txt",

 // "graph1__2.75_1000_5.txt",// FALTA
 "graph1__2_1000_10.txt",
 "graph1__2_1000_20.txt",
 "graph1__2_1000_30.txt",
 //"graph1__2_1000_5.txt", //FALTA
 "graph1__3_1000_10.txt",
 "graph1__3_1000_20.txt",
 "graph1__3_1000_30.txt",
 //"graph1__3_1000_5.txt" //FALTA
 };
*/
/*
vector<string> graphs = {
 "ER_1000_10_0.dimacs",
 "ER_1000_10_1.dimacs",
 "ER_1000_10_2.dimacs",
 "ER_1000_10_3.dimacs",
 "ER_1000_15_0.dimacs",
 "ER_1000_15_1.dimacs",
 "ER_1000_15_2.dimacs",
 "ER_1000_15_3.dimacs",
 "ER_1000_20_0.dimacs",
 "ER_1000_20_1.dimacs",
 "ER_1000_20_2.dimacs",
 "ER_1000_20_3.dimacs",

 "ER_2000_10_0.dimacs",
 "ER_2000_10_1.dimacs",
 "ER_2000_10_2.dimacs",
 
 "ER_2000_10_3.dimacs",

 "ER_2000_15_0.dimacs",
 "ER_2000_15_1.dimacs",
 "ER_2000_15_2.dimacs",
 "ER_2000_15_3.dimacs",
 "ER_2000_20_0.dimacs",
 "ER_2000_20_1.dimacs",
 "ER_2000_20_2.dimacs",
 "ER_2000_20_3.dimacs",

 "ER_5000_10_0.dimacs",
 "ER_5000_10_1.dimacs",
 "ER_5000_10_2.dimacs",
 "ER_5000_10_3.dimacs",
 "ER_5000_15_0.dimacs",
 "ER_5000_15_1.dimacs",
 "ER_5000_15_2.dimacs",
 "ER_5000_15_3.dimacs",
 "ER_5000_20_0.dimacs",
 "ER_5000_20_1.dimacs",
 "ER_5000_20_2.dimacs",
 "ER_5000_20_3.dimacs"
 };
*/


bool option_compare(const Option& o1, const Option& o2) {

    return o1.value > o2.value;
}

inline int stoi(string &s) {

  return atoi(s.c_str());
}

inline double stof(string &s) {

  return atof(s.c_str());
}

void read_parameters(int argc, char **argv) {

    int iarg = 1;

    while (iarg < argc) {
        if (strcmp(argv[iarg],"-i")==0) inputFile = argv[++iarg];
        else if (strcmp(argv[iarg],"-probfile")==0) {
            probabilityFile = argv[++iarg];
            probabilities_provided = true;
        }
        else if (strcmp(argv[iarg],"-n_ants")==0) n_of_ants = atoi(argv[++iarg]);
        else if (strcmp(argv[iarg],"-l_rate")==0) learning_rate = atof(argv[++iarg]);
        else if (strcmp(argv[iarg],"-d_rate")==0) aco_determinism_rate = atof(argv[++iarg]);
        else if (strcmp(argv[iarg],"-l_size")==0) {
            candidate_list_size = atoi(argv[++iarg]);
            candidate_list_size_provided = true;
        }
        else if (strcmp(argv[iarg],"-t")==0) {
            computation_time_limit = atof(argv[++iarg]);
            time_limit_provided = true;
        }
        else if (strcmp(argv[iarg],"-trials")==0) trials = atoi(argv[++iarg]);
        else if (strcmp(argv[iarg],"-tuning")==0) tuning = true;
        iarg++;
    }
}

int produce_random_integer(int max, double rnum) {

    int num = int(double(max) * rnum);
    if (num == max) num = num - 1;
    return num;
}

int get_random_element(const set<int>& s, double rnum) {

    int r = produce_random_integer(int(s.size()), rnum);
    set<int>::iterator it = s.begin();
    advance(it, r);
    return *it; 
}

bool diffusion(set<int>& input, vector<bool>& member, vector<int>& covered_by, int& node, set<int>& positions, vector<int>& has_position) {

    set<int> to_add;
    to_add.insert(node);

    int n_of_vertices_covered = int(input.size());
    while (int(to_add.size()) > 0) {
        set<int> new_to_add;
        for (set<int>::iterator sit = to_add.begin(); sit != to_add.end(); ++sit) {
            for (set<int>::iterator sit2 = neigh[*sit].begin(); sit2 != neigh[*sit].end(); ++sit2) {
                if (not member[*sit2]) {
                    ++covered_by[*sit2];
                    if (covered_by[*sit2] >= required[*sit2]) {
                        new_to_add.insert(*sit2);
                        input.insert(*sit2);
                        member[*sit2] = true;
                        positions.erase(has_position[*sit2]);
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

bool diffusion_check(set<int> to_add) {


    vector<int> covered_by(n_of_vertices, 0);
    vector<bool> done(n_of_vertices, false);
    for (set<int>::iterator sit = to_add.begin(); sit != to_add.end(); ++sit) {
        done[*sit] = true;
    }

    int n_of_vertices_covered = int(to_add.size());
    while (int(to_add.size()) > 0) {
        set<int> new_to_add;
        for (set<int>::iterator sit = to_add.begin(); sit != to_add.end(); ++sit) {
            for (set<int>::iterator sit2 = neigh[*sit].begin(); sit2 != neigh[*sit].end(); ++sit2) {
                if (not done[*sit2]) {
                    ++covered_by[*sit2];
                    if (covered_by[*sit2] >= required[*sit2]) {
                        new_to_add.insert(*sit2);
                        done[*sit2] = true;
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

void generate_aco_solution(Solution& iSol, vector<double>& pheromone, std::default_random_engine& generator, std::uniform_real_distribution<double>& distribution, vector<bool>& isolated_vertex) {

    set<int> input;
    iSol.target_set_size = 0;

    vector<Option> options;
    for (int i = 0; i < n_of_vertices; ++i) {
        if (not isolated_vertex[i]) {
            Option o;
            o.node = i;
            if (probabilities_provided) o.value = double(degree[i] + 1)*pheromone[i]*probabilities[i];
            else o.value = double(degree[i] + 1)*pheromone[i];
            options.push_back(o);
        }
        else {
            (iSol.vertices).insert(i);
            iSol.target_set_size += 1;
            input.insert(i);
        }
    }
    sort(options.begin(), options.end(), option_compare);
    vector<int> has_position(n_of_vertices);
    for (int i = 0; i < n_of_vertices; ++i) {
        if (not isolated_vertex[i]) has_position[options[i].node] = i;
    }

    int n_of_candidates = candidate_list_size;
    if (n_of_candidates > int(options.size())) n_of_candidates = int(options.size());
    set<int> positions;
    for (int i = 0; i < n_of_candidates; ++i) positions.insert(i);

    vector<int> covered_by(n_of_vertices, 0);
    bool finished = false;
    vector<bool> member = isolated_vertex;
    while (not finished) {
        int pos = 0;
        double dec = distribution(generator);
        if (dec > aco_determinism_rate) {
            vector<double> probs;
            for (set<int>::iterator sit = positions.begin(); sit != positions.end(); ++sit) probs.push_back(options[*sit].value);
            std::discrete_distribution<> distr(probs.begin(), probs.end());
            pos = distr(generator);
        }
        set<int>::iterator sit = positions.begin();
        advance(sit, pos);
        int chosen_node = options[*sit].node;
        positions.erase(*sit);
        (iSol.vertices).insert(chosen_node);
        iSol.target_set_size += 1;
        input.insert(chosen_node);
        member[chosen_node] = true;
        finished = diffusion(input, member, covered_by, chosen_node, positions, has_position);
        if (not finished) {
            int cpos;
            if (int(positions.size()) == 0) {
                 cpos = 0;
                 while (member[options[cpos].node]) cpos += 1;
                 cpos += 1;
            }
            else {
                cpos = *(--(positions.end()));
                cpos += 1;
            }
            while (cpos < n_of_vertices and int(positions.size()) < n_of_candidates) {
                if (not member[options[cpos].node]) positions.insert(cpos);
                cpos += 1;                
            }
        }
    }
    //bool check = diffusion_check(iSol.vertices);
    //if (not check) cout << "OHOOHOHOOHOHOHOHOH" << endl;
}

double compute_convergence_factor(vector<double>& pheromone) {

    double ret_val = 0.0;
    for (int i = 0; i < n_of_vertices; ++i) {
        if ((tau_max - pheromone[i]) > (pheromone[i] - tau_min)) ret_val += tau_max - pheromone[i];
        else ret_val += pheromone[i] - tau_min;
    }
    ret_val = ret_val / (double(n_of_vertices) * (tau_max - tau_min));
    ret_val = (ret_val - 0.5) * 2.0;
    return ret_val;
}

/**********
Main function
**********/

int main( int argc, char **argv ) {
    
    read_parameters(argc,argv);

    string path_to_instances = "../BRKGA/instances/scalefree/train/dimacs/";
    string path_to_save = "../BRKGA/instances/scalefree/train/optimal_aco/";
    
    for (int j = 0; j < graphs.size(); ++j) {

        std::cout << std::setprecision(2) << std::fixed;

        // initializing the random number generator. A random number between 0 and 1 is obtained with: distribution(generator);
        unsigned seed1 = chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed1);
        std::uniform_real_distribution<double> distribution(0,1);

        // reading an instance
        ifstream indata;
        inputFile = path_to_instances + graphs[j];

        indata.open(inputFile.c_str());
        if(!indata) { // file couldn't be opened
            if (not tuning) cout << "Error: file " +inputFile+ " could not be opened" << endl;
            exit(1);
        }
        cout << "Working in " +inputFile << endl;

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
        
        vector<bool> isolated_vertex(n_of_vertices, false);
        
        degree = vector<int>(n_of_vertices);
        required = vector<int>(n_of_vertices);
        for (int i = 0; i < n_of_vertices; i++) {
            degree[i] = int(neigh[i].size());
            if (degree[i] == 0) isolated_vertex[i] = true;
            required[i] = ceil((double)degree[i] / 2);
        }

        // automatially limit the computation time if no time limit is provided
        if (not time_limit_provided) {
            computation_time_limit = double(n_of_vertices)/100.0;
            if (computation_time_limit < 100.0) computation_time_limit = 100.0;
            if (computation_time_limit > 1800.0) computation_time_limit = 1800.0;
        }

        // automatically set the canddiate list size if not provided
        if (not candidate_list_size_provided) {
            candidate_list_size = int(double(n_of_vertices)/100.0);
            if (candidate_list_size < 10) candidate_list_size = 10;
            if (candidate_list_size > 1000) candidate_list_size = 1000;
        }

        // reading GAT data
        if (probabilities_provided) {
            ifstream probdata;
            probdata.open(probabilityFile.c_str());
            if(!probdata) { // file couldn't be opened
                cout << "Error: GAT data file could not be opened" << endl;
                exit(1);
            }
            double prob;
            while (probdata >> prob) {
                probabilities.push_back(prob);
            }
            probdata.close();
        }

        vector<double> target_set_sizes;
        vector<double> times;

        // looping over all trials
        for (int trial = 0; trial < trials; trial++) {

            target_set_sizes.push_back(0.0);
            times.push_back(0.0);

            if (not tuning) cout << "start trial " << trial + 1 << endl;

            // the computation time starts now
            clock_t start = clock();

            // initialization of the pheromone values
            vector<double> pheromone(n_of_vertices, 0.5);
            
            Solution bestSol;
            bestSol.target_set_size = std::numeric_limits<int>::max();

            Solution restartSol;
            restartSol.target_set_size = std::numeric_limits<int>::max();

            bool global_convergence = false;
            double cf = 0.0;

            clock_t current = clock();
            double ctime = double(current - start) / CLOCKS_PER_SEC;
            bool stop = false;
            while (not stop and ctime < computation_time_limit) {

                // generate n_of_ants solutions
                double iteration_average = 0.0;
                Solution iBestSol;
                iBestSol.target_set_size = std::numeric_limits<int>::max();
                for (int na = 0; not stop and na < n_of_ants; ++na) {
                    Solution iSol;
                    generate_aco_solution(iSol, pheromone, generator, distribution, isolated_vertex);
                    clock_t current2 = clock();
                    ctime = double(current2 - start) / CLOCKS_PER_SEC;
                    iteration_average += double(iSol.target_set_size);
                    if (iSol.target_set_size < iBestSol.target_set_size) iBestSol = iSol;
                    if (iSol.target_set_size < restartSol.target_set_size) restartSol = iSol;
                    if (iSol.target_set_size < bestSol.target_set_size) {
                        bestSol = iSol;
                        target_set_sizes[trial] = double(bestSol.target_set_size);
                        times[trial] = ctime;
                        if (not tuning) cout << "best " << bestSol.target_set_size << "\ttime " << times[trial] << endl;
                    }
                    if (ctime >= computation_time_limit) stop = true;
                }
                if (not stop) {
                    iteration_average /= double(n_of_ants);

                    double i_weight, r_weight, g_weight;
                    if (global_convergence) {
                        i_weight = 0.0;
                        r_weight = 0.0;
                        g_weight = 1.0;
                    }
                    else {
                        if (cf < 0.2) {
                            i_weight = 1.0;
                            r_weight = 0.0;
                            g_weight = 0.0;
                        }
                        else if (cf < 0.5) {
                            i_weight = 2.0/3.0;
                            r_weight = 1.0/3.0;
                            g_weight = 0.0;
                        }
                        else if (cf < 0.8) {
                            i_weight = 1.0/3.0;
                            r_weight = 2.0/3.0;
                            g_weight = 0.0;
                        }
                        else {
                            i_weight = 0.0;
                            r_weight = 1.0;
                            g_weight = 0.0;
                        }
                    }

                    vector<double> contribution(n_of_vertices, 0.0);
                    for (set<int>::iterator sit = (iBestSol.vertices).begin(); sit != (iBestSol.vertices).end(); ++sit) contribution[*sit] += i_weight;
                    for (set<int>::iterator sit = (restartSol.vertices).begin(); sit != (restartSol.vertices).end(); ++sit) contribution[*sit] += r_weight;
                    for (set<int>::iterator sit = (bestSol.vertices).begin(); sit != (bestSol.vertices).end(); ++sit) contribution[*sit] += g_weight;

                    for (int i = 0; i < n_of_vertices; ++i) {
                        pheromone[i] += (learning_rate * (contribution[i] - pheromone[i]));
                        if (pheromone[i] > tau_max) pheromone[i] = tau_max;
                        if (pheromone[i] < tau_min) pheromone[i] = tau_min;
                    }

                    cf = compute_convergence_factor(pheromone);
                    if (cf > 0.99) {
                        cf = 0;
                        if (global_convergence) {
                            global_convergence = false;
                            pheromone = vector<double>(n_of_vertices, 0.5);
                            restartSol.target_set_size = std::numeric_limits<int>::max();
                        }
                        else global_convergence = true;
                    }
                
                    clock_t current_end = clock();
                    ctime = double(current_end - start) / CLOCKS_PER_SEC;
                }
            }   
            if (not tuning) {
                cout << "Solución óptima:\n" <<endl;

                for (set<int>::iterator sit = (bestSol.vertices).begin(); sit != (bestSol.vertices).end(); ++sit) cout << " " << *sit;
                cout << endl;
                //cout << "aqui"<< endl;
                
                std::ofstream outFile(path_to_save + graphs[j]);
                for (const auto &e : bestSol.vertices) outFile << e << "\n";

            }
            if (not tuning) cout << "end trial " << trial + 1 << endl;
        }

        int best_result = std::numeric_limits<int>::max();
        double r_mean = 0.0;
        double g_mean = 0.0;
        for (int i = 0; i < target_set_sizes.size(); i++) {
            r_mean = r_mean + target_set_sizes[i];
            g_mean = g_mean + times[i];
            if (int(target_set_sizes[i]) < best_result) best_result = int(target_set_sizes[i]);
        }
        r_mean = r_mean / ((double)target_set_sizes.size());
        g_mean = g_mean / ((double)times.size());
        double rsd = 0.0;
        double gsd = 0.0;
        for (int i = 0; i < target_set_sizes.size(); i++) {
            rsd = rsd + pow(target_set_sizes[i]-r_mean,2.0);
            gsd = gsd + pow(times[i]-g_mean,2.0);
        }
        rsd = rsd / ((double)(target_set_sizes.size()-1.0));
        if (rsd > 0.0) {
            rsd = sqrt(rsd);
        }
        gsd = gsd / ((double)(times.size()-1.0));
        if (gsd > 0.0) {
            gsd = sqrt(gsd);
        }
        if (not tuning) cout << best_result << "\t" << r_mean << "\t" << rsd << "\t" << g_mean << "\t" << gsd<< endl;
        else cout << target_set_sizes[0] << endl;
    }
}

