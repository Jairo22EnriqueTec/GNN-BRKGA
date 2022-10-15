/***************************************************************************
                         brkga.cpp  -  description
                             -------------------
    begin                : Mon Jan 20 2020
    copyright            : (C) 2019 by Christian Blum
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

struct Individual {
    vector<double> vec;
    int target_set_size;
};

struct Option {
    int node;
    double value;
};

string inputFile;

// instance data
int n_of_vertices;
int n_of_arcs;
vector< set<int> > neigh;                                   //neighbors vector
vector<int> degree;
vector<int> required;

//BRKGA parameters
int population_size = 30;
double elite_proportion = 0.15; // normally between 0.1 and 0.25
double mutant_proportion = 0.20; // normally between 0.1 and 0.3
double elite_inheritance_probability = 0.7; // normally greater than 0.5 and <= 0.8
double threshold = 0.7;
int seeding = 0;
bool tuning = false;

//general parameters
double computation_time_limit = 100.0;
int trials = 1;

bool option_compare(const Option& o1, const Option& o2) {

    return (o1.value > o2.value);
}

bool individual_compare(const Individual& i1, const Individual& i2) {

    return  i1.target_set_size < i2.target_set_size;
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
        else if (strcmp(argv[iarg],"-p")==0) population_size = atoi(argv[++iarg]);
        else if (strcmp(argv[iarg],"-seed")==0) seeding = atoi(argv[++iarg]);
        else if (strcmp(argv[iarg],"-pe")==0) elite_proportion = atof(argv[++iarg]);
        else if (strcmp(argv[iarg],"-pm")==0) mutant_proportion = atof(argv[++iarg]);
        else if (strcmp(argv[iarg],"-rhoe")==0) elite_inheritance_probability = atof(argv[++iarg]);
        else if (strcmp(argv[iarg],"-t")==0) computation_time_limit = atof(argv[++iarg]);
        else if (strcmp(argv[iarg],"-th")==0) threshold = atof(argv[++iarg]);
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

bool diffusion(set<int>& input, vector<bool>& member, vector<int>& covered_by, int& node) {

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

void evaluate(Individual& ind) {

    vector<Option> options(n_of_vertices);
    for (int i = 0; i < n_of_vertices; ++i) {
        options[i].node = i;
        options[i].value = double(degree[i])*(ind.vec)[i];
    }
    sort(options.begin(), options.end(), option_compare);

    vector<int> covered_by(n_of_vertices, 0);
    bool finished = false;
    set<int> input;
    set<int> target_set;
    vector<bool> member(n_of_vertices, false);
    while (not finished) {
        int pos = first_pos_not_member(options, member);
        target_set.insert(options[pos].node);
        input.insert(options[pos].node);
        member[options[pos].node] = true;
        finished = diffusion(input, member, covered_by, options[pos].node);
    }
    //bool check = diffusion_check(target_set);
    //if (not check) cout << "OHOOHOHOOHOHOHOHOH" << endl;
    ind.target_set_size = int(target_set.size());
}

void generate_random_solution(Individual& ind, std::default_random_engine& generator, std::uniform_real_distribution<double>& distribution) {

    ind.vec = vector<double>(n_of_vertices);
    for (int i = 0; i < n_of_vertices; ++i) (ind.vec)[i] = distribution(generator);
    evaluate(ind);
}

void generate_all_ones_solution(Individual& ind) {

    ind.vec = vector<double>(n_of_vertices);
    for (int i = 0; i < n_of_vertices; ++i) (ind.vec)[i] = 0.5;
    evaluate(ind);
}

/**********
Main function
**********/

int main( int argc, char **argv ) {
    




    read_parameters(argc,argv);

    std::cout << std::setprecision(3) << std::fixed;

    // initializing the random number generator. A random number between 0 and 1 is obtained with: distribution(generator);
    unsigned seed1 = chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed1);
    std::uniform_real_distribution<double> distribution(0,1);

    // reading an instance
    ifstream indata;
    indata.open(inputFile.c_str());
    if(!indata) { // file couldn't be opened
        if (not tuning) cout << "Error: file could not be opened" << endl;
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

    vector<double> target_set_sizes;
    vector<double> times;

    // looping over all trials
    for (int trial = 0; trial < trials; trial++) {

        target_set_sizes.push_back(0.0);
        times.push_back(0.0);

        if (not tuning) cout << "start trial " << trial + 1 << endl;

        // the computation time starts now
        clock_t start = clock();

        int best_target_set_size = n_of_vertices;

        int n_elites = int(double(population_size)*elite_proportion);
        if (n_elites < 1) n_elites = 1;

        int n_mutants = int(double(population_size)*mutant_proportion);
        if (n_mutants < 1) n_mutants = 1;

        int n_offspring = population_size - n_elites - n_mutants;
        if (n_offspring < 1) {
            if (not tuning) cout << "OHOHOH: wrong parameter settings" << endl;
            exit(0);
        }

        double ctime;
        vector<Individual> population(population_size);
        for (int pi = 0; pi < population_size; ++pi) {
            if (pi == 0 and seeding == 1) generate_all_ones_solution(population[pi]);
            else generate_random_solution(population[pi], generator, distribution);
            if (population[pi].target_set_size < best_target_set_size) {
                best_target_set_size = population[pi].target_set_size;
                clock_t current_init = clock();
                ctime = double(current_init - start) / CLOCKS_PER_SEC;
                target_set_sizes[trial] = double(best_target_set_size);
                times[trial] = ctime;
                if (not tuning) cout << "target set size " << best_target_set_size << "\ttime " << ctime << endl;
            }
        }

        clock_t current = clock();
        ctime = double(current - start) / CLOCKS_PER_SEC;
        while (ctime < computation_time_limit) {
            sort(population.begin(), population.end(), individual_compare);
            vector<Individual> new_population(population_size);
            for (int ic = 0; ic < n_elites; ++ic) {
                new_population[ic].vec = population[ic].vec;
                new_population[ic].target_set_size = population[ic].target_set_size;
            }
            for (int ic = 0; ic < n_mutants; ++ic) {
                generate_random_solution(new_population[n_elites + ic], generator, distribution);

                if (new_population[n_elites + ic].target_set_size < best_target_set_size) {
                    best_target_set_size = new_population[n_elites + ic].target_set_size;
                    clock_t current_mut = clock();
                    ctime = double(current_mut - start) / CLOCKS_PER_SEC;
                    target_set_sizes[trial] = double(best_target_set_size);
                    times[trial] = ctime;
                    if (not tuning) cout << "target set size " << best_target_set_size << "\ttime " << ctime << endl;
                }
            }
            for (int ic = 0; ic < n_offspring; ++ic) {
                double rnum1 = distribution(generator);
                int first_parent = produce_random_integer(n_elites, rnum1);
                double rnum2 = distribution(generator);
                int second_parent = n_elites + produce_random_integer(population_size - n_elites, rnum2);
                new_population[n_elites + n_mutants + ic].vec = vector<double>(n_of_vertices);
                for (int i = 0; i < n_of_vertices; ++i) {
                    double rnum = distribution(generator);
                    if (rnum <= elite_inheritance_probability) (new_population[n_elites + n_mutants + ic].vec)[i] = (population[first_parent].vec)[i];
                    else (new_population[n_elites + n_mutants + ic].vec)[i] = (population[second_parent].vec)[i];
                }
                evaluate(new_population[n_elites + n_mutants + ic]);

                if (new_population[n_elites + n_mutants + ic].target_set_size < best_target_set_size) {
                    best_target_set_size = new_population[n_elites + n_mutants + ic].target_set_size;
                    clock_t current_off = clock();
                    ctime = double(current_off - start) / CLOCKS_PER_SEC;
                    target_set_sizes[trial] = double(best_target_set_size);
                    times[trial] = ctime;
                    if (not tuning) cout << "target set size " << best_target_set_size << "\ttime " << ctime << endl;
                }
            }
            population.clear();
            population = new_population;
            clock_t current_end = clock();
            ctime = double(current_end - start) / CLOCKS_PER_SEC;
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

