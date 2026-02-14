#include <iostream>
#include <fstream>
#include <random>

int main() {
    std::ofstream file("test_file.txt");

    const int N = 1000000;  // change size here
    int position = 0;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> energy_dist(1, 1000);
    std::uniform_int_distribution<int> step_dist(1, 5);

    // Write the number of particles as the first line (required by the storm file format)
    file << N << std::endl;

    for (int i = 0; i < N; ++i) {
        position += step_dist(rng);
        int energy = energy_dist(rng);
        file << position << " " << energy << "\n";
    }

    file.close();
}