#include <iostream>
#include <fstream>
#include <random>

int main() {
    const int test_sizes[5] = {1000, 10000, 100000, 500000, 1000000};
    for (int i = 0; i < 5; i++) {
      int N = test_sizes[i];
      int N_filename = (i < 4) ? (int)(N / 1000) : 1;
      char *filename = (char*)malloc(24*sizeof(char));
      char unit = (i < 4) ? 'k' : 'M';
      sprintf(filename, "test_file_%d%c.txt", N_filename, unit);
      std::ofstream file(filename);

      int position = 0;

      std::mt19937 rng(42);
      std::uniform_int_distribution<int> energy_dist(1, 1000000);
      std::uniform_int_distribution<int> step_dist(1, 1000000);

      // Write the number of particles as the first line (required by the storm file format)
      file << N << std::endl;

      for (int i = 0; i < N; ++i) {
          position = step_dist(rng);
          int energy = energy_dist(rng);
          file << position << " " << energy << "\n";
      }

      file.close();
   }
}
