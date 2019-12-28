# ising-nd
n-dimensional cluster Monte Carlo Ising model simulations

See [report](report.pdf) for details.

Developed for Stanford Physics 212 in the year 2018.

Wolff Algorithm:
------------
  1. Start with a random spin configuration.
  2. Select a lattice site `i` at random. This site will be the “seed” of the cluster.
  3. For each of `i`’s nearest neighbors `j`, if `i` and `j` are parallel and the bond between `i` and `j` has not already been counted, add `j` to the cluster with probability `1 − e−2βJ`.
  4. Place each spin `j` that gets added to the cluster onto the stack. After all of `i`’s neighbors have been considered, repeat step 2 with the site `j` that is on the top of the stack. Repeat until the stack is empty.
  5. Finally, flip all spins in the cluster.
