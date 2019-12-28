import pickle
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class Lattice(object):
    """Defines a square lattice of linear size L in dimension dim 
    at temperature T (in units of J/kB).  
    """
    def __init__(self, L=None, dim=None, T=None):
        assert all(arg is not None for arg in [L, dim, T])
        self.L = L
        self.dim = dim
        self.T = T
        self.init_lattice()

    def init_lattice(self):
        """Populate lattice with +/- 1 at random.
        """
        self.config = np.random.choice([-1, 1], size=tuple([self.L] * self.dim))
        
    def random_site(self):
        """Returns the position of a randomly-chosen site on the lattice.
        """
        return list(np.random.randint(0, self.L, size=self.dim))
        
    def get_neighbors(self, pos):
        """Returns the nearest neighbors of lattice site `pos`.
        """
        dim = len(pos)
        neighbors = []
        for i in range(dim):
            neighbors.append(pos[:i] + [(pos[i]-1)%self.L] + pos[(i+1):])
            neighbors.append(pos[:i] + [(pos[i]+1)%self.L] + pos[(i+1):])
        return neighbors

    def get_spin(self, pos):
        """Returns the value (-1 or +1) of the spin at lattice site `pos`.
        """
        return self.config[tuple(pos)]

    def flip_spin(self, pos):
        """Flips the spin at lattice site `pos`.
        """
        self.config[tuple(pos)] = -self.config[tuple(pos)]

    def build_cluster(self, pos, cluster, visited):
        """Builds a spin cluster according to the Wolff algorithm.
        """
        spin = self.get_spin(pos)
        neighbors = self.get_neighbors(pos)
        cluster.append(pos)
        visited.append(pos)
        prob = 1 - np.exp(-2 / self.T)
        for n in neighbors:
            if self.get_spin(n) == spin:
                if n not in visited and np.random.random() < prob:
                    cluster = self.build_cluster(n, cluster, visited)
        return cluster
    
    def display(self):
        """Plot the current spin configuration, if `self.dim == 2`.
        """
        if self.dim == 2:
            self.image = plt.imshow(self.config)
            plt.show()
        
class Ising(object):
    """ Simulates the Ising model on a lattice of linear size L
    in dimension dim at temperature T (in units of J/kB).		
    """
    def __init__(self, L=None, dim=None, T=None):
        assert all(arg is not None for arg in [L, dim, T])
        self.lattice = Lattice(L=L, dim=dim, T=T)
        self.L = L
        self.dim = dim
        self.T = T
        self.N = L**dim
        self.rs = np.arange(0, self.L // 2)

    def flip_cluster(self, cluster):
        """Flip all spins in the given cluster.
        """
        for s in cluster:
            self.lattice.flip_spin(s)
    
    def run_wolff(self):
        """Run a single time step of the Wolff algorithm. Namely, build a cluster and then flip it.
        """
        start = self.lattice.random_site()
        cluster = self.lattice.build_cluster(start, [], [])
        self.flip_cluster(cluster)
            
    def simulate(self, Neq=10**4, Nruns=None, progbar=False):
        """Performs `Neq` equilibrium steps and then `Nruns` measurement steps
        of the Wolff algorithm, and calculates observables.
        """
        print('Running {N} eqilibration iterations.'.format(N=Neq))
        for _ in tqdm(range(Neq), disable=(not progbar)):
            self.run_wolff()
        Nruns = Nruns or Neq // 2
        mag_tot = 0
        mag2_tot = 0
        C_tot = 0
        s0sr_tot = np.zeros_like(self.rs)
        print('Done with equilibration.')
        print('Running {} measurement iterations.'.format(Nruns))
        for _ in tqdm(range(Nruns), disable=(not progbar)):
            self.run_wolff()
            config = self.lattice.config
            m = self.get_mag_per_spin(config)
            mag_tot += m
            mag2_tot += m**2
            C_tot += self.get_specific_heat(config)
            s0sr_tot += self.get_spin_spin_corr(config, self.rs)
        self.Neq = Neq
        self.Nruns = Nruns
        self.mag_avg = mag_tot / Nruns
        self.mag2_avg = mag2_tot / Nruns
        self.susc = (mag2_tot / Nruns - (mag_tot / Nruns)**2) / self.T
        self.C = C_tot / Nruns
        self.binderQ = (mag2_tot / Nruns) / (mag_tot / Nruns)**2
        self.s0sr = s0sr_tot / Nruns
    
    def get_mag_per_spin(self, config):
        """Returns the magnetization per spin for spin configuration `config`.
        """
        return abs(config.mean())
    
    def get_specific_heat(self, config):
        """Returns specific heat for spin configuration `config`.
        """
        E, Esq = self.get_energy_per_spin(config)
        return (Esq - E**2) / self.T**2

    def get_energy_per_spin(self, config):
        """Returns the energy per spin and energy square per spin for
        spin configuration `config`.
        """
        E = 0
        Esq = 0
        for pos, _ in np.ndenumerate(config):
            Enn = (-0.5 * sum([config[pos] * 
                    config[tuple(n)] for n in 
                    self.lattice.get_neighbors(list(pos))]))
            E += Enn
            Esq += Enn**2	
        E /= self.N
        Esq /= self.N
        return E, Esq

    def get_spin_spin_corr(self, config, rs):
        """Returns an array containing the product of the spin at
        the origin and the spin r lattice sites away for r in `rs`.
        """
        s0sr = np.zeros(len(rs))
        pos0 = [0] * self.dim
        s0sr[0] = 1
        for i in range(1,len(rs)):
            posr = []
            for j in range(self.dim):
                posr.append(pos0[:j] + [(pos0[j]-rs[i])] + pos0[(j+1):])
                posr.append(pos0[:j] + [(pos0[j]+rs[i])] + pos0[(j+1):])
            s0sr[i] = (sum(config[tuple(pos0)] * config[tuple(p)]
                for p in posr) / (2 * self.dim))
        return s0sr
    
class IsingTempSeries(object):
    """Simulates the Ising model on a square lattice of linear size
    L in dimension dim at `Nts` temperatures from `Tmin` to `Tmax`
    (in units of J/kB), using the Wolff algorithm with `Neq`
    equilibration iterations and `Neq//2` measurement iterations
    for each temperature. Calculates magnetization, susceptibility, specific heat,
    and spin-spin correlation function at each temperature.
    """
    def __init__(self, L=None, dim=None, Tmin=None, Tmax=None, Nts=None, Neq=10**4, Nruns=None):
        assert(arg is not None for arg in [L, dim, Tmin, Tmax, Nts, Nruns])
        self.L = L
        self.dim = dim
        self.Ts = np.linspace(Tmin, Tmax, Nts)
        self.Neq = Neq
        self.Nruns = Nruns or Neq // 2
        self.mag = np.zeros(Nts)
        self.susc = np.zeros(Nts)
        self.C = np.zeros(Nts)
        self.binderQ = np.zeros(Nts)
        self.rs = np.arange(0, self.L // 2)
        self.s0sr = []
           
    def do_series(self, Ts=None, progbar=True, plot=True):
        """Run the Wolff algorithm at a series of temperatures `Ts`.
        """
        if Ts is not None:
            self.Ts = Ts
        for i in tqdm(range(len(self.Ts)), disable=(not progbar)):
            ising = Ising(L=self.L, dim=self.dim, T=self.Ts[i])
            ising.simulate(Neq=self.Neq, Nruns=self.Nruns, progbar=False)
            self.mag[i] = ising.mag_avg
            self.susc[i] = ising.susc
            self.C[i] = ising.C
            self.binderQ[i] = ising.binderQ
            self.s0sr.append(ising.s0sr)
        if plot:
            fig, ax1 = plt.subplots()
            ax1.plot(self.Ts, self.mag, 'bo-')
            ax1.set_ylabel('Magnetization', color='b', fontsize=16)
            ax1.set_xlabel(r'Temperature [$J/k_B$]', fontsize=16)
            ax1.tick_params('y', colors='b')
            ax1.grid(b='off')
        
            ax2 = ax1.twinx()
            ax2.plot(self.Ts, self.susc, 'ro-')
            ax2.set_ylabel(r'Susceptibility [$1/k_B$]', color='r',
                fontsize=16)
            ax2.tick_params('y', colors='r')
            ax2.grid(b='off')
            fig.tight_layout()
            plt.show()
        
            plt.plot(self.Ts, self.C, 'bo-')
            plt.ylabel(r'Specific Heat [$J/k_B^2$]', fontsize=16)
            plt.xlabel(r'Temperature [$J/k_B$]', fontsize=16)
            plt.show()

            plt.plot(self.Ts, self.binderQ, 'bo-')
            plt.ylabel(r'$\langle{m^2}\rangle/\langle|m|\rangle^2$',
                fontsize=16)
            plt.xlabel(r'Temperature [$J/k_B$]', fontsize=16)
            plt.show()

            plt.plot(self.rs[:], self.s0sr[0][:], 'bo-')
            plt.xlabel(r'$r$', fontsize=16)
            plt.ylabel(r'$<s_0s_r>$', fontsize=16)
            plt.title('Spin-spin correlation at T = {}'
                .format(self.Ts[0]))
            plt.show()

    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))