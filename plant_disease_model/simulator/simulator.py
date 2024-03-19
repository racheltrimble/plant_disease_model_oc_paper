# Simple utility class to simulate anything that looks like a Node or a Network.

class Simulator:
    def __init__(self, victim, sim_duration, repeats):
        self.victim = victim
        self.sim_duration = sim_duration
        self.repeats = repeats

    # Run multiple simulations
    def run(self, verbose=False):
        for n in range(self.repeats):
            if verbose:
                print("Repeat ", n)
            self.victim.reset()
            self.victim.progress(self.sim_duration)
            self.victim.stash_results(final_time=self.sim_duration)
        # Final reset to stash the last set of results
        self.victim.reset()

    def run_quantised(self, number_of_timepoints):
        timestep = self.sim_duration/number_of_timepoints
        for n in range(self.repeats):
            self.victim.reset()
            for t in range(number_of_timepoints):
                self.victim.progress(timestep)
            self.victim.stash_results(final_time=self.sim_duration)
        # Final reset to stash the last set of results
        self.victim.reset()


if __name__ == "__main__":
    from node import Node
    from numpy.random import default_rng
    node = Node({'n': 100, 'beta': 5, 'gamma': 1, 'initial_infected': 1, 'idx': 0}, default_rng())
    sim = Simulator(node, 4, 100)
    sim.run()
