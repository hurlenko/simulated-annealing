import numpy as np
import matplotlib.pyplot as plt


class Function(object):
    f = None
    a = None
    b = None

    @staticmethod
    def set_func(func):
        Function.f = func
        Function.a, Function.b = {
            Function.sphere: (-5.12, 5.12),
            Function.rastrigin: (-5.12, 5.12),
            Function.rosenbrock: (-2.048, 2.048),
            Function.ackley: (-32, 32),
            Function.sn: (-1.02, 1.02),
            Function.sn2: (-1.05, 1.05)
        }.get(func, None)
        pass

    @staticmethod
    def sphere(arg_vec):
        return np.sum([x ** 2 for x in arg_vec])

    @staticmethod
    def rosenbrock(arg_vec):
        return sum([(100 * (xj - xi ** 2) ** 2 + (xi - 1) ** 2) for xi, xj in zip(arg_vec[:-1], arg_vec[1:])])

    @staticmethod
    def rastrigin(arg_vec):
        return 10 * len(arg_vec) + np.sum([x ** 2 - 10 * np.cos(2 * np.pi * x) for x in arg_vec])

    @staticmethod
    def ackley(arg_vec):
        s1 = -0.2 * np.sqrt(np.sum([x ** 2 for x in arg_vec]) / len(arg_vec))
        s2 = np.sum([np.cos(2 * np.pi * x) for x in arg_vec]) / len(arg_vec)
        return 20 + np.e - 20 * np.exp(s1) - np.exp(s2)

    @staticmethod
    def sn(arg_vec):
        s1 = (sum(arg_vec) - sum(x*x for x in arg_vec)) * sum(np.cos(x) for x in arg_vec)
        s2 = 4 / (np.sqrt(np.abs(np.tan(sum(arg_vec))))) + int(sum(x*x for x in arg_vec))
        return s1 / s2

    @staticmethod
    def sn2(arg_vec):
        s1 =  arg_vec[0] * np.cos(arg_vec[1])
        s2 = (arg_vec[0] ** 2 - arg_vec[1] + 1) - (1 - arg_vec[1] ** 2)
        return s1 / s2


class Individual(object):
    def __init__(self, phenotypes):
        self.phenotypes = np.array(phenotypes)  # phenotype
        self.fitness = Function.f(self.phenotypes)  # value of the fitness function

    def __str__(self):
        return '{0} = {1}'.format(self.phenotypes, self.fitness)


class SA(object):
    def __init__(self,
                 l_bound,
                 u_bound,
                 dimension,
                 number_cycles,
                 number_trials,
                 prob_start,
                 prob_finish):
        self.l_bound = l_bound
        self.u_bound = u_bound
        self.dimension = dimension
        self.number_cycles = number_cycles
        self.number_trials = number_trials
        self.prob_start = prob_start
        self.prob_finish = prob_finish
        self.temp_start = -1.0 / np.log(prob_start)
        self.temp_finish = -1.0 / np.log(prob_finish)
        self.fraction = (self.temp_finish / self.temp_start) ** (1.0 / (self.number_cycles - 1.0))
        self.best = Individual(self.float_rand(self.l_bound, self.u_bound, self.dimension))
        self.x = [self.best]
        self.delta_avg = 0
        self.number_accepted = 1

    def run(self):
        t = self.temp_start
        for i in range(self.number_cycles):
            print('Cycle: ' + str(i) + ' with Temperature: ' + str(t))
            for j in range(self.number_trials):
                # Generate new trial points
                d = 0.3
                # step = (d - 0.01) / self.number_cycles
                xi = Individual([max(min(a + self.float_rand(-t, t), self.u_bound), self.l_bound)
                                 for a in self.best.phenotypes])
                delta = abs(xi.fitness - self.best.fitness)
                if xi.fitness > self.best.fitness:
                    # Initialize delta_avg if a worse solution was found on the first iteration
                    if i == 0 and j == 0:
                        self.delta_avg = delta
                    # objective function is worse
                    # generate probability of acceptance
                    p = np.exp(-delta / (self.delta_avg * t))
                    # determine whether to accept worse point
                    if np.random.random() < p:
                        # accept the worse solution
                        accept = True
                    else:
                        # don't accept the worse solution
                        accept = False
                else:
                    # objective function is lower, automatically accept
                    accept = True

                if accept is True:
                    # update currently accepted solution
                    self.best = xi
                    # increment number of accepted solutions
                    self.number_accepted += 1.0
                    # update delta_avg
                    self.delta_avg = (self.delta_avg * (self.number_accepted - 1.0) + delta) / self.number_accepted
                    # d -= step
            # Record the best x values at the end of every cycle
            self.x.append(self.best)
            # Lower the temperature for next cycle
            t *= self.fraction

    def plot(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.plot([a.fitness for a in self.x], 'r.-')
        ax1.legend(['Objective'])
        ax2 = fig.add_subplot(212)
        for i in range(self.dimension):
            ax2.plot([a.phenotypes[i] for a in self.x], 'b--')
        ax2.legend(['x' + str(i + 1) for i in range(self.dimension)])

        plt.show()

    def float_rand(self, a, b, size=None):
        return a + ((b - a) * np.random.random(size))


# Number of cycles.
CYCLES = 1000
# Number of trials per cycle
TRIALS = 200
# Probability of accepting worse solution at the start
PROB_START = 0.9
# Probability of accepting worse solution at the end
PROB_FINISH = 1e-20

DIMENSION = 2

Function.set_func(Function.sphere)


def main():
    sa = SA(Function.a, Function.b, DIMENSION, CYCLES, TRIALS, PROB_START, PROB_FINISH)
    sa.run()
    print('Best solution: {0}'.format(sa.best.phenotypes))
    print('Best objective: {0}'.format(sa.best.fitness))
    sa.plot()
if __name__ == '__main__':
    main()