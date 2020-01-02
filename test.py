from aisearch import PSO, DE
import matplotlib.pyplot as plt
import benchmark as bm


def plot_population(aisearch, solutions, fitness):
    # print(solutions.shape, solutions[:, 1])
    ax.scatter(solutions[:, 0], solutions[:, 1], alpha=0.2)
    plt.draw()
    plt.pause(0.01)


if __name__ == '__main__':
    popSize = 1000
    dimensions = 10
    iterations = 1000
    plot_it = False

    aisearch = PSO(bm.schwefel, dimensions)
    aisearch.population_size = popSize
    aisearch.max_iterations = iterations
    if plot_it:
        ax = bm.plot_3d(bm.schwefel, 100)
        aisearch.at_each_evaluation = plot_population
    aisearch.start()

    solution = aisearch.best_solution.copy()
    fitness = bm.schwefel(solution)
    print(solution, fitness)

    if plot_it:
        fig2, ax2 = plt.subplots(2, 1)
        ax2[0].plot(aisearch.best_fitness_hist)
        ax2[1].plot(aisearch.iteration_time_hist)
        plt.show()
