from random import randint


def newboard(N):
    board = []
    for i in range(N):
        chromosome = []
        for j in range(N):
            chromosome.append('*')
        board.append(chromosome)
    return board


def placeQueens(board, board_size):
    queen = 0
    for i in range(board_size):
        queen = randint(0, 4)
        board[queen] = 'Q'
    return board


# def makehromosome():
#

# def initail_population(size):
#     population = []
#     for i in range(size):
#         population.append(newboard)
#

# def f_f(population):
#     di_pop = {}
#     di_pop.ke = population[i]
#     di_pop.value = f.s
#

# def parent_sE(di_pop):  # elitist
#     di.sort
#     parents = []
#     top 10% = 120
#     for i in range(120):
#         parents.append(di.ke)
#     return parents
#

# def ross_over(parents):
#     gen = []
#     point = random(0, 7)  # out of index
#
#     for i in range(parents.size):
#         first
#         A = parent[i][0:point]
#         B = parent[i + 1][point:]
#         final1 = A + B
#         Se
#         A = parent[i][point:]
#         B = parent[i + 1][0:point]
#         final2 = A + B
#
#     gen.append(final1)
#     gen.append(final2)
#

# def mutation(gen):
#     gen.size
#     lessNum = less
#     number
#     of
#     hromosomes - mutate
#
#     for i in lessNum:
#         random
#         position - index   ->0
#         random
#         value - range(0 - 7) ->3
#
#         [4, 5, 3, 1, 3, 4]
#         [3, 5, 3, 1, 3, 4]
#

# def GA(max_gen, population):
#
#     i = 0
#     while (i < max_gen):
#         f.f = pop
#         gen = rossover
#         mut_gen = mutation
#
#         for i in range(len(mut_gen)):
#             mut_gen[i].f.s == max or
#             mut_gen[i].f.s == 0
#             return
#         population = mut_gen
#         i = i + 1
