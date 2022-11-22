from random import randint


def newboard(N):
    board = []
    for i in range(N):
        col = []
        for j in range(N):
            col.append('*')
        board.append(col)
    return board


def placeQueens(board, board_size):
    i = 0
    while i < board_size:
        row = randint(0, board_size - 1)
        if 'Q' not in board[row]:
            board[row][i] = 'Q'
            i += 1
    print(board)


def makehromosome(board, board_size):
    chromosome = []
    for j in range(board_size):
        for i in range(board_size):
            if board[i][j] == 'Q':
                chromosome.append(4-i)
                break
    return chromosome


def initail_population(size):
    population = []
    for i in range(size):
        population.append(newboard(size))
        placeQueens(population[i], size)

    # print(population)
    for i in range(size):
        population[i] = makehromosome(population[i], size)

    print(population)
    return population


def f_f(population):
    size = len(population)
    fitnesses = []
    for i in range(size):
        fitness = 0
        for j in range(size):
            for k in range(size-1):
                if population[i][j] == population[i][k+1]:
                    fitness += 1
                if population[i][j] - population[i][k+1] == i-k+1:
                    fitness += 1
        fitnesses.append(fitness)
    return fitnesses


def parent_sE(fitness, chromosome):  # elitist
    maxs = max(fitness)
    sublist = [x for x in fitness if x < max(fitness)]
    maxs2 = max(sublist)
    print(maxs)
    print(maxs2)

    # return parents


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


if __name__ == "__main__":
    d = initail_population(5)
    x = f_f(d)
    parent_sE(x, d)
    print(x)
