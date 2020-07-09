"""
Accepts the sudoku board and returns the solved board
"""


def ConstarintBacktracking(sudoku):
    import constraint
    problem = constraint.Problem()

    # adding and storing variables with range of acceptable values
    # a variable name is the column index 11, 12, 13..., 19, 21,....99
    for i in range(1, 10):
        problem.addVariables(range(i * 10 + 1, i * 10 + 10), range(1, 10))
    
    # ------------ADDING CONSTRAINTS
    # 1. Every row should have different value
    for i in range(1, 10):
        problem.addConstraint(constraint.AllDifferentConstraint(), range(i * 10 + 1, i * 10 + 10))
    
    # 2. Every column should have different value
    for i in range(1, 10):
        problem.addConstraint(constraint.AllDifferentConstraint(), range(10 + i, 100 + i, 10))
    
    # 3. Every sub-block have different value
    for i in [1, 4, 7]:
        for j in [1, 4, 7]:
            square = [10 * i +j, 10 * i + j +1, 10 * i + j +2,
                      10 *( i + 1 ) +j, 10 *( i + 1 ) + j +1, 10 *( i + 1 ) + j +2,
                      10 *( i + 2 ) +j, 10 *( i + 2 ) + j +1, 10 *( i + 2 ) + j +2]
            # ex: 11, 12, 13, 21,22,23, 31,32,33 have to be different
            problem.addConstraint(constraint.AllDifferentConstraint(), square)
    
    # 4. Adding the numbers already in the sudoku
    for i in range(9):
        for j in range(9):
            if sudoku[i][j] != 0:
                # this is a constraint method. There are a lot of ways to add constraints this is one of them
                # it helps us check if the number is equal to the suodoku value
                # SYNTAX: addConstraint(which_constraint, list_of_variable_order)
                def constraint(variable_value, value_in_table = sudoku[i][j]):
                    if variable_value == value_in_table:
                        return True
    
                problem.addConstraint(constraint, [(( i +1 ) *10 + ( j +1))])
    
    # Getting the solutions
    solutions = problem.getSolutions()
    sudoku_solv =[[0 for x in range(9)] for y in range(9)]
    solavble = False

    # if there is no solution to the problem
    if len(solutions) == 0:
        print("No solutions found.")
    else:
        solution = solutions[0]
        solavble = True

        # Storing the solution in 2d array
        for i in range(1, 10):
            for j in range(1, 10):
                sudoku_solv[i - 1][j - 1] = (solution[i * 10 + j])

        # print(sudoku_solv)

    return solavble, sudoku_solv
