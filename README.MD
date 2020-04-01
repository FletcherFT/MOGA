# MOGA
A python repository for implementing multi-objective genetic algorithms (MOGAs).

# Problems
## Zermelo's Navigation Problem
[Zermelo's navigation Problem (ZNP)](https://en.wikipedia.org/wiki/Zermelo%27s_navigation_problem).

### Implementation 1
- Branch: znp-imp1
- Description
    - Solutions randomly initialised (start/finish bounds enforced).
    - One solution seeded with straight-line solution.
    - Distance and energy objective fitness.
    - No constraint fitness.
- Observations
    - Pareto Frontier rapidly found.
    - Exploration of PF is poor (tied to xover and mutation functions).
    - Population quickly games the energy fitness function (tied to fitness function and lack of constraints).
- Recommendations
    - Constraints
        1. add a constraint fitness variable.
        2. do ndsa on objective fitness and constraint fitness separately.
        3. modify selection function to prioritise constraint fitness ranking until all solutions are rank 1.
    - Exploration
        1. Mutation function: add new strategies.
        2. Xover function: probably fine.
        
### Implementation 2
- Branch: znp-imp2
- Description
    - Solutions randomly initialised (start/finish bounds enforced).
    - No seeding with straight-line solution.
    - Distance and energy objective fitness
    - No constraints
    
This is a revision of znp-imp1. The cost matrix is a vector field,
and the fitnesses are derived from this vector field.

The fitnesses are:
1. Line integral distance: $$\int_a^bSds$$
2. Speed effort: $$\int_a^b{||V(S)+S'(S)||}ds$$

The functions are:
- Selection Function
    - Based on proportional roulette wheel
- Crossover Function
    1. Two random parents are selected, random segment from second parent inserted into first.
    2. Two random parents are selected, child is the minimum.
    3. Two random parents are selected, child is the maximum.
    4. Minimum two random parents are selected, child is the mean of the parents.
- Mutation Function
    1. Random inversion (random segment of solution is inverted)
    2. Random perturbation (random segment of solution is perturbed by a random vector)
    3. Moving average (solution is smoothed by moving average function of kernel size 3)
- Survival Function
    - Based on NSGA-II (Pareto Rank and Crowding Distance filtering).
    
Observations:
- The selection, crossover (X), and mutation (M) functions are supposed to explore and exploit the search space.
- The selection function applies higher probabilities for better performing solutions (based on non-domination rank).
- Crossovers:
    - X1: The random parents is a bit problematic, perhaps selection should determine the parents for each child.
    - X2/X3/X4: These functions push the solutions to converge to the minimum, maximum or mean respectively, this is fine as long as the population is diverse.
    - X4: Tends to make the solutions converge around the mean, which doesn't promote exploration.
- Mutations:
    - M1: This contributes to exploitation and exploration.
    - M2: This is the primary contributor to exploring the solution space.
    - M3: This results in nice smooth solutions, but it appears to not promote exploration.
- Summary: The exploitation is quite strong due to X2/X3/X4 and M3, but exploration is suffering!