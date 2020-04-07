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

Recommendations
1. Trial exchange mutation (like X1 without the segment)
2. Modify selection fcn with k-tournament (rank and crowding distance)
3. Keep M1, M2

### Implementation 3
- Branch: znp-imp3
- Description:
    - Implemented recommendations from znp-imp2, except recommendation 3 (all mutations and xovers have been kept).
    - Solutions achieved were as good as znp-imp2.
    - Adding mutation 5 allows the MOGA to find extrema quickly and without the straight-line seed.

Observations
This table shows the parameters set for finding solutions. Adding M5 drastically reduced the number of iterations
 (a strong mutation function). Each run has a results video attached in results folder.
 
|Run|Iterations|Search Size|Front Size|Size|Flow Type|Straight Seed|# Mutations|# XOvers|
|---|----------|-----------|----------|----|---------|-------------|-----------|--------|
|1|30000|50|100|915|u=y, v=0|Yes|4|4|
|2|15000|50|100|915|u=y, v=0|No|4|4|
|3|1000|50|100|915|u=c, v=c|No|5|4|
|4|1000|50|100|915|u=y, v=0|No|5|4|
|5|1000|50|100|915|u=3y, v=0|No|5|4|
|6|3000|50|100|915|u=3y, v=0|No|5|4|
|7|3000|50|100|915|u=3y, v=0|No|5|4|

Recommendations
- The maximum control velocity of the vehicle is still unconstrained. It is directly calculated by the difference
 between the path velocity and the flow velocity vectors for each point. Add constraints to NGSA-II algorithm using 
 chapter 5 of [Multi-Objective Optimization Using Evolutionary Algorithms: An Introduction](https://www.egr.msu.edu/~kdeb/papers/k2011003.pdf).
- The solution-space could be brought into the real-numbers space.
- The solution-space could also include y-coordinates, the elements are then just a time increment.
- Focus on constraints, then y-coordinates, then real-numbers.
- Just a bit of extra programming work on getting zermelo to run from command-line (for easy batch testing).

### Implementation 4
- Branch: znp-imp4
- Description:
    - Implemented maximum speed magnitude inequality constraint.

|Run|Iterations|Search Size|Front Size|Size|Flow Type|Straight Seed|# Mutations|# XOvers|Max Allowed Velocity|
|---|----------|-----------|----------|----|---------|-------------|-----------|--------|--------------------|
|8|1000|50|100|915|u=y, v=0|No|5|4|inf.|
|9|1000|50|100|915|u=y, v=0|No|5|4|10|
|10|1000|50|100|915|u=y, v=0|No|5|4|2|
|11|1000|50|100|915|u=c, v=c|No|5|4|2|
|12|1000|50|100|915|u=c, v=c|No|5|4|10|
|13|1000|50|100|915|u=c, v=c|No|5|4|inf.|

Observations:
    - Results for low resolution problems look good.
    - Constant flow problem solved very fast (50 gens for 915 size search grid).
    - Shear flow problem also solved very fast (50 - 200 gens depending on speed constraint).
    
Recommendations:
    - Investigate flow functions that are stochastic.
    - Flexibility: Allow algorithm to search in x and y (not just x).
    - Scalability: Allow algorithm to specify t for each x(t), y(t) (currently dt is fixed).
    - Scalability 2: Convert search space into Real Numbers for x, y and t.