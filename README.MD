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
    - One solution seeded with straight-line solution.
    - Distance and energy objective fitness.
    - Max traverse distance as constraint fitness.