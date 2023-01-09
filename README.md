# Multi-armed bandit (MBA) game

## Epsilon-greedy applied to the stochastic MBA problem

### Stochastic process

Let there be a bandit with `K` non-identical arms, each following a different probability density function (PDF). The graph below shows the PDF of a 7-armed bandit, where each PDF is a normal distribution, truncated within `[0,1]`, with mean `mu[k]` and standard deviation `sigma[k]`.

![distributions.png](../readme_images/distributions.png?raw=true)

### Game description

The true underlying PDF of the bandit arms is unknown a priori. A player is given a total of `T` shots to obtain rewards by interacting with the bandit. At each timestep `t`, the player decides which arm to pull, and subsequently receives a reward sampled from the PDF of the pulled arm. The value of the received reward is the only information provided to the player. Throughout the game, the rewards distribution is neither modified contextually (i.e., by the environment or the player actions), nor adversarially (i.e., by the bandit itself).

The player interest is to maximize the sum of their rewards, by adopting the optimal strategy combining exploration (i.e., trying out each arm to find the best one) and exploitation (i.e., playing the arm believed to give the best payoff).

### Epsilon-greedy strategy

The exploration-exploitation dilemma is addressed as such:

- Let `epsilon` be a real-valued number in `[0,1]` (in this example, `0.5`).
- Let `decay` be a real-valued number in `[0,1]` (in this example, `0.99`).

```
IF: random(0,1) < epsilon
THEN: pull a random arm # Exploration
ELSE: pull the arm with the highest expected quality # Exploitation
epsilon *= decay # Reduce exploration in favor of exploitation
```

![epsilon.png](../readme_images/epsilon.png?raw=true)

### Q-learning algorithm

Let `Q` be a vector of length `K` that represents the estimated expectation of each arm. Quality is updated as such:

```
Initialize Q with zeros
FOR EACH timestep t:
Select one arm (identified by index k) to pull with the epsilon-greedy strategy (see above)
Obtain a reward by sampling the distribution of the k-th arm (see above)
Increase the total rewards corresponding to the k-th arm
Increment the number of times the k-th arm was pulled
Update the quality table s.t. Q[k] = total_rewards_of_arm_k / number_of_pulls_on_arm_k
```

The graph below shows the evolution of the quality table over time. Note that the estimated expectation of each arm converges towards the true expectation of each distribution. 

![quality_per_arm.png](../readme_images/quality_per_arm.png?raw=true)

## Instantaneous rewards

The graph below shows the rewards that were received at each timestep (blue), the true expectation of the arm that was pulled at each timestep (orange), and the maximal expectation among all arms (green). 

- In the early stage of the epislon-greedy strategy, different arms are explored, resulting of sub-optimal rewards expectation.
- At later timesteps, the optimal arm is exploited, resulting in optimal rewards expectation.

![instantaneous_rewards.png](../readme_images/instantaneous_rewards.png?raw=true)

## Regret minimization

The graph below shows the regret, defined as the cumulative sum of the difference between the maximal expectation and the expectation of the pulled arm (green), and the cumulative rewards obtained by the epsilon-greedy strategy (blue) compared against the maximal cumulative expectation (orange) and the result of a fully-random pulling strategy (exploration-only, pink). 

![regret_and_cumulative_rewards.png](../readme_images/regret_and_cumulative_rewards.png?raw=true)

## Sequence of actions

The graph below shows which arm was pulled for each timestep.

- In the early stage of the epislon-greedy strategy, some amount of exploration takes place and all arms are pulled.
- At later timesteps, exploitation prevails and the arm that was found to yield optimal rewards is completely dominant.

![nb_pulls_per_arm.png](../readme_images/nb_pulls_per_arm.png?raw=true)

## Howto

```
julia add_necessary_packages.jl
jupyter notebook
```

## Bibliography

>> Auer P, Cesa-Bianchi N, Freund Y, Schapire RE. Gambling in a rigged casino: The adversarial multi-armed bandit problem. In Proceedings of IEEE 36th annual foundations of computer science, pp. 322-331, 1995.
