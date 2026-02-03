# Dissertation

## Notes
- I am happy with a comparison on the use of more exotic / gradient based sampling for the inverse problem.
- This is also a good way to try and get some other experiments into the mix and see how the method fairs in those experiments

## Ideas
- There will be some relationship between the amount of noise that is added and the feasible range of alpha
- Voltammetry with Adsorbed Species poses a difficult problem in high-dimensions would would be good to use
- As well as weakly supported media since that seems to be of application in batteries
- Don't assume that the formal potential is known. We can add that to the simulation

## Update 01/02
I currently have three sampling methods--Metropolis-Hastings MCMC, MCLMC, Pathfinder--done on two voltammetry techniques -- DC and AC voltammetry.

The results on a sinple inverse problem with two parameters are good, but they are very similar. I want to meet with Kathryn to discuss that I need to move into a higher dimensional problem. When I do this I need to prepare the actually PDE and the non-dimensionalisation that is done to get the questions and compare that to what is done in the Bayesian paper.

I think the issue is that the problem is in low dimension and so it is not particularly difficult to solve. But if we add more variables I think it will become more interesting. I think focusing on the problem in the Bayesian inference paper will be useful to understanding what is going on and defining terms and things. Then I can use ideas from Understanding Voltammetry to find some other things to do. Like the 2D problem.


# For using pmap on single CPU

import os
import multiprocessing

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)
