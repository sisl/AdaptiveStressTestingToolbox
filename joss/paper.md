---
title: 'POMDPStressTesting.jl: Adaptive Stress Testing for Black-Box Systems'
tags:
  - Julia
  - stress testing
  - black-box systems
  - POMDPs.jl
authors:
  - name: Robert J. Moss
    orcid: 0000-0003-2403-454X
    affiliation: 1
affiliations:
 - name: Stanford University
   index: 1
date: 26 August 2020
bibliography: paper.bib
header-includes: |
    \usepackage{listings}
---
\lstdefinelanguage{Julia}{
    keywords=[3]{initialize!, transition!, evaluate!, distance, isevent, isterminal, environment},
    keywords=[2]{Nothing, Tuple, Real, Bool, Simulation, BlackBox, GrayBox, Sampleable, Environment},
    keywords=[1]{function, abstract, type, end},
    sensitive=true,
    morecomment=[l]{\#},
    morecomment=[n]{\#=}{=\#},
    morestring=[s]{"}{"},
    morestring=[m]{'}{'},
    alsoletter=!?,
    literate={,}{{\color[HTML]{0F6FA3},}}1
             {\{}{{\color[HTML]{0F6FA3}\{}}1
             {\}}{{\color[HTML]{0F6FA3}\}}}1
}

\lstset{
    language         = Julia,
    backgroundcolor  = \color[HTML]{F2F2F2},
    basicstyle       = \small\ttfamily\color[HTML]{19177C},
    numberstyle      = \ttfamily\scriptsize\color[HTML]{7F7F7F},
    keywordstyle     = [1]{\bfseries\color[HTML]{1BA1EA}},
    keywordstyle     = [2]{\color[HTML]{0F6FA3}},
    keywordstyle     = [3]{\color[HTML]{0000FF}},
    stringstyle      = \color[HTML]{F5615C},
    commentstyle     = \color[HTML]{AAAAAA},
    rulecolor        = \color[HTML]{000000},
    frame=lines,
    xleftmargin=10pt,
    framexleftmargin=10pt,
    framextopmargin=4pt,
    framexbottommargin=4pt,
    tabsize=4,
    captionpos=b,
    breaklines=true,
    breakatwhitespace=false,
    showstringspaces=false,
    showspaces=false,
    showtabs=false,
    columns=fullflexible,
    keepspaces=true,
    numbers=none,
}


# Summary

\href{https://github.com/sisl/POMDPStressTesting.jl}{POMDPStressTesting.jl} is a package that uses reinforcement learning and stochastic optimization to find likely failures in black-box systems through a technique called adaptive stress testing [@ast].
Adaptive stress testing (AST) has been used to find failures in safety-critical systems such as aircraft collision avoidance systems [@ast_acasx], flight management systems [@ast_fms], and autonomous vehicles [@ast_av].
The POMDPStressTesting.jl package is written in Julia [@julia] and is part of the wider POMDPs.jl ecosystem [@pomdps_jl], which provides access to simulation tools, policies, visualizations, and---most importantly---solvers.
We provide different solver variants including online planning algorithms such as Monte Carlo tree search [@mcts] and deep reinforcement learning algorithms such as trust region policy optimization (TRPO) [@trpo] and proximal policy optimization (PPO) [@ppo].
Stochastic optimization solvers such as the cross-entropy method [@cem] are also available and random search is provided as a baseline.
Additional solvers can easily be added by adhering to the POMDPs.jl interface.

The AST formulation treats the falsification problem (i.e. finding failures) as a Markov decision process (MDP) with a reward function that uses a measure of distance to a failure event to guide the search towards failure.
The reward function also uses the state transition probabilities to guide towards \textit{likely} failures.
Reinforcement learning aims to maximize the discounted sum of expected rewards, therefore maximizing the sum of log-likelihoods is equivalent to maximizing the likelihood of a trajectory.
A gray-box simulation environment steps the simulation and outputs the state transition probabilities, and the black-box system under test is evaluated in the simulator and outputs an event indication and the real-valued distance metric (i.e. how close we are to failure).
To apply AST to a general black-box system, a user has to implement the following Julia interface:

\begin{lstlisting}[language=Julia]
# GrayBox simulator and environment
abstract type GrayBox.Simulation end
function GrayBox.environment(sim::Simulation)::GrayBox.Environment end
function GrayBox.transition!(sim::Simulation)::Real end

# BlackBox.interface(input::InputType)::OutputType
function BlackBox.initialize!(sim::Simulation)::Nothing end
function BlackBox.evaluate!(sim::Simulation)::Tuple{Real, Real, Bool} end
function BlackBox.distance(sim::Simulation)::Real end
function BlackBox.isevent(sim::Simulation)::Bool end
function BlackBox.isterminal(sim::Simulation)::Bool end
\end{lstlisting}

Our package builds off work originally done in the AdaptiveStressTesting.jl package [@ast], but POMDPStressTesting.jl adheres to the interface defined by POMDPs.jl and provides different action modes and solver types.
Related falsification tools (i.e. tools that do not include most-likely failure analysis) are \textsc{S-TaLiRo} [@staliro], Breach [@breach], and \textsc{FalStar} [@falstar].
These packages use a combination of optimization, path planning, and reinforcement learning techniques to solve the falsification problem.
The tool most closely related to POMDPStressTesting.jl is the AST Toolbox in Python [@ast_av], which wraps around the gym reinforcement learning environment [@gym].
The author has contributed to the AST Toolbox and found the need to create a similar package in pure Julia for better performance and to interface with the POMDPs.jl ecosystem.

# Statement of Need

Validating autonomous systems is a crucial requirement before their deployment into real-world environments.
Searching for likely failures using automated tools enable engineers to address potential problems during development.
Because many autonomous systems are in environments with rare failure events, it is especially important to incorporate likelihood of failure within the search to help inform the potential problem mitigation.
This tool provides a simple interface for general black-box systems to fit into the adaptive stress testing problem formulation and gain access to solvers.
Due to varying simulation environment complexities, random seeds can be used as the AST action when the user does not have direct access to the environmental probability distributions or when the environment is complex.
Alternatively, directly sampling from the distributions allows for finer control over the search.
The interface is designed to easily extend to other autonomous system applications and explicitly separating the simulation environment from the system under test allows for wider validation of complex black-box systems.



# Research and Industrial Usage

POMDPStressTesting.jl has been used to find likely failures in aircraft trajectory prediction systems [@ast_fms], which are flight-critical subsystems used to aid in-flight automation.
A developmental commercial flight management system was stress tested so the system engineers could mitigate potential issues before system deployment [@ast_fms].
In addition to traditional requirements-based testing for avionics certification [@do178c], this work is being used to find potential problems during development.
There is also ongoing research on the use of POMDPStressTesting.jl for assessing the risk of autonomous vehicles and determining failure scenarios of autonomous lunar rovers.


# Acknowledgments

We acknowledge Ritchie Lee for his guidance and original work on adaptive stress testing and the AdaptiveStressTesting.jl package and Mark Koren, Xiaobai Ma, and Anthony Corso for their work on the AST Toolbox Python package and the CrossEntropyMethod.jl package.
We also acknowledge Shreyas Kowshik for his initial implementation of the TRPO and PPO algorithms in Julia.
We want to thank the Stanford Intelligent Systems Laboratory for their development of the POMDPs.jl ecosystem and the MCTS.jl package; particular thanks to Zachary Sunberg.
We also want to thank Mykel J. Kochenderfer for his support and research input and for advancing the Julia community.


# References
