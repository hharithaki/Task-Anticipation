# Thinking Together, Apart: Reasoning and Learning in Ad Hoc Teamwork

Embodied AI agents deployed in assistive roles often have to collaborate with other agents (humans, AI systems) without prior coordination. Methods considered state of the art for such ad hoc teamwork often pursue a data-driven approach that needs a large labeled dataset of prior observations, lacks transparency, and makes it difficult to rapidly revise existing knowledge in response to changes. This repository contains the supplementary files for a hybrid architecture for ad hoc teamwork leveraging the complementary strengths of knowledge-based and data-driven methods for reasoning and learning for ad hoc teamwork. 

Our architecture enables an ad hoc agent to determine its actions through non-monotonic logical reasoning at different abstractions with: 
- a. prior commonsense domain-specific knowledge
- b. models learned and revised rapidly to predict the behavior of other agents
- c. anticipated abstract future goals based on generic knowledge of similar domains in an existing Foundation Model.

We use VirtualHome, a 3D simulation environment to evalute the architecture.

## Video
Example videos of human and ad hoc agents collaborating together to perform household tasks:

Video showing three agents-(two ad hoc agents and one human) collaborating to complete a series of household tasks.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/RRnCDx6D4zc/0.jpg)](https://www.youtube.com/watch?v=RRnCDx6D4zc)


Video showing four agents-(three ad hoc agents and one human) collaborating to complete a series of household tasks.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/l-4D-LtNX8k/0.jpg)](https://www.youtube.com/watch?v=l-4D-LtNX8k)

The action selection of ad hoc agent enhances team collaboration and prevents conflicts, enabling the agents to achieve their goal more efficiently and within less time.
Moreover, the architecture scales efficiently to accommodate a larger number of agents, leading to improved performance.

## Providing Explanations of Decisions
The use of knowledge-based reasoning and simple predictive models provide the ability for the ad hoc agent to generate relational descriptions in response to four types of questions: Causal, Contrastive, Justify beliefs and Counterfactual Questions. The performance of the architecture when creating these explanations can be found here together with execution examples: [Explanations](https://github.com/natsu-dragneel-ig/LLM_AHT/tree/main/Explanations)

## Results on Scalability
Following table shows the average number of steps and time taken by different teams, Team1 (human, ad hoc agent), Team2 (human, two ad hoc agents) and Team3 (human, three ad hoc agents) to complete 100 task routines when collaborating together. The architecture scale to different number of agents efficiently improving performance.

|            Agent Team           |   Steps   | Time(s) |
| :------------------------------ | :-------: | -----:  |
| Team1 (human + 1 ad hoc agent)  |    26.8   |  361.0  |
| Team2 (human + 2 ad hoc agents) |    22.8   |  329.4  |
| Team3 (human + 3 ad hoc agents) |    19.5   |  307.9  |
    
## Folder Structure

```bash
.
├── ASP/ahagent.sp          # Answer Set Prolog implementation of the ad hoc agent after refinment.
├── ASP/ahagent_pre.sp      # Answer Set Prolog implementation of the ad hoc agent.
├── ASP/human.sp            # Answer Set Prolog implementation of the human after refinment.
├── ASP/human_pre.sp        # Answer Set Prolog implementation of the human.
├── Explanations            # Code and results(with examples) for ad hoc agents providing explanations of its behaviour.
├── Models                  # Behaviour models learned by the ad hoc agent for other agents.
├── simulation              # Files from the VirtualHome domain with modification.
├── main.py                 # Main file
└── utils.py                # Utility file.
```
