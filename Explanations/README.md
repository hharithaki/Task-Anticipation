Our architecture enables the ad hoc agent to generate relational descriptions as explanations of its decisions and beliefs and those of the other agents.

## Experiments
For our experiments we designed different configurations in the VirtualHome each with a different arrangement and status of objects in the initial condition, 
**e.g.**, *bread on the kitchen table instead of the counter, microwave open instead of closed*.
We then measured the ability of the agent to come up with answers to 32 different questions (divided between the four types of questions- causal, contrastive, justify beliefs and counterfactuals) by recording the precision and recall values of retrieving the literals to answer these questions.

## Results
| Question type        | Precision | Recall |
| :------------------- | :-------: | -----: |
| Action justification |    1.00   |  1.00  |
| Contrastive          |    0.89   |  0.99  |
| Belief justification |    0.88   |  0.94  |
| Counterfactual       |    1.00   |  0.78  |

High values of precision and recall indicate the ability to automatically extract the correct literals to provide relational descriptions as explanations in response to different types of questions.

## Execution Example

### Execution Example 1.Action Justification, Contrastive, Counterfactual

*Questioner*: Why did you find bread in step 0 ?

*Ad hoc Agent*: Because I had not found the bread yet and I wanted to grab it in step 1.

The agent's response draws attention to the target action's outcome being a requirement for executing a subsequent action. The agent can also be asked why it did not consider picking up a different object.
______________________________________________________________________

*Questioner*: Why did you not find the water glass in step 0?

*Ad hoc Agent*: Because I predicted that the human will find the water glass in step 0.
    
*Questioner*: Why do you think human will grab water glass in step 1?

*Ad hoc Agent*: Because my prediction is that the human wants to bring the glass to the table.

To answer questions about hypothetical situations, the ad hoc agent has to simulate the evolution of state, and the execution of actions by the human and the agent, for a few steps in order to identify and report the motivation for specific action choices
### Execution Example 2. Action Justification, Contrastive, Belief Tracing

*Questioner*: Why did you find cutlets in step 1 ?

*Ad hoc Agent*: Because I have not found the cutlets and I wanted to grab them in time step 2.

*Questioner*: Why did you not grab bread in step 1?

*Ad hoc Agent*: Because the human was holding the bread.

This exchange demonstrates the ability of the ad hoc agent to change its plan in order to prevent a conflict with the human and to justify this decision.
