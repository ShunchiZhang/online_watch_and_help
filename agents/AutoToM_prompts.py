question = """\
## Task Description

Human wants to set up a table.
Human wants to choose exactly 2 types of objects and exactly 2 of each type, then put them on the same target location.
The possible objects are: ["wineglass", "waterglass", "cutleryfork", "plate"]
The possible target locations are: ["kitchentable", "coffeetable"]

Your are a helpful assistant. In order to help human finish the task, please predict human's overall goals (i.e., 2 types of objects and 2 of each type, and a target location).

## Question

What is Human's overall goal (i.e., 2 types of objects and 2 of each type, and a target location)?\
"""

propose_1 = """\
In the story below, human is searching for an object or objects and plans to place them somewhere. It could be setting up table, putting something in the dishwasher, putting something in the fridge, preparing food, or watching TV.
Please predict human's overall goal (including both completed subgoals and possible future subgoals).
Provide the most possible overall goal hypothesis (in the format that can be directly parsed as JSON) without any explanations.

Story: {story}
Example response format: {{"objects": {{"plate": 1, "wineglass": 1}}, "target": "kitchentable"}}
Proposed overall goal hypothesis:\
"""

propose_n = """\
## Output Format Requirements
Instead of simply giving one hypothesis of human's overall goals, please provide a probability distribution over n={n} overall goal hypotheses.
Your response should be able to directly parsed as JSON, WITHOUT any explanations.
The JSON list should be sorted by the probability in descending order.
An example (n=3): [{{"objects": ["waterglass", "wineglass"], "target": "kitchentable", "p": 0.40}}, {{"objects": ["plate", "wineglass"], "target": "coffeetable", "p": 0.30}}, {{"objects": ["plate", "cutleryfork"], "target": "coffeetable", "p": 0.30}}]

## Specific Problem
Story:
{story}

Please propose a probability distribution over n={n} overall goal hypotheses:\
"""
