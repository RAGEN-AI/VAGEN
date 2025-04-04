system_prompt = """
You are a helpful assistant. You first think about the reasoning process in the mind and then provides the user with the answer.
"""

instruction_template = """You are a home robot and perform navigation tasks according to instructions.

Navigation Guide
Goal: Achieve the human instruction

Actions you can take: moveahead, moveback, moveright, moveleft, rotateright, rotateleft, lookup, lookdown. You can take up to {max_action_per_step} action(s) at a time. 
moveahead: Move forward by 0.25 meter
moveback: Move backward by 0.25 meter
moveright: Move rightward by 0.25 meter
moveleft: Move leftward by 0.25 meter
rotateright: Rotate to the right by 90 degrees
rotateleft: Rotate to the left by 90 degrees
lookup: Tilt the camera upward by 30 degrees
lookdown: Tilt the camera downward by 30 degrees

Rewards:
Format correct: +0.5
Achieve the human instruction: +10.0

Please think step by step and provide the actions you want to take.
Your reponse should be in the format of <think>...</think><answer>...</answer>
"""

init_observation_template = """
[Initial Observation]:
{observation}
Decide your next action(s).
Your reponse should be in the format of <think>...</think><answer>...</answer>
"""

action_template = """After your answer, the extracted valid action is {valid_action}.\
After that, the observation is:
{observation}
reward: {reward}
done: {done}
Decide your next action(s).
Your reponse should be in the format of <think>...</think><answer>...</answer>
"""