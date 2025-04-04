system_prompt = "You are a helpful assistant. You first think about the reasoning process in the mind and then provides the user with the answer."

instruction_template = """
You are a precise SVG code generator. You will be given an image, and your task is to generate SVG code that reproduces this image as accurately as possible.

SVG Quick Guide
Goal: Transform the provided image into precise SVG code that replicates the image.

Basic SVG Elements:
- <rect> for rectangles and squares
- <circle> for circles
- <ellipse> for ellipses
- <line> for straight lines
- <polyline> for connected lines
- <polygon> for closed shapes
- <path> for complex shapes and curves
- <text> for text elements
- <g> for grouping elements

Process:
1. First analyze the image carefully, identifying distinct visual elements
2. Break down complex shapes into basic SVG elements when possible
3. Identify colors, dimensions, positions, and relationships between elements
4. Generate accurate SVG code that reproduces the image

Rewards:
- Overall visual similarity: +5.0
- Structural accuracy: +3.0
- Color fidelity: +2.0
- Code efficiency (using appropriate elements): +2.0
- Complete reproduction: +10.0

Here's an example:

assume here is an image

<think>
The image contains two main elements:
  1. A blue rectangle/square in the top-left
  2. A red triangle in the top-right
</think>

<answer>
<svg width="100" height="100"> 
  <rect x="10" y="10" width="30" height="30" fill="blue"/> 
  <path d="M60 10 L90 40 L60 40 Z" fill="red"/> 
</svg>
</answer>

Please think step by step and provide the svg code.
Your reponse should be in the format of <think>...</think><answer>...</answer>
"""

init_observation_template = """
[Initial Observation]:
{observation}
please carefully observe the image observation, and generate SVG code.
Your reponse should be in the format of <think>...</think><answer>...</answer>
"""

action_template = """
You have successfully generate a code, and its image looks like:
{observation}
with reward: {reward}
Try to revise your code to make it much more precise and similiar to original image.
Your response should be in the format of <think>...</think><answer>...</answer>

"""