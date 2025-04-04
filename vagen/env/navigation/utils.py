import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Optional

def draw_target_box(
    image,
    instance_detections: Dict[str, np.ndarray],
    object_id: str,
    output_path: str,
    color: tuple = (0, 255, 0),  # Default color: green
    thickness = 1
):

    if object_id in instance_detections:

        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if image is None:
            raise ValueError(f"Could not read image")
        
        # Get coordinates for the specified object ID
        bbox = instance_detections[object_id]
        start_point = (int(bbox[0]), int(bbox[1]))  # Upper left corner
        end_point = (int(bbox[2]), int(bbox[3]))    # Lower right corner
        
        # Draw the rectangle
        cv2.rectangle(
            image,
            start_point,
            end_point,
            color,
            thickness
        )
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output_image = Image.fromarray(image_rgb)
        output_image.save(output_path)

    else:

        output_image.save(output_path)


valid_objs = ['Cart', 'Potato', 'Faucet', 'Ottoman', 'CoffeeMachine', 'Candle', 'CD', 'Pan', 'Watch',
                'HandTowel', 'SprayBottle', 'BaseballBat', 'CellPhone', 'Kettle', 'Mug', 'StoveBurner', 'Bowl', 'Spoon', 'TissueBox', 'Apple', 'TennisRacket', 'SoapBar',
                'Cloth', 'Plunger', 'FloorLamp', 'ToiletPaperHanger', 'Spatula', 'Plate',
                'Glassbottle', 'Knife', 'Tomato', 'ButterKnife', 'Dresser', 'Microwave',
                'GarbageCan', 'WateringCan', 'Vase', 'ArmChair', 'Safe', 'KeyChain', 'Pot', 'Pen', 'Newspaper', 'Bread', 'Book', 'Lettuce', 'CreditCard', 'AlarmClock',
                'ToiletPaper', 'SideTable', 'Fork', 'Box', 'Egg', 'DeskLamp', 'Ladle', 'WineBottle', 'Pencil',
                'Laptop', 'RemoteControl', 'BasketBall', 'DishSponge', 'Cup', 'SaltShaker', 'PepperShaker',
                'Pillow', 'Bathtub', 'SoapBottle', 'Statue', 'Fridge', 'Toaster', 'LaundryHamper']

def random_color():
    return tuple(np.random.choice(range(256), size=3))

def draw_boxes(image, classes_and_boxes, image_path):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    font.size = 8
    # Loop through each class and its associated bounding boxes
    for class_name, box in classes_and_boxes.items():
        if class_name.split('|')[0] in valid_objs:
            color = random_color()
            # if class_name in name_translation:
            #     name = name_translation[class_name]
            # else:
            #     name = class_name.split('|')[0]
            
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=1)
            # Add class name above the rectangle
            # text_position = (x1, max(0, y1 - 12))  # Position text above box
            # draw.text(text_position, name, fill=color, font=font)
    image.save(image_path)
    # return image

