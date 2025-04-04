import os, math, re
import textwrap

import numpy as np
from scipy import spatial
from PIL import Image, ImageDraw, ImageFont
import logging

from vagen.env.eb_alfred.envdir.thor_env import ThorEnv
from vagen.env.eb_alfred.gen import constants
from vagen.env.eb_alfred.gen.utils.game_util import get_objects_with_name_and_prop
from vagen.env.eb_alfred.utils import natural_word_to_ithor_name


log = logging.getLogger(__name__)

log.setLevel(level=logging.ERROR)

class ThorConnector(ThorEnv):
    def __init__(self, x_display=constants.X_DISPLAY,
                 player_screen_height=constants.DETECTION_SCREEN_HEIGHT,
                 player_screen_width=constants.DETECTION_SCREEN_WIDTH,
                 quality='MediumCloseFitShadows',
                 build_path=constants.BUILD_PATH):
        super().__init__(x_display, player_screen_height, player_screen_width, quality, build_path)
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf", 24)
        self.agent_height = 0.9
        self.cur_receptacle = None
        self.reachable_positions, self.reachable_position_kdtree = None, None
        self.sliced = False
        self.task = None
        self.put_count_dict = {}

    def restore_scene(self, object_poses, object_toggles, dirty_and_empty):
        # print(object_poses)
        super().restore_scene(object_poses, object_toggles, dirty_and_empty)
        self.reachable_positions, self.reachable_position_kdtree = self.get_reachable_positions()
        self.cur_receptacle = None

    def get_reachable_positions(self):
        free_positions = super().step(dict(action="GetReachablePositions")).metadata["actionReturn"]
        free_positions = np.array([[p['x'], p['y'], p['z']] for p in free_positions])
        kd_tree = spatial.KDTree(free_positions)
        return free_positions, kd_tree

    def write_step_on_img(self, cfg, idx, description):
        img = Image.fromarray(self.last_event.frame)
        text = str(idx) + ':' + description['action']
        lines = textwrap.wrap(text, width=20)
        y_text = 6
        draw = ImageDraw.Draw(img)
        for line in lines:
            width, height = self.font.getsize(line)
            draw.text((6, y_text), line, font=self.font, fill=(255, 255, 255))
            y_text += height
        if cfg is True:
            if not description['success']:
                text_msg = 'error : ' + description['message']
                lines = textwrap.wrap(text_msg, width=20)
                for line in lines:
                    width, height = self.font.getsize(line)
                    draw.text((6, y_text + 6), line, font=self.font, fill=(255, 0, 0))
                    y_text += height
        return img


    def find_close_reachable_position(self, loc, nth=1):
        d, i = self.reachable_position_kdtree.query(loc, k=nth + 1)
        selected = i[nth - 1]
        return self.reachable_positions[selected]

    def llm_skill_interact(self, instruction: str):
        if instruction.startswith("put down ") or instruction.startswith("open "):
            pass
        else:
            self.cur_receptacle = None

        if instruction.startswith("find "):
            obj_name = instruction.replace('find a ', '').replace('find an ', '')
            self.cur_receptacle = obj_name
            is_recep_id = any(i.isdigit() for i in obj_name)
            ret = self.nav_obj(natural_word_to_ithor_name(obj_name), self.sliced)
        elif instruction.startswith("pick up "):
            obj_name = instruction.replace('pick up the ', '')
            is_recep_id = any(i.isdigit() for i in obj_name)
            ret = self.pick(natural_word_to_ithor_name(obj_name))
        elif instruction.startswith("put down "):
            # m = re.match(r'put down (.+) on (.+)', instruction)
            # obj = m.group(1).replace('the ', '')
            # receptacle = m.group(2).replace('the ', '')
            if self.cur_receptacle is None:
                ret = self.drop()
            else:
                m = re.match(r'put down (.+)', instruction)
                obj = m.group(1).replace('the ', '')

                if self.cur_receptacle  in self.put_count_dict:
                    self.put_count_dict[self.cur_receptacle ] += 1
                else:
                    self.put_count_dict[self.cur_receptacle ] = 1

                receptacle = self.cur_receptacle
                ret = self.put(natural_word_to_ithor_name(receptacle))

                if len(ret) > 16 and self.put_count_dict[receptacle] >= 3:
                    # if put down failed, then drop the object
                    self.drop()
                    ret += f'. The robot dropped the object instead.'
                    self.last_event.metadata['lastActionSuccess'] = False

        elif instruction.startswith("open "):
            obj_name = instruction.replace('open the ', '')
            ret = self.open(natural_word_to_ithor_name(obj_name))
        elif instruction.startswith("close "):
            obj_name = instruction.replace('close the ', '')
            ret = self.close(natural_word_to_ithor_name(obj_name))
        elif instruction.startswith("turn on "):
            obj_name = instruction.replace('turn on the ', '')
            ret = self.toggleon(natural_word_to_ithor_name(obj_name))
        elif instruction.startswith("turn off "):
            obj_name = instruction.replace('turn off the ', '')
            ret = self.toggleoff(natural_word_to_ithor_name(obj_name))
        elif instruction.startswith("slice "):
            obj_name = instruction.replace('slice the ', '')
            ret = self.slice(natural_word_to_ithor_name(obj_name))
            self.sliced = True
        elif instruction.startswith("drop"):
            ret = self.drop()
        else:
            assert False, 'instruction not supported'

        if not self.last_event.metadata['lastActionSuccess']:
            log.warning(f"llm_skill_interact failed")
            log.warning(f"errorMessage: {self.last_event.metadata['errorMessage']}")
            log.warning(f"returned msg: {ret}")
        else:
            log.info(f"Last action succeeded")

        ret_dict = {
            'action': instruction,
            'success': len(ret) <= 0,
            'message': ret
        }

        return ret_dict

    def get_object_prop(self, name, prop, metadata):
        for obj in metadata['objects']:
            if name in obj['objectId']:
                return obj[prop]
        return None

    @staticmethod
    def angle_diff(x, y):
        x = math.radians(x)
        y = math.radians(y)
        return math.degrees(math.atan2(math.sin(x - y), math.cos(x - y)))
    
    def nav_obj(self, target_obj: str, prefer_sliced=False):
        objects = self.last_event.metadata['objects']
        action_name = 'object navigation'
        ret_msg = ''
        print(f'{action_name} ({target_obj})')

        # get the object location
        if '|' in target_obj:
            obj_id = target_obj
            target_obj = target_obj.split('|')[0]
            tmp_id, tmp_obj_data = self.get_obj_id_from_name(target_obj, priority_in_visibility=True, priority_sliced=prefer_sliced)
            # if sliced object 
            if 'Sliced' in tmp_id and obj_id in tmp_id:
                obj_id = tmp_id
                obj_data = tmp_obj_data
        else:
            obj_id, obj_data = self.get_obj_id_from_name(target_obj, priority_in_visibility=True, priority_sliced=prefer_sliced)

        # find object index from id
        obj_idx = -1
        for i, o in enumerate(objects):
            if o['objectId'] == obj_id:
                obj_idx = i
                break
        if obj_idx == -1:
            ret_msg = f'Cannot find {target_obj}. This object may not exist in this scene. Try to explore other instances instead.'
        else:
            # teleport sometimes fails even with reachable positions. if fails, repeat with the next closest reachable positions.
            max_attempts = 20
            teleport_success = False

            # get obj location
            loc = objects[obj_idx]['position']
            obj_rot = objects[obj_idx]['rotation']['y']

            # # do not move if the object is already visible and close
            # if objects[obj_idx]['visible'] and objects[obj_idx]['distance'] < 1.0:
            #     log.info('Object is already visible')
            #     max_attempts = 0
            #     teleport_success = True

            # try teleporting
            reachable_pos_idx = 0
            for i in range(max_attempts):
                reachable_pos_idx += 1
                if i == 10 and (target_obj == 'Fridge' or target_obj == 'Microwave'):
                    reachable_pos_idx -= 10

                closest_loc = self.find_close_reachable_position([loc['x'], loc['y'], loc['z']], reachable_pos_idx)
                # calculate desired rotation angle (see https://github.com/allenai/ai2thor/issues/806)
                rot_angle = math.atan2(-(loc['x'] - closest_loc[0]), loc['z'] - closest_loc[2])
                if rot_angle > 0:
                    rot_angle -= 2 * math.pi
                rot_angle = -(180 / math.pi) * rot_angle  # in degrees

                if i < 10 and (target_obj == 'Fridge' or target_obj == 'Microwave'):  # not always correct, but better than nothing
                    angle_diff = abs(self.angle_diff(rot_angle, obj_rot))
                    if target_obj == 'Fridge' and \
                            not ((90 - 20 < angle_diff < 90 + 20) or (270 - 20 < angle_diff < 270 + 20)):
                        continue
                    if target_obj == 'Microwave' and \
                            not ((180 - 20 < angle_diff < 180 + 20) or (0 - 20 < angle_diff < 0 + 20)):
                        continue

                # calculate desired horizon angle
                camera_height = self.agent_height + constants.CAMERA_HEIGHT_OFFSET
                xz_dist = math.hypot(loc['x'] - closest_loc[0], loc['z'] - closest_loc[2])
                hor_angle = math.atan2((loc['y'] - camera_height), xz_dist)
                hor_angle = (180 / math.pi) * hor_angle  # in degrees
                hor_angle *= 0.9  # adjust angle for better view
                # hor_angle = -30
                # hor_angle = 0

                # teleport ### Full
                super().step(dict(action="TeleportFull", x=closest_loc[0], y=self.agent_height, z=closest_loc[2], rotation=rot_angle, horizon=-hor_angle))

                if not self.last_event.metadata['lastActionSuccess']:
                    log.warning(
                        f"TeleportFull action failed: {self.last_event.metadata['errorMessage']}, trying again...")
                else:
                    teleport_success = True
                    break

            if not teleport_success:
                ret_msg = f'Cannot move to {target_obj}'

        return ret_msg

    def get_obj_id_from_name(self, obj_name, only_pickupable=False, only_toggleable=False, priority_sliced=False, get_inherited=False,
                             parent_receptacle_penalty=True, priority_in_visibility=False, exclude_obj_id=None):
        obj_id = None
        obj_data = None
        min_distance = 1e+8

        if any(i.isdigit() for i in obj_name):
            for i in range(len(self.last_event.metadata['objects'])):
                if obj_name in self.last_event.metadata['objects'][i]['name']:
                    obj_id = self.last_event.metadata['objects'][i]['objectId']
                    obj_data = self.last_event.metadata['objects'][i]
                    break
            return obj_id, obj_data
        for obj in self.last_event.metadata['objects']:
            if obj['objectId'] == exclude_obj_id:
                continue
            
            if (only_pickupable is False or obj['pickupable']) and \
                    (only_toggleable is False or obj['toggleable']) and \
                    obj['objectId'].split('|')[0].casefold() == obj_name.casefold() and \
                    (get_inherited is False or len(obj['objectId'].split('|')) == 5):
                
                if obj["distance"] < min_distance:
                    penalty_advantage = 0  # low priority for objects in closable receptacles such as fridge, microwave
                    if parent_receptacle_penalty and obj['parentReceptacles']:
                        for p in obj['parentReceptacles']:
                            is_open = self.get_object_prop(p, 'isOpen', self.last_event.metadata)
                            openable = self.get_object_prop(p, 'openable', self.last_event.metadata)
                            if openable is True and is_open is False:
                                penalty_advantage += 100000
                                break

                    if obj_name.casefold() == 'stoveburner':
                        # try to find an empty stove
                        if len(obj['receptacleObjectIds']) > 0:
                            penalty_advantage += 10000

                    if priority_in_visibility and obj['visible'] is False:
                        penalty_advantage += 1000

                    if priority_sliced and '_Slice' in obj['name']:
                        penalty_advantage += -100  # prefer sliced objects; this prevents picking up non-sliced objects

                    if obj["distance"] + penalty_advantage < min_distance:
                        min_distance = obj["distance"] + penalty_advantage
                        obj_data = obj
                        obj_id = obj["objectId"]

        return obj_id, obj_data

    def pick(self, obj_name):
        obj_id, obj_data = self.get_obj_id_from_name(obj_name, only_pickupable=True, priority_in_visibility=True, priority_sliced=self.sliced)

        ret_msg = ''
        log.info(f'pick {obj_id}')

        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to pick up. Find the object before picking up it'
        else:
            if obj_data['visible'] is False and obj_data['parentReceptacles'] is not None and len(obj_data['parentReceptacles']) > 0:
                # recep_name = obj_data["parentReceptacles"][0].split('|')[0]
                recep_name  = obj_data["parentReceptacles"][0]
                ret_msg = f'{obj_name} is not visible because it is in {recep_name}. Note: multiple instances of {recep_name} may exist'

                # try anyway
                super().step(dict(
                    action="PickupObject",
                    objectId=obj_id,
                    forceAction=False
                ))
            else:
                super().step(dict(
                    action="PickupObject",
                    objectId=obj_id,
                    forceAction=False
                ))
                
                if not self.last_event.metadata['lastActionSuccess']:
                    if len(self.last_event.metadata['inventoryObjects']) == 0:
                        ret_msg = f'Robot is not holding any object'
                    else:
                        # check if the agent is holding the object
                        holding_obj_id = self.last_event.metadata['inventoryObjects'][0]['objectId']
                        holding_obj_type = self.last_event.metadata['inventoryObjects'][0]['objectType']
                        ret_msg = f'Robot is currently holding {holding_obj_type}'

            if self.last_event.metadata['lastActionSuccess']:
                ret_msg = ''

        return ret_msg

    def put(self, receptacle_name):
        # assume the agent always put the object currently holding
        ret_msg = ''
        orig_receptacle_name = receptacle_name

        if len(self.last_event.metadata['inventoryObjects']) == 0:
            ret_msg = f'Robot is not holding any object'
            return ret_msg
        else:
            holding_obj_id = self.last_event.metadata['inventoryObjects'][0]['objectId']

        
        halt = False
        last_recep_id = None
        exclude_obj_id = None
        for k in range(2):  # try closest and next closest one
            for j in range(7):  # move/look around or rotate obj
                for i in range(2):  # try inherited receptacles too (e.g., sink basin, bath basin)
                    if k == 1 and exclude_obj_id is None:
                        exclude_obj_id = last_recep_id  # previous recep id

                    # for the second round, find another receptacle
                    if k == 0 and '|' in orig_receptacle_name: 
                        if i == 1:
                            continue
                        recep_id = orig_receptacle_name
                        receptacle_name = orig_receptacle_name.split('|')[0]
                    else:
                        if 'Sink' in receptacle_name or 'Bathtub' in receptacle_name: # sink base
                            if i == 0:
                                recep_id, _ = self.get_obj_id_from_name(receptacle_name, get_inherited=True, exclude_obj_id=exclude_obj_id)
                            else:
                                recep_id, _ = self.get_obj_id_from_name(receptacle_name, exclude_obj_id=exclude_obj_id)
                        else:
                            if i == 0:
                                recep_id, _ = self.get_obj_id_from_name(receptacle_name, exclude_obj_id=exclude_obj_id)
                            else:
                                recep_id, _ = self.get_obj_id_from_name(receptacle_name, get_inherited=True, exclude_obj_id=exclude_obj_id)

                    if not recep_id:
                        ret_msg = f'Putting the object on {receptacle_name} failed. First check whether the receptacle is open or not. Also try other instances of the receptacle'
                        continue

                    print(f'put {holding_obj_id} on {recep_id}')

                    # look up (put action fails when a receptacle is not visible)
                    if j == 1:
                        super().step(dict(action="LookUp"))
                        super().step(dict(action="LookUp"))
                    elif j == 2:
                        super().step(dict(action="LookDown"))
                        super().step(dict(action="LookDown"))
                        super().step(dict(action="LookDown"))
                        super().step(dict(action="LookDown"))
                    elif j == 3:
                        super().step(dict(action="LookUp"))
                        super().step(dict(action="LookUp"))
                        super().step(dict(action="MoveBack"))
                    elif j == 4:
                        super().step(dict(action="MoveAhead"))
                        for r in range(4):
                            super().step(dict(action="MoveRight"))
                    elif j == 5:
                        for r in range(8):
                            super().step(dict(action="MoveLeft"))
                    elif j == 6:
                        for r in range(4):
                            super().step(dict(action="MoveRight"))
                        super().step(dict(  # this somehow make putobject success in some cases
                            action="RotateHand",
                            x=40
                        ))

                    super().step(dict(action="PutObject",objectId=holding_obj_id, receptacleObjectId=recep_id, forceAction=True))
                    last_recep_id = recep_id

                    if not self.last_event.metadata['lastActionSuccess']:
                        logging.warning(f"PutObject action failed: {self.last_event.metadata['errorMessage']}, trying again...")
                        ret_msg = f'Putting the object on {receptacle_name} failed. First check the receptacle is open or not. Also try other instances of the receptacle'
                    else:
                        ret_msg = ''
                        halt = True
                        break
                if halt:
                    break
            if halt:
                break

        return ret_msg

    def drop(self):
        log.info(f'drop')
        ret_msg = ''
        super().step(dict(
            action="DropHandObject",
            forceAction=True
        ))

        if not self.last_event.metadata['lastActionSuccess']:
            if len(self.last_event.metadata['inventoryObjects']) == 0:
                ret_msg = f'Robot is not holding any object'
            else:
                ret_msg = f"Drop action failed"
        else:
            ret_msg = ''

        return ret_msg

    def open(self, obj_name):
        log.info(f'open {obj_name}')
        ret_msg = ''
        # obj_id, _ = self.get_obj_id_from_name(obj_name)
        # get the object location
        if '|' in obj_name:
            obj_id = obj_name
            obj_name = obj_name.split('|')[0]
        else:
            obj_id, _ = self.get_obj_id_from_name(obj_name)


        if obj_id is None:
            ret_msg = f"Cannot find {obj_name} to open. Find the object before opening it"
        else:
            open_flag = False
            for ob in self.last_event.metadata['objects']:
                if ob['objectId'] == obj_id and ob['openable'] and ob['isOpen']:
                    open_flag = True
                    break

            for i in range(4):
                super().step(dict(
                    action="OpenObject",
                    objectId=obj_id,
                ))

                if not self.last_event.metadata['lastActionSuccess']:
                    log.warning(
                        f"OpenObject action failed: {self.last_event.metadata['errorMessage']}, moving backward and trying again...")
                    if open_flag:
                        ret_msg = f"Open action failed. The {obj_name} is already open"
                    else:
                        ret_msg = f"Open action failed."

                    # move around to avoid self-collision
                    if i == 0:
                        super().step(dict(action="MoveBack"))
                    elif i == 1:
                        super().step(dict(action="MoveBack"))
                        super().step(dict(action="MoveRight"))
                    elif i == 2:
                        super().step(dict(action="MoveLeft"))
                        super().step(dict(action="MoveLeft"))
                else:
                    ret_msg = ''
                    break

        return ret_msg

    def close(self, obj_name):
        log.info(f'close {obj_name}')
        ret_msg = ''
        if '|' in obj_name:
            obj_id = obj_name
            obj_name = obj_name.split('|')[0]
        else:
            obj_id, _ = self.get_obj_id_from_name(obj_name)

        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to close'
        else:
            super().step(dict(
                action="CloseObject",
                objectId=obj_id,
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                ret_msg = f"Close action failed"
            
                for ob in self.last_event.metadata['objects']:
                    if ob['objectId'] == obj_id and ob['openable'] and not ob['isOpen']:
                        ret_msg += f". The {obj_name} is already closed"
                        break

        return ret_msg

    def toggleon(self, obj_name):
        log.info(f'toggle on {obj_name}')
        ret_msg = ''
        obj_id, _ = self.get_obj_id_from_name(obj_name, only_toggleable=True)
        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to turn on'
        else:
            try:
                super().step(dict(
                    action="ToggleObjectOn",
                    objectId=obj_id,
                ))
                if not self.last_event.metadata['lastActionSuccess']:
                    ret_msg = f"Turn on action failed"
            except:
                ret_msg = f"Turn on action failed"
                self.last_event.metadata['lastActionSuccess'] = False

        return ret_msg

    def toggleoff(self, obj_name):
        log.info(f'toggle off {obj_name}')
        ret_msg = ''
        obj_id, _ = self.get_obj_id_from_name(obj_name, only_toggleable=True)
        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to turn off'
        else:
            super().step(dict(
                action="ToggleObjectOff",
                objectId=obj_id,
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                ret_msg = f"Turn off action failed"

        return ret_msg

    def slice(self, obj_name):
        log.info(f'slice {obj_name}')
        ret_msg = ''
        obj_id, _ = self.get_obj_id_from_name(obj_name)
        if obj_id is None:
            ret_msg = f'Cannot find {obj_name} to slice'
        else:
            super().step(dict(
                action="SliceObject",
                objectId=obj_id,
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                ret_msg = f"Slice action failed"

        return ret_msg
