''' Implementation of ZoneMapAltCnt Paper.

    For details, please go through the below:
    - ZoneMap(https://bit.ly/2QSE3on)
    - ZoneMapAlt(https://bit.ly/389ruuF)
    - ZoneMapAltCnt(https://bit.ly/2NqvSxo)
'''

import os
import math
import copy
from xml.etree import ElementTree
from operator import itemgetter
from lxml import etree
import pandas as pd
from shapely.geometry import box

ground_truth_folder = './ground-truth'
ocr_output_folder = './ocr-detection'
# Threshold coefficient for coverage area between two zones.
BETA = 0.2
# Split-Merge Coefficient.
MS = 0.5

# Collecting ground-truth files.
gt_files = os.listdir(ground_truth_folder)
dt_files = os.listdir(ocr_output_folder)

class Zone:

    zone_id_counter = 1

    def __init__(self, bbox, text):
        self.id = Zone.zone_id_counter
        Zone.zone_id_counter += 1
        self.bbox = self.get_polygon(bbox)
        self.text = text
        self.groups = []
        self.linked_zones = []

    def get_polygon(self, bbox):
        ''' Return a shapely polygon from bounding
        box cordinates xmin, ymin, xmax, ymax.'''

        xmin, ymin, xmax, ymax = bbox
        polygon = box(xmin, ymin, xmax, ymax)
        return polygon

class ZoneGroup:

    zonegroup_id_counter = 1
    configwise_groups = {'Match': [],
                         'Miss' : [],
                         'FA' : [],
                         'Merge' : [],
                         'Split' : [],
                         'Multiple' : []}

    @classmethod
    def assign_config(cls, gt_card, dt_card):
        ''' Assigns group configuration. '''

        # Miss
        if gt_card > 0 and dt_card == 0:
            return 'Miss'
        # False Alarm
        elif gt_card == 0 and dt_card > 0:
            return 'FA'
        # Match
        elif gt_card == 1 and dt_card == 1:
            return 'Match'
        # Merge
        elif gt_card == 1 and dt_card > 1:
            return 'Split'
        # Split
        elif gt_card > 1 and dt_card == 1:
            return 'Merge'
        # Multiple
        elif gt_card > 1 and dt_card > 1:
            return 'Multiple'
        else:
            return 'Unknown'

    def __init__(self, polygon, gt_zone=None, dt_zone=None, config=None):

        self.id = ZoneGroup.zonegroup_id_counter
        ZoneGroup.zonegroup_id_counter += 1
        self.gt_zone = gt_zone
        self.dt_zone = dt_zone
        self.zone_polygon = polygon
        self.gt_card = 0
        self.dt_card = 0
        if dt_zone == None:
            self.gt_card = 1
        elif gt_zone == None:
            self.dt_card = 1
        else:
            self.gt_card = len(self.dt_zone.linked_zones)
            self.dt_card = len(self.gt_zone.linked_zones)
        if config == None:
            self.group_config = ZoneGroup.assign_config(self.gt_card, self.dt_card)
        else:
            self.group_config = config
        self.configwise_groups[self.group_config].append(self.id)

def extract_bounding_boxes(annotation_file):
    ''' Reads an xml annotation file and returns
        all the objects.'''

    parser = etree.XMLParser(encoding='utf-8')
    xmltree = ElementTree.parse(annotation_file, parser=parser).getroot()
    annotations = {}

    for object in xmltree.findall('object'):
        text = object.find('name').text
        bndbox = object.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        if xmin >= xmax or ymin >= ymax:
            continue
        zone = Zone((xmin, ymin, xmax, ymax), text)
        annotations[zone.id] = zone
    return annotations

def get_link_strength(zoneA, zoneB):
    '''Calculates the link strength between any two 
        bounding boxes.'''

    intersection = zoneA.intersection(zoneB)
    if intersection == 0: return 0
    strength = (math.pow(intersection.area/zoneA.area, 2)
                + math.pow(intersection.area/zoneB.area, 2))
    return strength

def get_linked_zones(gt_zones, dt_zones):
    '''Compares all the gt and dt boxes to find boxes with
       non-zero link forces.'''

    link_table = []
    for gt_idx, gt_zone in gt_zones.items():
        for dt_idx, dt_zone in dt_zones.items():
            link_strength = get_link_strength(gt_zone.bbox, dt_zone.bbox)
            if link_strength != 0:
                link_table.append([gt_idx, dt_idx, link_strength])
    return sorted(link_table, key=itemgetter(2), reverse=True)

def get_coverage(polyA, polyB):
    ''' Returns the overlap ratio of two polygons. '''

    overlap_ratio = 0
    if polyB.area > 0:
        overlap_ratio = polyA.area/polyB.area
    return overlap_ratio

def group_linked_zones(linked_zones, gt_zones, dt_zones):
    '''Allots a link of two zones into Zonegroups.'''

    zone_groups = {}
    for links in linked_zones:
        gt_zone = gt_zones[int(links[0])]
        temp_gt_zone = copy.copy(gt_zone.bbox)
        dt_zone = dt_zones[int(links[1])]
        temp_dt_zone = copy.copy(dt_zone.bbox)
        # Remove used gt references 
        if dt_zone.linked_zones != []:
            for used_gt_zone_id in dt_zone.linked_zones:
                temp_used_gt_zone = copy.copy(gt_zones[used_gt_zone_id].bbox)
                temp_gt_zone = temp_gt_zone.difference(temp_gt_zone.intersection(temp_used_gt_zone))
                temp_dt_zone = temp_dt_zone.difference(temp_dt_zone.intersection(temp_used_gt_zone))
        # Remove used dt references.
        if gt_zone.linked_zones != []:
            for used_dt_zone_id in gt_zone.linked_zones:
                temp_used_dt_zone = copy.copy(dt_zones[used_dt_zone_id].bbox)
                temp_gt_zone = temp_gt_zone.difference(temp_gt_zone.intersection(temp_used_dt_zone))

        remaining_intersection = temp_gt_zone.intersection(temp_dt_zone)
        if get_coverage(remaining_intersection, temp_gt_zone) > BETA:
            gt_zone.linked_zones.append(dt_zone.id)
            dt_zone.linked_zones.append(gt_zone.id)
            zone_group = ZoneGroup(remaining_intersection, gt_zone, dt_zone)
            zone_groups[zone_group.id] = zone_group
    return zone_groups

def group_non_linked_zones(zone_groups, gt_zones, dt_zones):
    ''' Assigns all unlinked zones as either miss or
        false alarms.'''

    #Detect missed out regions.
    for idx, gt_zone in gt_zones.items():
        temp_gt_zone = copy.copy(gt_zone.bbox)
        if gt_zone.linked_zones != []:
            for linked_dt_zone in gt_zone.linked_zones:
                temp_gt_zone = temp_gt_zone.difference(temp_gt_zone.intersection(
                                dt_zones[linked_dt_zone].bbox))
        if temp_gt_zone.area > 0:
            miss_zone_group = ZoneGroup(temp_gt_zone, gt_zone=gt_zone, config='Miss')
            zone_groups[miss_zone_group.id] = miss_zone_group
    #Detect False Alarms.
    for idx, dt_zone in dt_zones.items():
        temp_dt_zone = copy.copy(dt_zone.bbox)
        if dt_zone.linked_zones != []:
            for linked_gt_zone in dt_zone.linked_zones:
                temp_dt_zone = temp_dt_zone.difference(temp_dt_zone.intersection(
                        gt_zones[linked_gt_zone].bbox))
        if temp_dt_zone.area > 0:
            fa_zone_group = ZoneGroup(temp_dt_zone, dt_zone=dt_zone, config='FA')
            zone_groups[fa_zone_group.id] = fa_zone_group
    return zone_groups

def get_total_area(zones):
    ''' Computes the total area for the dict of zones.'''

    area = 0
    for _, zone in zones.items():
        area += zone.bbox.area
    return area

def calc_detection_score(errors, gt_zones):
    ''' Calculates the zonemap detection score.'''

    gt_zones_area = get_total_area(gt_zones)
    total_error_area = (errors['miss'] + errors['false_alarm'] +
                        errors['split'] + errors['merge'] + errors['multiple'])
    zonemapaltcnt_detect = ((float(total_error_area)*100)/float(gt_zones_area))
    errors['zonemapaltcnt_detection_score'] = round(zonemapaltcnt_detect, 2)
    errors['total_gt_area'] = round(gt_zones_area, 2)
    return errors

def calc_detection_error(zone_groups):
    ''' Calculates the detection error based on the group config. '''

    match, miss, false_alarm, split, merge, multiple = (0, 0, 0, 0, 0, 0)
    n_match, n_miss, n_fa, n_split, n_merge, n_multiple = (0, 0, 0, 0, 0, 0)

    for _, zone_group in zone_groups.items():
        config = zone_group.group_config
        area = zone_group.zone_polygon.area
        if config == 'Match':
            match += area
            n_match += 1
        elif config == 'Miss':
            miss += area
            n_miss += 1
        elif config == 'FA':
            false_alarm += area
            n_fa += 1
        elif config == 'Split':
            split += area * MS * zone_group.dt_card
            n_split += 1
        elif config == 'Merge':
            merge += area * MS * zone_group.gt_card
            n_merge += 1
        elif config == 'Multiple':
            multiple += area * MS * (zone_group.gt_card + zone_group.dt_card)
            n_multiple += 1
        else:
            print("Incorrect group config.")
    return {'match':round(match,2),
            'miss':round(miss,2),
            'false_alarm':round(false_alarm,2),
            'split':round(split,2),
            'merge':round(merge,2),
            'multiple':round(multiple,2)}, {'n_match':n_match,
                                    'n_miss':n_miss,
                                    'n_false_alarm':n_fa,
                                    'n_split':n_split,
                                    'n_merge':n_merge,
                                    'n_multiple':n_multiple}

def edit_distance(a, b, threshold=99999):
    ''' Calculates the insertions, deletions and substitutions
        required to change a into b.'''

    if a == b:
        return 0
    m = len(a)
    n = len(b)
    distances = [[threshold for j in range(n + 1)] for i in range(m + 1)]
    for i in range(m + 1):
        distances[i][0] = i
    for j in range(n + 1):
        distances[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                cij = 0
            else:
                cij = 1
            d = min(distances[i - 1][j] + 1, distances[i][j - 1] + 1,
                    distances[i - 1][j - 1] + cij)
            if d >= threshold:
                return d
            distances[i][j] = d
    return distances[m][n]

def arrange_by_pos(zones_list):
    '''Returns a list of zone text sorted by position.'''

    zones_list.sort(key=lambda x : x.bbox.bounds[1]*1000 + x.bbox.bounds[0])
    text_list = [zone.text for zone in zones_list]
    return text_list

def get_char_precision_recall(gt_len, dt_len, edits):
    ''' Returns character level precision, recall.'''

    # Nothing gets recognized.
    if dt_len == 0:
        return (0, 0)
    if gt_len >= dt_len:
        char_correct = gt_len - edits
    else:
        char_correct = dt_len - edits
    if char_correct < 0:
        char_correct = 0
    char_precision = float(char_correct/dt_len)
    char_recall = float(char_correct/gt_len)
    return (char_precision, char_recall)

def get_word_precision_recall(gt_words_count, dt_words_count, correct_words_count):
    '''Returns word level precision, recall.'''

    # Nothing gets recognized.
    if dt_words_count == 0:
        return (0, 0)
    words_precision = float(correct_words_count/dt_words_count)
    words_recall = float(correct_words_count/gt_words_count)
    return (words_precision, words_recall)

def get_word_match_count(gt_word_list, dt_word_list):
    '''Returns number of correctly recognized words.'''

    correct_count = 0
    for word in gt_word_list:
        if word in dt_word_list:
            correct_count += 1
            gt_word_list.remove(word)
            dt_word_list.remove(word)
    return correct_count

def calc_recognition_error(gt_zones, dt_zones, zone_groups, errors):
    ''' Calculates precision and recall for text.'''

    used_gt_zones = []
    used_dt_zones = []
    edits, gt_len, dt_len = (0, 0, 0)
    num_gt_words, num_dt_words, num_correct_words = (0, 0, 0)
    # Calculate for Multiple groups.
    multiples = ZoneGroup.configwise_groups['Multiple']
    for group_id in multiples:
        zone_group = zone_groups[group_id]
        unused_dt_zones = []
        unused_gt_zones = []
        for dt_zone_id in zone_group.gt_zone.linked_zones:
            if dt_zone_id in used_dt_zones: continue
            used_dt_zones.append(dt_zone_id)
            unused_dt_zones.append(dt_zones[dt_zone_id])
        for gt_zone_id in zone_group.dt_zone.linked_zones:
            if gt_zone_id in used_gt_zones: continue
            used_gt_zones.append(gt_zone_id)
            unused_gt_zones.append(gt_zones[gt_zone_id])
        gt_text_list = arrange_by_pos(unused_gt_zones)
        dt_text_list = arrange_by_pos(unused_dt_zones)
        gt_len += len("".join(gt_text_list))
        dt_len += len("".join(dt_text_list))
        edits += edit_distance("".join(dt_text_list), "".join(gt_text_list))
        num_gt_words += len(gt_text_list)
        num_dt_words += len(dt_text_list)
        num_correct_words += get_word_match_count(gt_text_list, dt_text_list)
    # Calculate for Splits.
    splits = ZoneGroup.configwise_groups['Split']
    for group_id in splits:
        zone_group = zone_groups[group_id]
        unused_dt_zones = []
        if zone_group.gt_zone.id in used_gt_zones: continue
        used_gt_zones.append(zone_group.gt_zone.id)
        for dt_zone_id in zone_group.gt_zone.linked_zones:
            if dt_zone_id in used_dt_zones: continue
            used_dt_zones.append(dt_zone_id)
            unused_dt_zones.append(dt_zones[dt_zone_id])
        gt_text = zone_group.gt_zone.text
        dt_text_list = arrange_by_pos(unused_dt_zones)
        gt_len += len(gt_text)
        dt_len += len("".join(dt_text_list))
        edits = edit_distance("".join(dt_text_list), gt_text)
        num_gt_words += 1
        num_dt_words += len(dt_text_list)
        num_correct_words += get_word_match_count([gt_text], dt_text_list)
    # Calculate for Merges.
    merges = ZoneGroup.configwise_groups['Merge']
    for group_id in merges:
        zone_group = zone_groups[group_id]
        unused_gt_zones = []
        if zone_group.dt_zone.id in used_dt_zones: continue
        used_dt_zones.append(zone_group.dt_zone.id)
        for gt_zone_id in zone_group.dt_zone.linked_zones:
            if gt_zone_id in used_gt_zones: continue
            used_gt_zones.append(gt_zone_id)
            unused_gt_zones.append(gt_zones[gt_zone_id])
        dt_text = zone_group.dt_zone.text
        gt_text_list = arrange_by_pos(unused_gt_zones)
        gt_len += len("".join(gt_text_list))
        dt_len += len(dt_text)
        edits += edit_distance(dt_text, "".join(gt_text_list))
        num_gt_words += len(gt_text_list)
        num_dt_words += 1
        num_correct_words += get_word_match_count(gt_text_list, [dt_text])
    # Calculate for Matches.
    matches = ZoneGroup.configwise_groups['Match']
    for group_id in matches:
        zone_group = zone_groups[group_id]
        if zone_group.gt_zone.id in used_gt_zones: continue
        if zone_group.dt_zone.id in used_dt_zones: continue
        used_gt_zones.append(zone_group.gt_zone.id)
        used_dt_zones.append(zone_group.dt_zone.id)
        gt_text = zone_group.gt_zone.text
        dt_text = zone_group.dt_zone.text
        gt_len += len(gt_text)
        dt_len += len(dt_text)
        edits += edit_distance(dt_text, gt_text)
        num_gt_words += 1
        num_dt_words += 1
        num_correct_words += get_word_match_count([gt_text], [dt_text])
    # Calculate for Miss.
    misses = ZoneGroup.configwise_groups['Miss']
    for group_id in misses:
        zone_group = zone_groups[group_id]
        if zone_group.gt_zone.id in used_gt_zones: continue
        used_gt_zones.append(zone_group.gt_zone.id)
        gt_len += len(zone_group.gt_zone.text)
        edits += len(zone_group.gt_zone.text)
        num_gt_words += 1
    # Calculate for False Alarms
    false_alarms = ZoneGroup.configwise_groups['FA']
    for group_id in false_alarms:
        zone_group = zone_groups[group_id]
        if zone_group.dt_zone.id in used_dt_zones: continue
        used_dt_zones.append(zone_group.dt_zone.id)
        dt_len += len(zone_group.dt_zone.text)
        edits += len(zone_group.dt_zone.text)
        num_dt_words += 1
    char_precision, char_recall = get_char_precision_recall(gt_len, dt_len, edits)
    word_precision, word_recall = get_word_precision_recall(num_gt_words, num_dt_words, num_correct_words)
    return char_precision, char_recall, word_precision , word_recall

def compare_files(ground_truth_file, ocr_output_file):
    ''' Calculates the ZoneMap Error for an ocr output.'''

    # Get bounding boxes.
    print(ground_truth_file, ocr_output_file)
    gt_zones = extract_bounding_boxes(ground_truth_file)
    dt_zones = extract_bounding_boxes(ocr_output_file)
    # Boxes with non-zero link forces.
    linked_zones = get_linked_zones(gt_zones, dt_zones)
    # Allot each box/zone from gt and dt into groups.
    zone_groups = group_linked_zones(linked_zones, gt_zones, dt_zones)
    zone_groups = group_non_linked_zones(zone_groups, gt_zones, dt_zones)
    errors, n_errors = calc_detection_error(zone_groups)
    errors = calc_detection_score(errors, gt_zones)
    cp, cr, wp, wr = calc_recognition_error(gt_zones, dt_zones, zone_groups, errors)
    errors["cp"], errors["cr"] = (cp, cr)
    errors["wp"], errors["wr"] = (wp, wr)
    return errors, n_errors

def dump_report(results, outfile, errors, n_errors):
    ''' Dumps the error info in results file.'''

    ret = {}
    total = errors["total_gt_area"]
    ret["file_id"] = os.path.splitext(outfile)[0]
    ret["zonemap"] = errors["zonemapaltcnt_detection_score"]
    ret["match"] = round(errors["match"]/total, 2)
    ret["merge"] = round(errors["merge"]/total, 2)
    ret["split"] = round(errors["split"]/total, 2)
    ret["miss"] = round(errors["miss"]/total, 2)
    ret["false_alarm"] = round(errors["false_alarm"]/total, 2)
    ret["multiple"] = round(errors["multiple"]/total, 2)
    ret["char_precision"] = round(errors["cp"], 2)
    ret["char_recall"] = round(errors["cr"], 2)
    ret["word_precision"] = round(errors["wp"], 2)
    ret["word_recall"] = round(errors["wr"], 2)
    for k, v in n_errors.items():
        ret[k] = v
    results = results.append(ret, ignore_index=True)
    return results

results = pd.DataFrame(columns=["file_id", "zonemap", "match", "merge", "split", "miss", "false_alarm",
                                "multiple", "char_precision", "char_recall", "word_precision", "word_recall",
                                 "n_match", "n_merge", "n_split", "n_miss", "n_false_alarm", "n_multiple"])

def reset_attributes():
    ''' Reset Zone and ZoneGroup attributes after 
        processing each pair.'''
        
    Zone.zone_id_counter = 1
    ZoneGroup.zonegroup_id_counter = 1
    for k, v in ZoneGroup.configwise_groups.items():
        ZoneGroup.configwise_groups[k] = []
    return
counter = 0
for gt_file in gt_files:

    if not os.path.isfile(os.path.join(ocr_output_folder, gt_file)):
        print("ocr output file not present: ", gt_file)
        continue
    print(counter)
    try:
        errors, n_errors = compare_files(os.path.join(ground_truth_folder, gt_file),
                                         os.path.join(ocr_output_folder, gt_file))
    except:
        print("ERROR ERROR ERROR ERROR ")
        continue
    results = dump_report(results, gt_file, errors, n_errors)
    reset_attributes()
    counter += 1
results.to_csv("results.csv", index=False)