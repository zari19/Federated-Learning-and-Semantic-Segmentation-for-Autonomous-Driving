import numpy as np
from ultralytics import YOLO
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import torch
import matplotlib.pyplot as plt
import shutil
from scipy.stats import mode
import random
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode


def yolo_model():
    model = YOLO("yolov8x-seg.pt")
    return model


def yolo_prediction(model, image):
    predict = model.predict(image , save = False , save_txt = True, show_labels=True, show=True)
    return predict


def pick_first_elem(txt):
    first_elements = []

    with open(txt, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts:  
                first_element = parts[0]
                first_elements.append(first_element)

    # Print the list of first elements
    #print(first_elements)
    return first_elements

def pick_mask(first_elements, predict, counter, flag_file):
    mask = []
    if flag_file == False:
        for i in range(len(first_elements)):
            mask_val = predict[counter].masks[i].data.clone() 
            mask_val[mask_val == 0] = -1  
            mask.append(mask_val)
    else:
        shape = (1, 384, 640)
        fill_value = -1
        mask.append(torch.full(shape, fill_value))
    return mask

def create_dict(first_elements, masks):
    result_dict = {}


    for key, value in zip(first_elements, masks):
        key_int = int(key)
        
        if key_int in result_dict:
            result_dict[key_int].append(value)
        else:
            result_dict[key_int] = [value]

    return result_dict

def process_tensor(tensor, value):
    return torch.where(tensor == 1, torch.tensor(value, device=tensor.device), tensor)

def process_dictionary(input_dict):
    processed_dict = {}

    for key, tensors in input_dict.items():
        processed_tensors = [process_tensor(tensor, key) for tensor in tensors]
        processed_dict[key] = processed_tensors

    return processed_dict

def get_result_matrix(output_dict):
    array_shape = output_dict[list(output_dict.keys())[0]][0].shape

    device = next(iter(output_dict.values()))[0].device
    result = torch.ones(array_shape, dtype=torch.int, device=device) * -1

    for key, tensor_list in output_dict.items():
        for tensor in tensor_list:
            tensor = tensor.to(device)  
            result = torch.where(tensor >= 0, torch.tensor(key, device=device), result)

    result_cpu = result.cpu()
    result_numpy = result_cpu.numpy()

    return result_numpy

def apply_class_mapping(value):
    return class_map.get(value, value)

def get_class_map():
    class_map = {
    0: 11,    # person
    1: 15,    # bicycle
    2: 13,    # car : vehicle
    3: 14,    # motorcycle
    4: -1,    # airplane (remapped to -1)
    5: 13,    # bus : vehicle
    6: -1,    # train (remapped to -1)
    7: 13,    # truck : vehicle
    8: -1,    # boat (remapped to -1)
    9: 6,     # traffic light
    10: -1,   # fire hydrant (remapped to -1)
    11: 7,   # stop sign (remapped to -1)
    12: -1,   # parking meter (remapped to -1)
    13: -1,   # bench (remapped to -1)
    14: -1,   # bird (remapped to -1)
    15: -1,   # cat (remapped to -1)
    16: -1,   # dog (remapped to -1)
    17: -1,   # horse (remapped to -1)
    18: -1,   # sheep (remapped to -1)
    19: -1,   # cow (remapped to -1)
    20: -1,   # elephant (remapped to -1)
    21: -1,   # bear (remapped to -1)
    22: -1,   # zebra (remapped to -1)
    23: -1,   # giraffe (remapped to -1)
    24: -1,   # backpack (remapped to -1)
    25: -1,   # umbrella (remapped to -1)
    26: -1,   # handbag (remapped to -1)
    27: -1,   # tie (remapped to -1)
    28: -1,   # suitcase (remapped to -1)
    29: -1,   # frisbee (remapped to -1)
    30: -1,   # skis (remapped to -1)
    31: -1,   # snowboard (remapped to -1)
    32: -1,   # sports ball (remapped to -1)
    33: -1,   # kite (remapped to -1)
    34: -1,   # baseball bat (remapped to -1)
    35: -1,   # baseball glove (remapped to -1)
    36: -1,   # skateboard (remapped to -1)
    37: -1,   # surfboard (remapped to -1)
    38: -1,   # tennis racket (remapped to -1)
    39: -1,   # bottle (remapped to -1)
    40: -1,   # wine glass (remapped to -1)
    41: -1,   # cup (remapped to -1)
    42: -1,   # fork (remapped to -1)
    43: -1,   # knife (remapped to -1)
    44: -1,   # spoon (remapped to -1)
    45: -1,   # bowl (remapped to -1)
    46: -1,   # banana (remapped to -1)
    47: -1,   # apple (remapped to -1)
    48: -1,   # sandwich (remapped to -1)
    49: -1,   # orange (remapped to -1)
    50: -1,   # broccoli (remapped to -1)
    51: -1,   # carrot (remapped to -1)
    52: -1,   # hot dog (remapped to -1)
    53: -1,   # pizza (remapped to -1)
    54: -1,   # donut (remapped to -1)
    55: -1,   # cake (remapped to -1)
    56: -1,   # chair (remapped to -1)
    57: -1,   # couch (remapped to -1)
    58: -1,   # potted plant (remapped to -1)
    59: -1,   # bed (remapped to -1)
    60: -1,   # dining table (remapped to -1)
    61: -1,   # toilet (remapped to -1)
    62: -1,   # tv (remapped to -1)
    63: -1,   # laptop (remapped to -1)
    64: -1,   # mouse (remapped to -1)
    65: -1,   # remote (remapped to -1)
    66: -1,   # keyboard (remapped to -1)
    67: -1,   # cell phone (remapped to -1)
    68: -1,   # microwave (remapped to -1)
    69: -1,   # oven (remapped to -1)
    70: -1,   # toaster (remapped to -1)
    71: -1,   # sink (remapped to -1)
    72: -1,   # refrigerator (remapped to -1)
    73: -1,   # book (remapped to -1)
    74: -1,   # clock (remapped to -1)
    75: -1,   # vase (remapped to -1)
    76: -1,   # scissors (remapped to -1)
    77: -1,   # teddy bear (remapped to -1)
    78: -1,   # hair drier (remapped to -1)
    79: -1    # toothbrush (remapped to -1)
}
    return class_map


def update_dictionary_values(original_dict, update_dict):
    for key, value in update_dict.items():
        if key in original_dict:
            original_dict[key] = value
    return original_dict

def update_dictionary_keys(original_dict, update_dict):
    updated_dict = {}
    for key in original_dict:
        if key in update_dict:
            updated_dict[update_dict[key]] = original_dict[key]
        else:
            updated_dict[key] = original_dict[key]
    return updated_dict

def upscale_matrix(matrix, output_shape):
    #matrix= matrix[0,:,:]
    input_rows, input_cols = matrix.shape
    output_rows, output_cols = output_shape
    
    row_scale = output_rows / input_rows
    col_scale = output_cols / input_cols
    
    new_matrix = np.zeros(output_shape, dtype=matrix.dtype)
    
    for i in range(output_rows):
        for j in range(output_cols):
            original_i = int(i // row_scale)
            original_j = int(j // col_scale)
            new_matrix[i, j] = matrix[original_i, original_j]
    
    return new_matrix

def create_dict_probs(first_elements, probs):
    result_dict = {}


    for key, value in zip(first_elements, probs):
        key_int = int(key)
        
        if key_int in result_dict:
            result_dict[key_int].append(value)
        else:
            result_dict[key_int] = [value]

    return result_dict

def weighted_average(probabilities, exponent):  
    probabilities_cpu = [p.cpu().numpy() for p in probabilities]  
    # Calculate the weights by raising each probability to the given exponent
    weights = np.power(probabilities_cpu, exponent)
    
    # Normalize the weights to make sure they sum up to 1
    normalized_weights = weights / np.sum(weights)
    
    # Calculate the weighted average
    weighted_avg = np.dot(probabilities_cpu, normalized_weights)
    
    return weighted_avg


def average_dictionary_values(dictionary):
    result = {}
    exponent = 0.95
    for key in dictionary:
        #print(f'key average_dictionary_values = {key}')
        if isinstance(dictionary[key], list):

            
            # Calculate the weighted average
            #average_value = sum(values) / len(values)
            #print(type(result))

            result[key] = weighted_average(dictionary[key], exponent)

    return result


def map_values(matrix, class_map):
    default_value = -1  # Choose a default value when no mapping is found
    remapped_matrix = np.vectorize(lambda x: class_map.get(x, default_value))(matrix)
    return remapped_matrix

def most_common_neighbor(matrix, x, y, radius):
    center_row = x
    center_col = y
    row_count = len(matrix)
    col_count = len(matrix[0])
    element_counter = Counter()

    for row in range(center_row - radius, center_row + radius + 1):
        for col in range(center_col - radius, center_col + radius + 1):
            if 0 <= row < row_count and 0 <= col < col_count:
                element_counter[matrix[row][col]] += 1

    if element_counter:
        most_common_element = element_counter.most_common(1)
        return most_common_element[0][0]
    else:
        # Return a default value when no valid elements are found
        return matrix[x,j]



def most_frequent(List):
    return max(set(List), key = List.count)


def get_neighbors(matrix, center_row, center_col, radius):
    neighbors = []
    
    for row in range(center_row - radius, center_row + radius + 1):
        for col in range(center_col - radius, center_col + radius + 1):
            if 0 <= row < len(matrix) and 0 <= col < len(matrix[0]):
                if matrix[row][col]!=-1:
                    neighbors.append(matrix[row][col])
    
    return most_frequent(neighbors)


def compare_arrays2(array1, array2, prob_array1, prob_array2):
    print("in compare arrays")
    DELTA = 0.3
    radius = 2
    THR = 0.1

    same_pred = 0
    diff_pred_delta = 0
    other_pred = 0
    pick_yolo = 0
    pick_dlb = 0
    no_yolo = 0

    prob_array1[prob_array1!=-2]+=0.2
    if array1.shape != array2.shape:
        raise ValueError("Input arrays must have the same shape")
    
    result_values = np.zeros(array1.shape, dtype=int)
  
    
    for i in range(array1.shape[0]):
        for j in range(array1.shape[1]):
            #print(array2[i, j])
            if array2[i, j]>=0:
                if prob_array1[i, j]>THR and prob_array2[i, j]> THR:

                    if abs(prob_array1[i, j] - prob_array2[i, j]) < DELTA:
                          if array1[i, j] == array2[i, j]:
                              result_values[i, j] = array1[i, j]
                              same_pred +=1

                          else:
                              diff_pred_delta+=1
                              value1, prob1 = get_neighbors_new(array1, prob_array1, i, j, radius)
                              value2, prob2 = get_neighbors_new(array2, prob_array2,i, j, radius)

                              if value1 == value2:
                                      result_values[i, j] = value1
                              elif prob1>prob2:
                                      result_values[i, j] = value1
                              elif prob2>prob1:
                                      result_values[i, j] = value2

                        

                    else:

                        if prob_array2[i, j]>prob_array1[i, j]:
                              result_values[i, j] = array2[i, j]
                        else:
                              other_pred+=1
                              value1, prob1 = get_neighbors_new(array1,prob_array1, i, j, radius)
                              value2, prob2 = get_neighbors_new(array2,prob_array2, i, j, radius)

                              if value1 == value2:
                                      result_values[i, j] = value1
                              elif prob1>prob2:
                                      result_values[i, j] = value1
                              elif prob2>prob1:
                                      result_values[i, j] = value2

                      
                elif prob_array1[i, j]>THR and prob_array1[i, j]<=THR:
                      pick_dlb+=1
                      result_values[i, j] = array1[i, j]

                elif prob_array1[i, j]<=THR and prob_array1[i, j]>THR:
                      pick_yolo +=1
                      result_values[i, j] = array2[i, j]



            else:
                  no_yolo+=1
                  result_values[i, j] = array1[i,j]



    tot = same_pred + diff_pred_delta + other_pred + pick_dlb + pick_yolo +no_yolo
    if tot !=0:
        print(f'same pred = {same_pred} ({(same_pred/tot):.4f}%), diff_pred_delta = {diff_pred_delta}({(diff_pred_delta/tot):.4f}%), other_pred = {other_pred}({(other_pred/tot):.4f}%), pick_yolo = {pick_yolo}({(pick_yolo/tot):.4f}%), pick_dlb = {pick_dlb}({(pick_dlb/tot):.4f}%), no_yolo = {no_yolo}({(no_yolo/tot):.4f}%)')
    else:
        print("tot = 0 error")
    return result_values
    


def most_recurrent_value(matrix, x, y, radius):
    values = []
    
    for i in range(max(0, x - radius), min(len(matrix), x + radius + 1)):
        for j in range(max(0, y - radius), min(len(matrix[0]), y + radius + 1)):
            if matrix[i][j] is not None:
                values.append(matrix[i][j])
    
    if not values:
        return None
    
    count = Counter(values)
    most_common_value = max(count, key=count.get)
    
    return most_common_value


def process_matrix(matrix, dictionary):
    rows = len(matrix)
    cols = len(matrix[0])
    result_matrix = np.full((rows, cols), 0)  # Create a NumPy array filled with -1

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] in dictionary:
                result_matrix[i][j] = dictionary[matrix[i][j]]

    return result_matrix


def map_and_insert(matrix, value_dict):
    mapped_matrix = np.empty(matrix.shape, dtype='object')
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            mapped_matrix[i, j] = value_dict.get(value, -1)
    
    return mapped_matrix


def merge_matrices(matrix1, matrix2):
                  rows, cols = matrix1.shape
                  merged_matrix = np.zeros((rows, cols), dtype=int)

                  for i in range(rows):
                      for j in range(cols):
                          if matrix2[i, j] is not None:  # Check for None before assigning
                              merged_matrix[i, j] = matrix2[i, j]
                          else:
                              merged_matrix[i, j] = matrix1[i, j]

                  return merged_matrix




def get_neighbors_new(matrix, prob, center_row, center_col, radius):
    dizionario = {}
    appearences = {}

    b = np.array((center_row, center_col))
    for row in range(center_row - radius, center_row + radius + 1):
        
        #print(f'row = {row}')
        for col in range(center_col - radius, center_col + radius + 1):
            #print(f' col = {col}')
            if 0 <= row < len(matrix) and 0 <= col < len(matrix[0]):
                if matrix[row][col]!=-1 and (center_row != row and center_col != col):
                    a = np.array((row, col))
                    
                    eucliderian_dist = np.linalg.norm(a-b)
                    #print(eucliderian_dist)

                    if matrix[row][col] in dizionario:
                        #print(dizionario)
                        temp = dizionario[matrix[row][col]]
                        temp_app = appearences[matrix[row][col]]

                        dizionario[matrix[row][col]] = prob[row][col] /eucliderian_dist+ temp
                        appearences[matrix[row][col]] = 1 + temp_app
                    else:
                        # If the key does not exist, simply add the new key-value pair
                        dizionario[matrix[row][col]] = prob[row][col]/eucliderian_dist
                        appearences[matrix[row][col]] = 1
                      #neighbors.append(matrix[row][col])

    for key in dizionario:
          dizionario[key] = dizionario[key]/ appearences[key]


    key_with_highest_key = max(dizionario, key=lambda k: int(k))
    value_with_highest_key = dizionario[key_with_highest_key]  
    return key_with_highest_key, value_with_highest_key