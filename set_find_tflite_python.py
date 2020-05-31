##Finds sets from an image of set cards
##'numcards' is set at 12 as default in 'find_set'

##Import stuff
import cv2
import tflite_runtime.interpreter as tflite
import numpy as np
import matplotlib.pyplot as plt
#import IPython.display as display
from PIL import Image
import os
from numpy import asarray
from itertools import combinations
##########

##define interpreters
number_interpreter = tflite.Interpreter('NUMBER_model_1.tflite')
shape_interpreter = tflite.Interpreter('SHAPE_model_2.tflite')
fill_interpreter = tflite.Interpreter('FILL_model_2.tflite')

#A function to view an array
def view(array):
    plt.imshow(array)
    plt.xticks(ticks=[])
    plt.yticks(ticks=[])
    plt.ylabel('sets')
    plt.xlabel('cards')
    plt.show()
    
##Given .jpg file, returns 'warps', 'rects', 'cards'
def get_cards(image, numcards=1):
    print('Processing image and finding cards...')
    cards = []
    warps = []
    rects = []
    numcards = numcards
    im = cv2.imread(image)
    #im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(1,1),1000)
    flag, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
    del blur, gray

    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea,reverse=True)[:numcards]

    for i in range(numcards):
        card = contours[i]
        cards.append(card)
        peri = cv2.arcLength(card,True)
        approx = cv2.approxPolyDP(card,0.02*peri,True)
        approx = approx.astype('float32')

        rect = cv2.minAreaRect(contours[i])
        r = cv2.boxPoints(rect)
        rects.append(rect)

        h = np.array([ [449,0],[0,0],[0,449],[449,449] ],np.float32)
        transform = cv2.getPerspectiveTransform(approx,h)
        warp = cv2.warpPerspective(im,transform,(450,450))
        warps.append(warp)
    del cards, contours, hierarchy

    return warps

##define card_color_from_array
red_lower = 80
red_upper = 70
green_upper = 50
green_lower = 40
def card_color_from_array(array):
    img = array
    if (img[:,:,2].min()>red_lower)&(img[:,:,0].min()<red_upper)&(max([img[:,:,0].min(), img[:,:,1].min(), img[:,:,2].min()]) == img[:,:,2].min()):
        return 0
    elif (img[:,:,1].min()>green_lower)&(img[:,:,2].min()<green_upper)&(max([img[:,:,0].min(), img[:,:,1].min(), img[:,:,2].min()]) == img[:,:,1].min()):
        return 1
    else: return 2
 
#define get_props to get properties of each card in photo from a file
#numcard is how many cards you expect to be in the photo
def get_props(file, numcards=12):

    #load and process image
    ############
    ############
    warps = get_cards(file, numcards=12)
    card_list = []
    for card in warps:
        card_list.append(card)
    card_arrays = np.stack(card_list, axis=0)
    card_array_proc = []
    for i in range(len(card_list)):
        card_array_proc.append(card_arrays[i]/255)
    card_array_proc = np.stack(card_array_proc, axis=0)
    ###########
    ###########

    #GET PROPERTIES
    #get number
    print('Running number model...')
    # Load TFLite model and allocate tensors.
    interpreter = number_interpreter
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    number_list = []
    for j in range(numcards):
    # Use TensorFlow Lite model on card_array_proc.
        input_data = np.array(card_array_proc[j].reshape((1,450,450,3)), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        tflite_results = interpreter.get_tensor(output_details[0]['index'])
        number_list.append(list(tflite_results[0]).index(max(list(tflite_results[0]))))
        ##############

    #get shape
    print('Running shape model...')
    # Load TFLite model and allocate tensors.
    #interpreter = tf.lite.Interpreter(model_content=tflite_shape)
    interpreter = shape_interpreter
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    shape_list = []
    for j in range(numcards):
        # Use TensorFlow Lite model on card_array_proc.
        input_data = np.array(card_arrays[j].reshape((1,450,450,3)), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        tflite_results = interpreter.get_tensor(output_details[0]['index'])
        shape_list.append(list(tflite_results[0]).index(max(list(tflite_results[0]))))
    #############

    #get fill
    print('Running fill model...')
    # Load TFLite model and allocate tensors.
    #interpreter = tf.lite.Interpreter(model_content=tflite_fill)
    interpreter = fill_interpreter
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    fill_list = []
    for j in range(numcards):
        # Use TensorFlow Lite model on card_array_proc.
        input_data = np.array(card_arrays[j].reshape((1,450,450,3)), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        tflite_results = interpreter.get_tensor(output_details[0]['index'])
        fill_list.append(list(tflite_results[0]).index(max(list(tflite_results[0]))))
    ############

    #get color
    print('Crunching color numbers...')
    color_list = []
    for i in range(numcards):
        color_list.append( card_color_from_array(warps[i]) )
    ###########

    return number_list, shape_list, fill_list, color_list, warps

def find_sets(file, numcards=12):
    
    number_preds, shape_preds, fill_preds, color_preds, warps = get_props(file, numcards)
    number_preds = np.array(number_preds)
    shape_preds = np.array(shape_preds)
    fill_preds = np.array(fill_preds)
    color_preds = np.array(color_preds)
    prop_array = np.stack([number_preds,shape_preds,fill_preds,color_preds])
     
    
    print('Finding sets...')
    sets = []
    for comb in combinations(range(numcards),3):
        comb_list = list(comb)
        i = comb_list[0]
        j = comb_list[1]
        k = comb_list[2]
        len0 = len(set([prop_array[0,i], prop_array[0,j], prop_array[0,k]]))
        len1 = len(set([prop_array[1,i], prop_array[1,j], prop_array[1,k]]))
        len2 = len(set([prop_array[2,i], prop_array[2,j], prop_array[2,k]]))
        len3 = len(set([prop_array[3,i], prop_array[3,j], prop_array[3,k]]))
        if ( (len0==1)|(len0==3) ) & ((len1==1)|(len1==3)) & ((len2==1)|(len2==3)) & ((len3==1)|(len3==3) ):
            print('SET!')
            print('cards: ',i,j,k)
            sets.append([i,j,k])
    if len(sets)==0:
        print('NO SET!')
    if len(sets)!=0:
        print('There are ', len(sets), ' sets!')
        set_arrays = []
        for set_list in sets:
            set_array = np.concatenate([warps[set_list[0]],warps[set_list[1]],warps[set_list[2]]],axis=1)
            set_arrays.append(set_array)
        all_sets = np.concatenate(set_arrays,axis=0)
        all_sets_rgb = cv2.cvtColor(all_sets, cv2.COLOR_BGR2RGB)
        view(all_sets_rgb)

    return sets

