import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

def change_brightness(image, value): 
    img = image.copy() 
    r_value, g_value, b_value = image_to_rgb(img)
    new_r = []
    new_g = []
    new_b = []
    for pixel in r_value:
        pixel += value
        if pixel > 255:
            pixel = 255
            new_r.append(pixel)
        elif pixel < 0:
            pixel = 0
            new_r.append(pixel)
        else:
            new_r.append(pixel)
    for pixel in g_value:
        pixel += value
        if pixel > 255:
            pixel = 255
            new_g.append(pixel)
        elif pixel < 0:
            pixel = 0
            new_g.append(pixel)
        else:
            new_g.append(pixel)
    for pixel in b_value:
        pixel += value
        if pixel > 255:
            pixel = 255
            new_b.append(pixel)
        elif pixel < 0:
            pixel = 0
            new_b.append(pixel)
        else:
            new_b.append(pixel)
    return rgb_to_image(new_r, new_g, new_b, img)
    

def change_contrast(image, value):
    img = image.copy()
    r_value, g_value, b_value = image_to_rgb(img)
    
    c_f = (259 * (value + 255)) / (255 * (259 - value))
    new_r = []
    new_g = []
    new_b = []
    
    for i in r_value:
        i = c_f * (i - 128) + 128
        if i > 255: 
            i = 255
            new_r.append(i)
        elif i < 0:
            i = 0
            new_r.append(i)
        else:
            i = int(i)
            new_r.append(i)
    for i in g_value:
        i = c_f * (i - 128) + 128
        if i > 255: 
            i = 255
            new_g.append(i)
        elif i < 0:
            i = 0
            new_g.append(i)
        else:
            i = int(i)
            new_g.append(i)
    for i in b_value:
        i = c_f * (i - 128) + 128
        if i > 255: 
            i = 255
            new_b.append(i)
        elif i < 0:
            i = 0
            new_b.append(i)
        else:
            i = int(i)
            new_b.append(i)
    
    
    return rgb_to_image(new_r, new_g, new_b, img) 
    


def grayscale(image):
    img = image.copy()
    r_value, g_value, b_value = image_to_rgb(img)

    new_r = []

    for i in range(len(r_value)):
        greyscale_val = r_value[i] * .3 + g_value[i] * .59 + b_value[i] * .11
        new_r.append(int(greyscale_val)) 
    
    return rgb_to_image(new_r, new_r, new_r, img)



def blur_effect(image): 
    img = image.copy()
    r, g, b = image_to_rgb(img)

    lst = np.array(img).tolist()
    col = len(lst[0])
    row = len(lst)

    rarray = np.array(r)
    garray = np.array(g)
    barray = np.array(b)
    rlist = np.pad(np.reshape(rarray, (row, col)), 1, 'constant')
    glist = np.pad(np.reshape(garray, (row, col)), 1, 'constant')
    blist = np.pad(np.reshape(barray, (row, col)), 1, 'constant')

    new_r = []
    new_g = []
    new_b = []

    for j in range(row):
        for i in range(col):
            try:
                val_r = int(rlist[j][i] * .0625 + rlist[j][i + 1] * .125 + rlist[j][i + 2] * .0625 + \
                rlist[j + 1][i] * .125 + rlist[j + 1][i + 1] * .25 + rlist[j + 1][i + 2] * .125 + \
                rlist[j + 2][i] * .0625 + rlist[j + 2][i + 1] * .125 + rlist[j + 2][i + 2] * .0625)
                new_r.append(val_r)
                val_g = int(glist[j][i] * .0625 + glist[j][i + 1] * .125 + glist[j][i + 2] * .0625 + \
                glist[j + 1][i] * .125 + glist[j + 1][i + 1] * .25 + glist[j + 1][i + 2] * .125 + \
                glist[j + 2][i] * .0625 + glist[j + 2][i + 1] * .125 + glist[j + 2][i + 2] * .0625)
                new_g.append(val_g)
                val_b = int(blist[j][i] * .0625 + blist[j][i + 1] * .125 + blist[j][i + 2] * .0625 + \
                blist[j + 1][i] * .125 + blist[j + 1][i + 1] * .25 + blist[j + 1][i + 2] * .125 + \
                blist[j + 2][i] * .0625 + blist[j + 2][i + 1] * .125 + blist[j + 2][i + 2] * .0625)
                new_b.append(val_b)
            except IndexError:
                continue

    new_img = rgb_to_image(new_r, new_g, new_b, img)

    new_img[0] = image[0]
    for j in range(1, row):
        new_img[j][0] = img[j][0]
        new_img[j][-1] = img[j][-1]
    new_img[-1] = img[-1]

    return new_img

def edge_detection(image):
    img = image.copy()
    r, g, b = image_to_rgb(img)

    lst = np.array(img).tolist()
    col = len(lst[0])
    row = len(lst)

    rarray = np.array(r)
    garray = np.array(g)
    barray = np.array(b)
    rlist = np.pad(np.reshape(rarray, (row, col)), 1, 'constant')
    glist = np.pad(np.reshape(garray, (row, col)), 1, 'constant')
    blist = np.pad(np.reshape(barray, (row, col)), 1, 'constant')

    new_r = []
    new_g = []
    new_b = []

    for j in range(row):
        for i in range(col):
            try:
                val_r = int(rlist[j][i] * -1 + rlist[j][i + 1] * -1 + rlist[j][i + 2] * -1 + \
                rlist[j + 1][i] * -1 + rlist[j + 1][i + 1] * 8 + rlist[j + 1][i + 2] * -1 + \
                rlist[j + 2][i] * -1 + rlist[j + 2][i + 1] * -1 + rlist[j + 2][i + 2] * -1)
                new_r.append(val_r)
                val_g = int(glist[j][i] * -1 + glist[j][i + 1] * -1 + glist[j][i + 2] * -1 + \
                glist[j + 1][i] * -1 + glist[j + 1][i + 1] * 8 + glist[j + 1][i + 2] * -1 + \
                glist[j + 2][i] * -1 + glist[j + 2][i + 1] * -1 + glist[j + 2][i + 2] * -1)
                new_g.append(val_g)
                val_b = int(blist[j][i] * -1 + blist[j][i + 1] * -1 + blist[j][i + 2] * -1 + \
                blist[j + 1][i] * -1 + blist[j + 1][i + 1] * 8 + blist[j + 1][i + 2] * -1 + \
                blist[j + 2][i] * -1 + blist[j + 2][i + 1] * -1 + blist[j + 2][i + 2] * -1)
                new_b.append(val_b)
            except IndexError:
                continue

    rval = []
    gval = []
    bval = []            

    for val in new_r:
        val += 128
        if val > 255:
            val = 255
            rval.append(val)
        elif val < 0:
            val = 0
            rval.append(val)
        else:
            rval.append(val)
    for val in new_g:
        val += 128
        if val > 255:
            val = 255
            gval.append(val)
        elif val < 0:
            val = 0
            gval.append(val)
        else:
            gval.append(val)
    for val in new_b:
        val += 128
        if val > 255:
            val = 255
            bval.append(val)
        elif val < 0:
            val = 0
            bval.append(val)
        else:
            bval.append(val)

    new_img = rgb_to_image(rval, gval, bval, img)

    new_img[0] = img[0]
    for j in range(1, row):
        new_img[j][0] = img[j][0]
        new_img[j][-1] = img[j][-1]
    new_img[-1] = img[-1]

    return new_img

    

def embossed(image):
    img = image.copy()
    r, g, b = image_to_rgb(img)

    lst = np.array(img).tolist()
    col = len(lst[0])
    row = len(lst)

    rarray = np.array(r)
    garray = np.array(g)
    barray = np.array(b)
    rlist = np.pad(np.reshape(rarray, (row, col)), 1, 'constant')
    glist = np.pad(np.reshape(garray, (row, col)), 1, 'constant')
    blist = np.pad(np.reshape(barray, (row, col)), 1, 'constant')

    new_r = []
    new_g = []
    new_b = []

    for j in range(row):
        for i in range(col):
            try:
                val_r = int(rlist[j][i] * -1 + rlist[j][i + 1] * -1 + rlist[j][i + 2] * 0 + \
                rlist[j + 1][i] * -1 + rlist[j + 1][i + 1] * 0 + rlist[j + 1][i + 2] * 1 + \
                rlist[j + 2][i] * 0 + rlist[j + 2][i + 1] * 1 + rlist[j + 2][i + 2] * 1)
                new_r.append(val_r)
                val_g = int(glist[j][i] * -1 + glist[j][i + 1] * -1 + glist[j][i + 2] * 0 + \
                glist[j + 1][i] * -1 + glist[j + 1][i + 1] * 0 + glist[j + 1][i + 2] * 1 + \
                glist[j + 2][i] * 0 + glist[j + 2][i + 1] * 1 + glist[j + 2][i + 2] * 1)
                new_g.append(val_g)
                val_b = int(blist[j][i] * -1 + blist[j][i + 1] * -1 + blist[j][i + 2] * 0 + \
                blist[j + 1][i] * -1 + blist[j + 1][i + 1] * 0 + blist[j + 1][i + 2] * 1 + \
                blist[j + 2][i] * 0 + blist[j + 2][i + 1] * 1 + blist[j + 2][i + 2] * 1)
                new_b.append(val_b)
            except IndexError:
                continue
    
    rval = []
    gval = []
    bval = []            

    for val in new_r:
        val += 128
        if val > 255:
            val = 255
            rval.append(val)
        elif val < 0:
            val = 0
            rval.append(val)
        else:
            rval.append(val)
    for val in new_g:
        val += 128
        if val > 255:
            val = 255
            gval.append(val)
        elif val < 0:
            val = 0
            gval.append(val)
        else:
            gval.append(val)
    for val in new_b:
        val += 128
        if val > 255:
            val = 255
            bval.append(val)
        elif val < 0:
            val = 0
            bval.append(val)
        else:
            bval.append(val)

    new_img = rgb_to_image(rval, gval, bval, img)

    new_img[0] = img[0]
    for j in range(1, row):
        new_img[j][0] = img[j][0]
        new_img[j][-1] = img[j][-1]
    new_img[-1] = img[-1]

    return new_img

def rectangle_select(image, x, y):  # image is a mask, x and y are tuples
    img = image.copy()
    x1, x2 = x
    y1, y2 = y
    mask = np.zeros((len(img),len(img[0])))
    for i in range(y2 - x2 + 1):
        for k in range(y1 - x1 + 1):
            mask[x1 + k, x2 + i] = 1
    return mask

def magic_wand_select(image, x, thres):   
    img = image.copy()
    r_dict = {}
    g_dict = {}
    b_dict = {}

    for row in range(len(img)):
        for col in range(len(img[0])):
            r_dict[row, col] = img[row, col][0]
            g_dict[row, col] = img[row, col][1]
            b_dict[row, col] = img[row, col][2]

    stack = []
    mask = np.zeros((len(img), len(img[0]))) # initialise by setting mask to full zeroes

    x1, x2= x # separating row and col 
    stack.append(x) 
    mask[x] = 1 
    r_val = r_dict[x1, x2]  # using first item in stack
    g_val = g_dict[x1, x2]
    b_val = b_dict[x1, x2]

################### FOOLING AROUND WITH STACK D' POINTS™ ###################### 

    while stack != []: # need make sure there is an endpoint, ie at one point goes to 0
        x1, x2 = stack[-1] 
        stack.pop(-1)   # yeet
        # the following was originally meant to be a for loop but then it was 4am and i realised i may be screwing up somewhere

        if x1 - 1 >= 0:
            stack.append((x1 - 1, x2))
            if mask[(x1 - 1, x2)] == 1: # the mask has already been selected so yeet
                stack.pop(-1)
            else: # not selected
                diff_r = abs(r_dict[(x1 - 1, x2)] - r_val)
                diff_g = abs(g_dict[(x1 - 1, x2)] - g_val)
                diff_b = abs(b_dict[(x1 - 1, x2)] - b_val)
                ave_r = ((r_dict[(x1 - 1, x2)] + r_val) / 2)
                dist = math.sqrt((2 + (ave_r / 256)) * (diff_r ** 2) + (4 * diff_g ** 2) + ((2 + (255 - ave_r) / 256) * diff_b ** 2))
                if dist <= thres: # want to select
                    mask[(x1 - 1, x2)] = 1 # select
                else: 
                    stack.pop(-1)
   
        if x1 + 1 <= len(img) - 1:
            stack.append((x1 + 1, x2))
            if mask[(x1 + 1, x2)] == 1: 
                stack.pop(-1)
            else: 
                diff_r = abs(r_dict[(x1 + 1, x2)] - r_val)
                diff_g = abs(g_dict[(x1 + 1, x2)] - g_val)
                diff_b = abs(b_dict[(x1 + 1, x2)] - b_val)
                ave_r = ((r_dict[(x1 + 1, x2)] + r_val) / 2)
                dist = math.sqrt((2 + (ave_r / 256)) * (diff_r ** 2) + (4 * diff_g ** 2) + ((2 + (255 - ave_r) / 256) * diff_b ** 2))
                if dist <= thres: 
                    mask[(x1 + 1, x2)] = 1 
                    
                else: 
                    stack.pop(-1)
                    
                
        if x2 - 1 >= 0:
            stack.append((x1, x2 - 1))
            if mask[(x1, x2 - 1)] == 1: 
                stack.pop(-1)
            else: 
                diff_r = abs(r_dict[(x1, x2 - 1)] - r_val)
                diff_g = abs(g_dict[(x1, x2 - 1)] - g_val)
                diff_b = abs(b_dict[(x1, x2 - 1)] - b_val)
                ave_r = ((r_dict[(x1, x2 - 1)] + r_val) / 2)
                dist = math.sqrt((2 + (ave_r / 256)) * (diff_r ** 2) + (4 * diff_g ** 2) + ((2 + (255 - ave_r) / 256) * diff_b ** 2))
                if dist <= thres: 
                    mask[(x1, x2 - 1)] = 1 
                    
                else:
                    stack.pop(-1)
                    
        
        if x2 + 1 <= len(img[0]) - 1:
            stack.append((x1, x2 + 1))
            if mask[(x1, x2 + 1)] == 1:
                stack.pop(-1)
            else: 
                diff_r = abs(r_dict[(x1, x2 + 1)] - r_val)
                diff_g = abs(g_dict[(x1, x2 + 1)] - g_val)
                diff_b = abs(b_dict[(x1, x2 + 1)] - b_val)
                ave_r = ((r_dict[(x1, x2 + 1)] + r_val) / 2)
                dist = math.sqrt((2 + (ave_r / 256)) * (diff_r ** 2) + (4 * diff_g ** 2) + ((2 + (255 - ave_r) / 256) * diff_b ** 2))
                if dist <= thres: 
                    mask[(x1, x2 + 1)] = 1 
                    
                else:
                    stack.pop(-1)
        
    return mask

def compute_edge(mask):           
    rsize, csize = len(mask), len(mask[0]) 
    edge = np.zeros((rsize,csize))
    if np.all((mask == 1)): return edge        
    for r in range(rsize):
        for c in range(csize):
            if mask[r][c]!=0:
                if r==0 or c==0 or r==len(mask)-1 or c==len(mask[0])-1:
                    edge[r][c]=1
                    continue
                
                is_edge = False                
                for var in [(-1,0),(0,-1),(0,1),(1,0)]:
                    r_temp = r+var[0]
                    c_temp = c+var[1]
                    if 0<=r_temp<rsize and 0<=c_temp<csize:
                        if mask[r_temp][c_temp] == 0:
                            is_edge = True
                            break
    
                if is_edge == True:
                    edge[r][c]=1
            
    return edge

def save_image(filename, image):
    img = image.astype(np.uint8)
    mpimg.imsave(filename,img)

def load_image(filename):
    img = mpimg.imread(filename)
    if len(img[0][0])==4: # if png file
        img = np.delete(img, 3, 2)
    if type(img[0][0][0])==np.float32:  # if stored as float in [0,..,1] instead of integers in [0,..,255]
        img = img*255
        img = img.astype(np.uint8)
    mask = np.ones((len(img),len(img[0]))) # create a mask full of "1" of the same size of the laoded image
    img = img.astype(np.int32)
    return img, mask

def display_image(image, mask):
    # if using Spyder, please go to "Tools -> Preferences -> IPython console -> Graphics -> Graphics Backend" and select "inline"
    tmp_img = image.copy()
    edge = compute_edge(mask)
    for r in range(len(image)):
        for c in range(len(image[0])):
            if edge[r][c] == 1:
                tmp_img[r][c][0]=255
                tmp_img[r][c][1]=0
                tmp_img[r][c][2]=0
 
    plt.imshow(tmp_img)
    plt.axis('off')
    plt.show()
    print("Image size is",str(len(image)),"x",str(len(image[0])))

### if you're reading this, i am aware the below functions are dumb and unnecessary but sir, i am but a one man show ###
# my (?) friend (?) caleb calls me stupid :^( #

def image_to_rgb(img): # converts array to sep list of r, g, b 
    r_value = []
    g_value = []
    b_value = []
    lst = np.array(img).tolist()
    for row in range(0, len(lst)):
        for pixel in range(0, len(lst[row])): 
            r_value.append(lst[row][pixel][0])
        for pixel in range(0, len(lst[row])):
            g_value.append(lst[row][pixel][1])
        for pixel in range(0, len(lst[row])):
            b_value.append(lst[row][pixel][2])   
    return r_value, g_value, b_value

def rgb_to_image(r, g, b, img): # converts sep list of r, g, b to array
    num_of_pixels = len(r) 

    lst = np.array(img).tolist()
    num_of_cols = len(lst[0])
    num_of_rows = len(lst)

    pixel_dict = {}
    for num in range(num_of_pixels):
        rgb_per_pixel = []
        rgb_per_pixel.append(r[num])
        rgb_per_pixel.append(g[num])
        rgb_per_pixel.append(b[num])
        pixel_dict['pixel' + str(num)] = rgb_per_pixel

    new_array = []
    for key in pixel_dict:
        new_array.append(pixel_dict[key])
    arrayed = np.array(new_array) 
    reshaped = np.reshape(arrayed, (num_of_rows, num_of_cols, 3))
    img = reshaped

    return img


def menu():
    def all_options(user_image, mask): # no thoughts head empty

        print(
'\n\
What do you want to do?\n\
e - exit\n\
l - load a picture\n\
s - save the current picture\n\
1 - adjust brightness\n\
2 - adjust contrast\n\
3 - apply greyscale\n\
4 - apply blur\n\
5 - edge detection\n\
6 - embossed\n\
7 - rectangle select\n\
8 - magic wand select\n\
        ')
        user_choice = input('Your choice: ')
        if user_choice == 'l': # load 
            filename = input('\n\
Do note that your existing loaded image will be overwritten. \n\
Please enter filename of your image, including extension. \n\
Example: \'sample_pic.png\' \n\
To exit, enter \'e\'. \n\
\n\
Enter here: ')
            while filename != 'e':
                try:
                    user_image, mask = load_image(filename) # user_image is rgb tuple
                    print()
                    display_image(user_image, mask)
                    return all_options(user_image, mask)
                except: 
                    filename = input('\n\
Error. File does not exist, please try again. \n\
Please enter filename of your image, including extension. \n\
Example: \'sample_pic.png\' \n\
To exit, enter \'e\' \n\
\n\
Enter here: ')
            else:
                return all_options(user_image, mask)

        elif user_choice == 's': # save
            new_name = input(' \n\
Please enter the filename you would like to save the image as, including extension. \n\
Example: \'new_pic.jpg\' \n\
To exit, enter \'e\'. \n\
\n\
Enter here: ')
            while new_name != 'e':
                try:
                    save_image(new_name, user_image) 
                    display_image(user_image, mask)
                    return all_options(user_image, mask)
                except ValueError:
                    new_name = input(' \n\
Error, invalid name. Please try again. \n\
Please enter the filename you would like to save the image as, including extension. \n\
Example: \'new_pic.jpg\' \n\
To exit, enter \'e\'. \n\
\n\
Enter here: ')
            else:
                return all_options(user_image, mask)

        elif user_choice == '1': # adjust brightness
            up_or_down = input(' \n\
Would you like to increase or reduce brightness of the image? \n\
(Enter \'i\' for increase, and \'r\' for reduce.) \n\
\n\
Your choice: ')
            while up_or_down != 'i' and up_or_down != 'r':
                if up_or_down == 'e':
                    return all_options(user_image, mask)
                else:
                    up_or_down = input('\n\
Error, please try again. \n\
Would you like to increase or reduce brightness of the image? \n\
(Enter \'i\' for increase, and \'r\' for reduce.) \n\
\n\
Your choice: ') 
            else:
                input_value = input('\n\
Please enter an integer value for the intensity to increase or decrease. \n\
Example: \'24\' \n\
\n\
Enter here: ') 
                while input_value != 'e':
                    if input_value.isdigit() == False or int(input_value) < 0 or int(input_value) > 255:
                        input_value = input('\n\
Invalid entry, please try again.\n\
Please enter an integer value for the intensity to increase or decrease. \n\
Example: \'24\' \n\
\n\
Enter here: ') 
                    else:
                        if input_value == 'e':
                            return all_options(user_image, mask)
                        else:
                            start = time.time()
                            value = int(input_value)
                            if up_or_down == 'r':
                                value = -value
                                output_img = change_brightness(user_image, value)
                                base = np.where(user_image != 0, 0, user_image) 
                                for row in range(len(user_image)):
                                    for col in range(len(user_image[0])):
                                        if mask[row, col] == 0: 
                                            base[row, col] = user_image[row, col] 
                                        else: 
                                            base[row, col] = output_img[row, col]
                                output_img = base
                                end = time.time()
                                print()
                                print('Time taken:', str((end - start)), 'sec')
                                print()
                                display_image(output_img, mask)
                                return all_options(output_img, mask) 
                            else:
                                output_img = change_brightness(user_image, value)
                                base = np.where(user_image != 0, 0, user_image) 
                                for row in range(len(user_image)):
                                    for col in range(len(user_image[0])):
                                        if mask[row, col] == 0: 
                                            base[row, col] = user_image[row, col] 
                                        else: 
                                            base[row, col] = output_img[row, col]
                                output_img = base
                                end = time.time()
                                print()
                                print('Time taken:', str((end - start)), 'sec')
                                print()
                                display_image(output_img, mask)
                                return all_options(output_img, mask)
                else:
                    return all_options(user_image, mask)

        elif user_choice == '2': # adjust contrast
            corr_value = input('\n\
Please enter factor value (between -255 to 255). \n\
Example: \'125\' \n\
To exit, enter \'e\'. \n\
\n\
Enter here: ') 
            while corr_value != 'e':
                if corr_value.lstrip('-').isdigit() == False or int(corr_value) < -255 or int(corr_value) > 255:
                    corr_value = input('\n\
Invalid input. Please try again. \n\
Please enter factor value (between -255 to 255). \n\
Example: \'125\' \n\
To exit, enter \'e\'. \n\
\n\
Enter here: ') 
                else:
                    start = time.time()
                    corr_value = int(corr_value)
                    output_img = change_contrast(user_image, corr_value)
                    base = np.where(user_image != 0, 0, user_image) 
                    for row in range(len(user_image)):
                        for col in range(len(user_image[0])):
                            if mask[row, col] == 0: 
                                base[row, col] = user_image[row, col] 
                            else: 
                                base[row, col] = output_img[row, col]
                    output_img = base
                    end = time.time()
                    print()
                    print('Time taken:', str((end - start)/60), 'sec')
                    print()
                    display_image(output_img, mask)
                    return all_options(output_img, mask) 
            else:
                return all_options(user_image, mask)
        
        elif user_choice == '3': # greyscale # sorry i spell it with an e.........
            start = time.time()
            output_img = grayscale(user_image)
            base = np.where(user_image != 0, 0, user_image) 
            for row in range(len(user_image)):
                for col in range(len(user_image[0])):
                    if mask[row, col] == 0: 
                        base[row, col] = user_image[row, col] 
                    else: 
                        base[row, col] = output_img[row, col]
            output_img = base
            end = time.time()
            print()
            print('Time taken:', str((end - start)), 'sec')
            print()
            display_image(output_img, mask)
            return all_options(output_img, mask)

        elif user_choice == '4': # blur
            start = time.time()
            output_img = blur_effect(user_image)
            base = np.where(user_image != 0, 0, user_image) 
            for row in range(len(user_image)):
                for col in range(len(user_image[0])):
                    if mask[row, col] == 0: 
                        base[row, col] = user_image[row, col] 
                    else: 
                        base[row, col] = output_img[row, col]
            output_img = base
            end = time.time()
            print()
            print('Time taken:', str((end - start)), 'sec')
            print()
            display_image(output_img, mask)
            return all_options(output_img, mask)
        
        elif user_choice == '5': # edge
            start = time.time()
            output_img = edge_detection(user_image)
            base = np.where(user_image != 0, 0, user_image) 
            for row in range(len(user_image)):
                for col in range(len(user_image[0])):
                    if mask[row, col] == 0: 
                        base[row, col] = user_image[row, col] 
                    else: 
                        base[row, col] = output_img[row, col]
            output_img = base
            end = time.time()
            print()
            print('Time taken:', str((end - start)), 'sec')
            print()
            display_image(output_img, mask)
            return all_options(output_img, mask)
        
        elif user_choice == '6': # emboss
            output_img = embossed(user_image)
            base = np.where(user_image != 0, 0, user_image) 
            for row in range(len(user_image)):
                for col in range(len(user_image[0])):
                    if mask[row, col] == 0: 
                        base[row, col] = user_image[row, col] 
                    else: 
                        base[row, col] = output_img[row, col]
            start = time.time()
            output_img = base
            end = time.time()
            print()
            print('Time taken:', str((end - start)), 'sec')
            print()
            display_image(output_img, mask)
            return all_options(output_img, mask)
        
        elif user_choice == '7': # rect select
            print('\n\
Please enter top left pixel position (x) and bottom right pixel position (y) to select. \n\
Example: \n\
x-row: 3 \n\
x-column: 4 \n\
y-row: 100 \n\
y_column: 50 \n\
To exit, enter \'e\'. \n\
')
            x1 = input('x-row: ')
            while x1 != 'e':
                if x1.isdigit() == False or int(x1) < 0 or int(x1) > len(user_image) - 1:
                    print('Invalid input. Please try again.')
                    x1 = input('x-row: ')
                else: 
                    break
            else: 
                return all_options(user_image, mask)

            x2 = input('x-column: ')
            while x2 != 'e':
                if x2.isdigit() == False or int(x2) < 0 or int(x2) > len(user_image[0]) - 1:
                    print('Invalid input. Please try again.')
                    x2 = input('x-column: ')
                else:
                    break
            else:
                return all_options(user_image, mask)

            y1 = input('y-row: ')
            while y1 != 'e':
                if y1.isdigit() == False or int(y1) < int(x1) or int(y1) > len(user_image) - 1:
                    print('Invalid input. Please try again.')
                    y1 = input('y-row: ')
                else:
                    break
            else:
                return all_options(user_image, mask)

            y2 = input('y-column: ')
            while y2 != 'e':
                if y2.isdigit() == False or int(y2) < int(x2) or int(y2) > len(user_image[0]) - 1:
                    print('Invalid input. Please try again.')
                    y2 = input('y-column: ')
                else:
                    break
            else:
                return all_options(user_image, mask)

            x = (int(x1), int(x2))
            y = (int(y1), int(y2))
            mask_copy = mask
            start = time.time()
            new_mask = rectangle_select(mask_copy, x, y)
            end = time.time()
            print()
            print('Time taken:', str((end - start)), 'sec')
            print()
            display_image(user_image, new_mask)
            return all_options(user_image, new_mask)

        elif user_choice == '8': # magic wand
            print('\n\
Please input a position (x) to conduct the magic wand on, as well as the threshold of distance from the input position. \n\
Example: \n\
x-row: 34 \n\
x-column: 58 \n\
Threshold: 200 \n\
To exit, enter \'e\'. \n\
')          
            x1 = input('x-row: ')
            while x1 != 'e':
                if x1.isdigit() == False or int(x1) < 0 or int(x1) > len(user_image):
                    print('Invalid input. Please try again.')
                    x1 = input('x-row: ')
                else:
                    break
            else: 
                return all_options(user_image, mask)

            x2 = input('x-column: ')
            while x2 != 'e':
                if x2.isdigit() == False or int(x2) < 0 or int(x2) > len(user_image[0]):
                    print('Invalid input. Please try again.')
                    x2 = input('x-column: ')
                else:
                    break
            else:
                return all_options(user_image, mask)

            thres = input('Threshold: ')
            while thres != 'e':
                if thres.isdigit() == False or int(thres) < 0:
                    print('Invalid input. Please try again.')
                    thres = input('Threshold: ')
                else:
                    break
            else:
                return all_options(user_image, mask)

            start = time.time()
            new_mask = magic_wand_select(user_image, (int(x1), int(x2)), int(thres))
            end = time.time()
            print()
            print('Time taken:', str((end - start)), 'sec')
            print()
            display_image(user_image, new_mask)

            return all_options(user_image, new_mask)

        elif user_choice == 'e':
            print('\n\
')
            return menu()
        
        else:
            print('\n\
Invalid input. Please choose again.')
            return all_options(user_image, mask)
            
            
##########################################################################################################

    # initialisation menu
    
    print(
'What do you want to do?\n\
me\n\
e - exit\n\
l - load a picture\n\
    ')

    user_choice = input('Your choice: ')
    while user_choice != 'e' and user_choice != 'l':
        user_choice = input('Invalid input, please try again. \n\
Your choice: ')
    else: 
        if user_choice == 'e':
            pass # what else can i do here, reboot the menu???
        elif user_choice == 'l':
            filename = input('\n\
Please enter filename of your image, including extension. \n\
Example: \'sample_pic.png\' \n\
To exit, enter \'e\' \n\
\n\
Enter here: ')
            while filename != 'e':
                try:
                    user_image, mask = load_image(filename) 
                    print()
                    display_image(user_image, mask)
                    return all_options(user_image, mask)

                except: 
                    filename = input('\n\
File does not exist, please try again. \n\
Please enter filename of your image, including extension. \n\
Example: \'sample_pic.png\' \n\
To exit, enter \'e\' \n\
\n\
Enter here: ')
            else: 
                if filename == 'e':
                    return menu() 
                elif filename != 'e':
                    filename = input('Invalid choice, please try again. \n\
Please enter filename of your image, including extension. \n\
Example: \'sample_pic.png\' \n\
To exit, enter \'e\' \n\
\n\
Enter here: ')  
        else:
            return menu()

# i used to think i wasn't doing too bad in terms of intellect but then there's 900+ lines in this file so please go easy on me      

if __name__ == "__main__":
    menu()
