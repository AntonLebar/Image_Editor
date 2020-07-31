# for image processing and reading/writing
from PIL import Image
# For matrix operations
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
# for cmd line arguments
import sys

# for plotting the image
import matplotlib as mpl
import matplotlib.pyplot as plt

# Define Colour Conversion, and Drawing Functions
def RGB_HSL(R, G, B):
    # Convert RGB matrixes to HSL and return HSL

    # calculation variables
    r_prime = R / 255
    g_prime = G / 255
    b_prime = B / 255
    C_max_1 = np.maximum(r_prime, np.maximum(g_prime, b_prime))
    C_min_1 = np.minimum(r_prime, np.minimum(g_prime, b_prime))
    Delta = C_max_1 - C_min_1

    # intialize Hue
    Hue_r = np.select([Delta != 0], [60 * (((g_prime - b_prime) / Delta) % 6)])
    Hue_g = np.select([Delta != 0], [60 * (((b_prime - r_prime) / Delta) + 2)])
    Hue_b = np.select([Delta != 0], [60 * (((r_prime - g_prime) / Delta) + 4)])
    condlist = [C_max_1 == r_prime, C_max_1 == g_prime, C_max_1 == b_prime]
    choicelist = [Hue_r, Hue_g, Hue_b]
    Hue = np.select(condlist, choicelist)

    # lightness
    Lightness = (C_max_1 + C_min_1) / 2

    # Saturation (For HSL)
    Saturation_L = np.select([Delta != 0], [Delta / (1 - abs(2 * Lightness - 1))])

    return Hue, Saturation_L, Lightness

def RGB_HSV(R, G, B):
    # Convert RGB matrixes to HSV and return HSV

    # calculation variables
    r_prime = R / 255
    g_prime = G / 255
    b_prime = B / 255
    C_max_1 = np.maximum(r_prime, np.maximum(g_prime, b_prime))
    C_min_1 = np.minimum(r_prime, np.minimum(g_prime, b_prime))
    Delta = C_max_1 - C_min_1

    # intialize Hue
    Hue_r = np.select([Delta != 0], [60 * (((g_prime - b_prime) / Delta) % 6)])
    Hue_g = np.select([Delta != 0], [60 * (((b_prime - r_prime) / Delta) + 2)])
    Hue_b = np.select([Delta != 0], [60 * (((r_prime - g_prime) / Delta) + 4)])
    condlist = [C_max_1 == r_prime, C_max_1 == g_prime, C_max_1 == b_prime]
    choicelist = [Hue_r, Hue_g, Hue_b]
    Hue = np.select(condlist, choicelist)

    # Saturation (For HSV)
    Saturation_V = np.select([C_max_1 != 0], [(Delta) / (C_max_1)])

    # Value
    Value = C_max_1

    return Hue, Saturation_V, Value

def HSV_to_RGB(H, S, V):
    # Convert HSV matrixes to RGB and return RGB

    C = V * S
    X = C * (1 - abs((H / 60) % 2 - 1))
    m = V - C

    Cond = np.array([(H >= 0) & (H < 60), (H >= 60) & (H < 120), (H >= 120) & (H < 180), (H >= 180) & (H < 240),
                     (H >= 240) & (H < 300), (H >= 300) & (H < 360)])
    R = np.floor((np.select(Cond, [C, X, 0, 0, X, C]) + m) * 255)
    G = np.floor((np.select(Cond, [X, C, C, X, 0, 0]) + m) * 255)
    B = np.floor((np.select(Cond, [0, 0, X, C, C, X]) + m) * 255)

    return R, G, B

def HSL_to_RGB(H, S, L):
    # Convert HSL matrixes to RGB and return RGB

    C = (1 - abs(2 * L - 1)) * S
    X = C * (1 - abs((H / 60) % 2 - 1))
    m = L - C / 2

    Cond = np.array([(H >= 0) & (H < 60), (H >= 60) & (H < 120), (H >= 120) & (H < 180), (H >= 180) & (H < 240),
                     (H >= 240) & (H < 300), (H >= 300) & (H < 360)])
    R = np.floor((np.select(Cond, [C, X, 0, 0, X, C]) + m) * 255)
    G = np.floor((np.select(Cond, [X, C, C, X, 0, 0]) + m) * 255)
    B = np.floor((np.select(Cond, [0, 0, X, C, C, X]) + m) * 255)

    return R, G, B

def drawing_image(image):
    # updates matplotlib figure
    # these commands are reqired to keep the figure from pausing the script
    ax.imshow(image, aspect='auto')
    plt.ion()
    plt.draw()
    plt.pause(0.001)

# Image grayscale, blurring, rotating, etc functions
def greyscale(im_matrix):
    temp_image = np.empty_like(im_matrix)
    r_grey = im_matrix[:, :, 0]
    g_grey = im_matrix[:, :, 1]
    b_grey = im_matrix[:, :, 2]

    temp = (0.3 * r_grey) + (0.59 * g_grey) + (0.11 * b_grey)

    temp_image[:, :, 0] = temp
    temp_image[:, :, 1] = temp
    temp_image[:, :, 2] = temp
    return temp_image

def image_blur(matrix):
    # return blurred matrix based on user input
    temp_image = np.empty_like(matrix)

    # make count matrix for averaging
    blur_dict_count = {}
    for a in range(9):
        blur_dict_count[a] = np.ones_like(matrix[:, :, 0])
    index = 0
    for i in range(3):
        for j in range(3):
            blur_dict_count[index] = np.pad(blur_dict_count[index], ((i, 2 - i), (j, 2 - j)))
            # print(blur_dict_count[index][0:10,0:10])
            index = index + 1

    count = np.zeros_like(blur_dict_count[0])
    for b in range(9):
        count = count + blur_dict_count[b]

    count_1 = count
    # final count matrix for proper division
    count = count[1:-1, 1:-1]

    # determine average for each channel
    for color in range(3):
        # print(color)
        color_matrix = matrix[:, :, color]
        blur_dict_colour = {}
        for a in range(9):
            blur_dict_colour[a] = color_matrix

        index = 0
        for i in range(3):
            for j in range(3):
                blur_dict_colour[index] = np.pad(blur_dict_colour[index], ((i, 2 - i), (j, 2 - j)))
                index = index + 1

        sum_color = np.zeros_like(blur_dict_colour[0])
        for c in range(9):
            # print(blur_dict_colour[c][0:10,0:10], "\n\n")
            sum_color = sum_color + (blur_dict_colour[c] / count_1)
            # print(sum_color[0:10,0:10], "\n\n")

        sum_color_final = sum_color[1:-1, 1:-1]
        average_color = sum_color_final  # / count

        # print(sum_color_final[0:10,0:10])
        # print(count[0:10,0:10])
        temp_image[:, :, color] = average_color.astype(int)

    return temp_image

    # 4 is original matrix, padded with zeroes on all sides
    # 0 1 2
    # 3 4 5
    # 6 7 8

def rotate(matrix):
    # return rotated matrix based on user input
    direction = input("Direction Left or right?: ").upper()
    if direction[0] == "L":
        direct = (0, 1)
    elif direction[0] == "R":
        direct = (1, 0)
    else:
        exit(1)

    try:
        rotations = input("Number of rotations?: ")
        rotations = abs(int(rotations))

    except:
        print("Use an integer")
        rotations = 0

    shape = matrix.shape
    if len(shape) == 3:
        temp_rotate = np.rot90(matrix, rotations, direct)
    if len(shape) == 2:
        temp_rotate = np.rot90(matrix, rotations, direct)
    return temp_rotate

# Main script starts here
arguments = sys.argv
if len(arguments) != 2:
    print("CMD line argument needed,include file name")
    exit(1)
else:
    image_name = arguments[1]
im = Image.open(image_name)

# initialize changes to dictionary storing changes
changes_RGB = {"R": 1, "G": 1, "B": 1}
changes_HSV = {"H": 0, "S": 1, "V": 1}
changes_HSL = {"H": 0, "S": 1, "L": 1}

# Get R,G,B Matrixes
data = np.array(im)
r_matrix = data[:, :, 0]
g_matrix = data[:, :, 1]
b_matrix = data[:, :, 2]

# initialize "preview" Window
fig = plt.figure(frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
drawing_image(data)

# intialize HSV, and HSL image matrices
H_HSV, S_HSV, V_HSV = RGB_HSV(r_matrix, g_matrix, b_matrix)
H_HSL, S_HSL, L_HSL = RGB_HSL(r_matrix, g_matrix, b_matrix)

# Default change to RGB, channel R
current_change = "RGB"
changing = "R"

# Initialize Temporary Matrices for each of the channels (Default to original image)
img_temp = data
img_temp_HSV_tmp = data
img_temp_HSL_tmp = data

# starting command line interface loops
while True:
    print("Currently Changing:", current_change)
    if input("Done making changes?: ") == "yes":
        break

    # Modify image using RGB channels
    while current_change == "RGB":
        # Change Channel
        if changing == "R":
            print("\n(R:", changes_RGB["R"], ") G:", changes_RGB["G"], " B:", changes_RGB["B"], sep='')
            try:
                changes_RGB["R"] = abs(float(input("Change R to:")))
            except:
                print("Use a valid number")
        if changing == "G":
            print("\nR:", changes_RGB["R"], " (G:", changes_RGB["G"], ") B:", changes_RGB["B"], sep='')

            try:
                changes_RGB["G"] = abs(float(input("Change G to:")))
            except:
                print("Use a valid number")

        if changing == "B":
            print("\nR:", changes_RGB["R"], " G:", changes_RGB["G"], " (B:", changes_RGB["B"], ")", sep='')

            try:
                changes_RGB["B"] = abs(float(input("Change B to:")))
            except:
                print("Use a valid number")

        # Update each of the temporary channels
        R_temp = np.minimum(255, r_matrix.astype(int) * changes_RGB["R"])
        G_temp = np.minimum(255, g_matrix.astype(int) * changes_RGB["G"])
        B_temp = np.minimum(255, b_matrix.astype(int) * changes_RGB["B"])
        img_temp[:, :, 0] = R_temp
        img_temp[:, :, 1] = G_temp
        img_temp[:, :, 2] = B_temp

        # draw updated image
        drawing_image(img_temp)

        # give user change to change "channel" or "channel type"        
        tmp = input("change colour stream (back to change colour style): ")
        if tmp.upper() == "R" or tmp.upper() == "G" or tmp.upper() == "B":
            changing = tmp.upper()
        elif tmp.upper() == "BACK":
            tmp_2 = input("new style 'HSV' or 'HSL': ")
            if tmp_2.upper() == "HSV" or tmp_2.upper() == "HSL":
                current_change = tmp_2.upper()
                changing = "H"
            break
    print("Currently Changing:", current_change)
    if input("Done making changes?: ") == "yes":
        break

    # Modify image using HSV channels
    while current_change == "HSV":
        change = False
        if changing == "H":
            print("\n(H:", changes_HSV["H"], ") S:", changes_HSV["S"], " V:", changes_HSV["V"], sep='')
            try:
                tmp = changes_HSV["H"]
                changes_HSV["H"] = abs(float(input("Change H to: ")))
                if tmp != changes_HSV["H"]:
                    change = True
            except:
                print("Use a valid number")
        if changing == "S":
            print("\nH:", changes_HSV["H"], " (S:", changes_HSV["S"], ") V:", changes_HSV["V"], sep='')
            try:
                tmp = changes_HSV["S"]
                changes_HSV["S"] = abs(float(input("Change S to: ")))
                if tmp != changes_HSV["S"]:
                    change = True
            except:
                print("Use a valid number")

        if changing == "V":
            print("\nH:", changes_HSV["H"], " S:", changes_HSV["S"], " (V:", changes_HSV["V"], ")", sep='')
            try:
                tmp = changes_HSV["V"]
                changes_HSV["V"] = abs(float(input("Change V to: ")))
                if tmp != changes_HSV["V"]:
                    change = True
            except:
                print("Use a valid number")

        # Update each of the temporary channels,
        # if it has changed from the last instance
        if change == True:
            H_HSV_tmp = (H_HSV + changes_HSV["H"]) % 360
            S_HSV_tmp = np.minimum(S_HSV * changes_HSV["S"], 1)
            V_HSV_tmp = np.minimum(V_HSV * changes_HSV["V"], 1)

            R_HSV_tmp, G_HSV_tmp, B_HSV_tmp = HSV_to_RGB(H_HSV_tmp, S_HSV_tmp, V_HSV_tmp)

            R_HSV_tmp = R_HSV_tmp.astype(int)
            G_HSV_tmp = G_HSV_tmp.astype(int)
            B_HSV_tmp = B_HSV_tmp.astype(int)

            img_temp_HSV_tmp[:, :, 0] = R_HSV_tmp
            img_temp_HSV_tmp[:, :, 1] = G_HSV_tmp
            img_temp_HSV_tmp[:, :, 2] = B_HSV_tmp

            # draw updated image
            drawing_image(img_temp_HSV_tmp)

        # give user change to change "channel" or "channel type" 
        tmp = input("change HSV stream (back to change colour style): ")
        if tmp.upper() == "H" or tmp.upper() == "S" or tmp.upper() == "V":
            changing = tmp.upper()
        elif tmp.upper() == "BACK":
            tmp_2 = input("new style 'RGB' or 'HSL': ")
            if tmp_2.upper() == "RGB":
                current_change = tmp_2.upper()
                changing = "R"
            elif tmp_2.upper() == "HSL":
                current_change = tmp_2.upper()
                changing = "H"
            break

    print("Currently Changing:", current_change)
    if input("Done making changes?: ") == "yes":
        break

    # Modify image using HSL channels
    while current_change == "HSL":
        change = False
        if changing == "H":
            print("\n(H:", changes_HSL["H"], ") S:", changes_HSL["S"], " L:", changes_HSL["L"], sep='')
            try:
                tmp = changes_HSL["H"]
                changes_HSL["H"] = abs(float(input("Change H to: ")))
                if tmp != changes_HSL["H"]:
                    change = True
            except:
                print("Use a valid number")
        if changing == "S":
            print("\nH:", changes_HSL["H"], " (S:", changes_HSL["S"], ") L:", changes_HSL["L"], sep='')
            try:
                tmp = changes_HSL["S"]
                changes_HSL["S"] = abs(float(input("Change S to: ")))
                if tmp != changes_HSL["S"]:
                    change = True
            except:
                print("Use a valid number")

        if changing == "L":
            print("\nH:", changes_HSL["H"], " S:", changes_HSL["S"], " (L:", changes_HSL["L"], ")", sep='')
            try:
                tmp = changes_HSL["L"]
                changes_HSL["L"] = abs(float(input("Change L to: ")))
                if tmp != changes_HSL["L"]:
                    change = True
            except:
                print("Use a valid number")

        # Update each of the temporary channels,
        # if it has changed from the last instance
        if change == True:
            H_HSL_tmp = (H_HSL + changes_HSL["H"]) % 360
            S_HSL_tmp = np.minimum(S_HSL * changes_HSL["S"], 1)
            L_HSL_tmp = np.minimum(L_HSL * changes_HSL["L"], 1)

            R_HSL_tmp, G_HSL_tmp, B_HSL_tmp = HSL_to_RGB(H_HSL_tmp, S_HSL_tmp, L_HSL_tmp)

            R_HSL_tmp = R_HSL_tmp.astype(int)
            G_HSL_tmp = G_HSL_tmp.astype(int)
            B_HSL_tmp = B_HSL_tmp.astype(int)

            img_temp_HSL_tmp[:, :, 0] = R_HSL_tmp
            img_temp_HSL_tmp[:, :, 1] = G_HSL_tmp
            img_temp_HSL_tmp[:, :, 2] = B_HSL_tmp

            # draw updated image
            drawing_image(img_temp_HSL_tmp)

        # give user change to change "channel" or "channel type" 
        tmp = input("change HSL stream (back to change colour style): ")
        if tmp.upper() == "H" or tmp.upper() == "S" or tmp.upper() == "L":
            changing = tmp.upper()
        elif tmp.upper() == "BACK":
            tmp_2 = input("new style 'RGB' or 'HSV': ")
            if tmp_2.upper() == "RGB":
                current_change = tmp_2.upper()
                changing = "R"
            elif tmp_2.upper() == "HSV":
                current_change = tmp_2.upper()
                changing = "H"
            break

# update Actual Image Matrix, after these changes are made

if current_change == "RGB":
    img_final = img_temp

elif current_change == "HSV":
    img_final = img_temp_HSV_tmp

elif current_change == "HSL":
    img_final = img_temp_HSL_tmp

else:
    print("Error Updating true image matrix")
    exit(1)

drawing_image(img_final)

# final updates, ie: rotate, grayscale
while True:
    print("Options are: BW, Blur, Rotate, Done")
    action = input("What would you like to do?: ").upper()

    if action == "BW":
        img_final = greyscale(img_final)
        drawing_image(img_final)

    elif action == "BLUR":
        img_final = image_blur(img_final)
        drawing_image(img_final)

    elif action == "ROTATE":
        img_final = rotate(img_final)
        drawing_image(img_final)

    elif action == "DONE":
        break
    else:
        print("Invalid Input, Please try again")

# file saving stuff
save = input("Save file? (Y or N): ").upper()
# name sanitization
if save[0] == "Y" or save[0] == "1":
    f_name = input("File name :")
    print(f_name[-5:])
    if f_name[-5:-1] == ".jpeg":
        f_name = f_name

    else:
        f_name = f_name.split('.')[0] + ".jpeg"

    # actually save name now
    test = Image.fromarray(img_final)
    test.save(f_name, "JPEG")
    print("File saved as", f_name)

print("Done")
