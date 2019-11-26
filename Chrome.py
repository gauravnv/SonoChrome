# We will use this file for image creation
import random, math
from random import randint
from PIL import Image, ImageFilter
from statistics import mean
from numpy import random, array_split, array

random.seed()
#
# Z = random.random((50, 50))  # Test data
#
# imshow(Z, cmap=get_cmap("Emotion"), interpolation='nearest')
# show()

emotion_dict = {
    'love': 1,
    'acceptance': 2,
    'trust': 3,
    'admiration': 4,
    'fear': 5,
    'awe': 6,
    'sadness': 7,
    'remorse': 8,
    'boredom': 9,
    'annoyance': 10,
    'aggressiveness': 11,
    'interest': 12,
    'anticipation': 13,
    'serenity': 14,
    'joy': 15,
    'ecstasy': 16,
    'optimism': 17,
    'surprise': 18
}
#
# colours = []
# for i in range(31):
#     colours.append('#%06X' % randint(0, 0xFFFFFF))


class X:
    def eval(self, x, y):
        return x

    def __str__(self):
        return "x"


class Y:
    def eval(self, x, y):
        return y

    def __str__(self):
        return "y"


class SinPi:
    def __init__(self, prob):
        self.arg = build_expression(prob * prob)

    def __str__(self):
        return "sin(pi*" + str(self.arg) + ")"

    def eval(self, x, y):
        return math.sin(math.pi * self.arg.eval(x, y))


class CosPi:
    def __init__(self, prob):
        self.arg = build_expression(prob * prob)

    def __str__(self):
        return "cos(pi*" + str(self.arg) + ")"

    def eval(self, x, y):
        return math.cos(math.pi * self.arg.eval(x, y))


class Times:
    def __init__(self, prob):
        self.lhs = build_expression(prob * prob)
        self.rhs = build_expression(prob * prob)

    def __str__(self):
        return str(self.lhs) + "*" + str(self.rhs)

    def eval(self, x, y):
        return self.lhs.eval(x, y) * self.rhs.eval(x, y)


def build_expression(prob=0.99):
    if random.random() < prob:
        return random.choice([SinPi, CosPi, Times])(prob)
    else:
        return random.choice([X, Y])()


def plot_intensity(emotions, exp, pixels_per_unit=150):
    canvas_width = 2 * pixels_per_unit + 1
    canvas = Image.new("L", (canvas_width, canvas_width))

    for py in range(canvas_width):
        for px in range(canvas_width):
            # Convert pixel location to [-1,1] coordinates
            x = float(px - pixels_per_unit) / pixels_per_unit
            y = -float(py - pixels_per_unit) / pixels_per_unit
            z = exp.eval(x, y)

            # Scale [-1,1] result to [0,255] by taking average over the given emotion labels
            intensities = []
            for index in range(len(emotions)):
                # Improve this formula!
                result = int(z * emotions[index] + 118.5*z + 118.5)
                intensities.append(result)

            intensity = mean(intensities)
            xy = (px, py)
            canvas.putpixel(xy, int(intensity))

    return canvas


def plot_color(emotions, red_exp, green_exp, blue_exp, pixels_per_unit=150):
    palettes = array_split(array(emotions), 3)

    red_plane = plot_intensity(palettes[0].tolist(), red_exp, pixels_per_unit)
    green_plane = plot_intensity(palettes[1].tolist(), green_exp, pixels_per_unit)
    blue_plane = plot_intensity(palettes[2].tolist(), blue_exp, pixels_per_unit)
    return Image.merge("RGB", (red_plane, green_plane, blue_plane))


def get_emotions_from_labels(labels):
    emotions = []
    for i in range(len(labels)):
        emotion = list(emotion_dict.keys())[list(emotion_dict.values()).index(labels[i])]
        emotions.append(emotion)

    return emotions


def build_image(labels, num_pics=20):
    emotions = get_emotions_from_labels(labels)
    print("The emotions found in the song are: ", emotions)
    with open("eqns.txt", 'w') as eqnsFile:
        for i in range(num_pics):
            red_exp = build_expression()
            green_exp = build_expression()
            blue_exp = build_expression()

            eqnsFile.write("img" + str(i) + ":\n")
            eqnsFile.write("red = " + str(red_exp) + "\n")
            eqnsFile.write("green = " + str(green_exp) + "\n")
            eqnsFile.write("blue = " + str(blue_exp) + "\n\n")

            image = plot_color(labels, red_exp, green_exp, blue_exp)
            image.save("Images/img" + str(i) + ".png", "PNG")
