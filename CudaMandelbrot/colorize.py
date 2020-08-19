from PIL import Image, warnings
import math


def get_color(iterations):
    a = 0.1
    r = int(255 * (0.5 * math.sin(a * iterations) + 0.5))
    g = int(255 * (0.5 * math.sin(a * iterations + 2.094) + 0.5))
    b = int(255 * (0.5 * math.sin(a * iterations + 4.188) + 0.5))
    return r, g, b


def colorize(values, filename):
    warnings.simplefilter('ignore', Image.DecompressionBombWarning)
    img = Image.new('RGB', (len(values[0]), len(values)), color=(255, 0, 0))
    img_data = img.load()

    for y in range(0, img.size[1]):
        for x in range(0, img.size[0]):
            img_data[x, y] = get_color(values[y][x])

    img.save(filename)


if __name__ == "__main__":
    colorize([[0, 0], [1, 1]], "file.png")