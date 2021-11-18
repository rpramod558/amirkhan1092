import time
import numpy as np
import cv2
import os
import sys
from glob import glob
from pathlib import Path
from PIL import Image, ExifTags
import piexif
from pathlib import Path

import FileHandler

from multiprocessing import Pool
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor, as_completed

from polygon import *
import json

# import shadow

# import importlib

# importlib.reload(shadow)

# from mpl_toolkits.mplot3d import Axes3D
# from numpy.random import rand
# from pylab import figure


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_hex(r, g, b):
    return ('#%02x%02x%02x' % (int(r), int(g), int(b))).upper()


def imshow(img, scale_down=1, rect=None, color=(0, 255, 0)):
    if rect is not None:
        x, y, w, h = rect
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color,
                      2)
    if scale_down > 1:
        cv2.imshow(
            "winname",
            cv2.resize(img,
                       None,
                       fx=float(1.0) / scale_down,
                       fy=float(1.0) / scale_down,
                       interpolation=cv2.INTER_CUBIC))
    else:
        cv2.imshow("winname", img)
    cv2.waitKey(0)


def imwrite(out_file_path,
            img,
            min_quality=5,
            max_quality=100,
            max_out_size=10000000,
            exif_bytes=None):
    Path(os.path.dirname(out_file_path)).mkdir(parents=True, exist_ok=True)
    res = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(res)
    quality = max_quality
    for quality in np.linspace(max_quality, min_quality,
                               int((max_quality - min_quality) / 5.0 + 1.0)):
        # print(quality, os.path.getsize(out_file_path))
        # cv2.imwrite(out_file_path, im_pil,
        #             [int(cv2.IMWRITE_JPEG_QUALITY),
        #              round(quality)])
        if exif_bytes is not None:
            im_pil.save(out_file_path,
                        quality=int(quality),
                        subsampling=0,
                        exif=exif_bytes)
        if os.path.getsize(out_file_path) < max_out_size:
            # print(os.path.getsize(out_file_path), max_out_size)
            break
    return quality, os.path.getsize(out_file_path)


def getFilesWithExtensions(dir_path, types):
    files = [str(p) for p in Path(dir_path).glob("**/*") if p.suffix in types]
    return files


def calcCanny(img, sigma=0.33):
    # gray = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(gray, lower, upper)
    # print("l/u",lower, upper)
    # plt.imshow(edged),plt.colorbar(),plt.show()
    return edged


def getEdgeMask(img, sigma=0.95):
    edges = calcCanny(img, sigma)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    erosion = cv2.morphologyEx(closing, cv2.MORPH_ERODE, kernel, iterations=1)

    # plt.imshow(out_img),plt.colorbar(),plt.show()
    return edges


def grabCutMask(img, edge):

    # When using Grabcut the mask image should be:
    #    0 - sure background
    #    1 - sure foreground
    #    2 - unknown

    mask = np.zeros(edge.shape[:2], np.uint8)
    mask[:] = 2
    mask[edge == 255] = 1
    #     plt.imshow(mask),plt.colorbar(),plt.show()
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)

    out_mask = mask.copy()
    out_mask, _, _ = cv2.grabCut(img, out_mask, None, bgdmodel, fgdmodel, 3,
                                 cv2.GC_INIT_WITH_MASK)
    return out_mask


def grabCutRect(img, edge, rect):

    # When using Grabcut the mask image should be:
    #    0 - sure background
    #    1 - sure foreground
    #    2 - unknown

    # mask = np.zeros(edge.shape[:2], np.uint8)
    # mask[:] = 2
    # mask[edge == 255] = 1
    #     plt.imshow(mask),plt.colorbar(),plt.show()
    bgdmodel = np.zeros((1, 65), np.float64)
    fgdmodel = np.zeros((1, 65), np.float64)

    out_mask = np.zeros(img.shape[:2], np.uint8)
    out_mask, _, _ = cv2.grabCut(img, out_mask, rect, bgdmodel, fgdmodel, 3,
                                 cv2.GC_INIT_WITH_RECT)
    # cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    return out_mask


def processImage(img, sigma):
    edges = getEdgeMask(img, sigma)

    rect = cv2.boundingRect(edges)

    # imshow(edges)

    if rect == (0, 0, img.shape[1], img.shape[0]):
        rect = (1, 1, img.shape[1] - 1, img.shape[0] - 1)
    # out_mask_unit = grabCutMask(img, edges)
    out_mask_unit = grabCutRect(img, edges, rect)

    out_mask = np.where((out_mask_unit == 2) | (out_mask_unit == 0), 0,
                        1).astype('uint8')

    # bgcolor = np.average(img[:, :, 0], weights=(out_mask_unit == 1))
    out_img = img * out_mask[:, :, np.newaxis]
    mask_f = out_mask * 255
    # x, y, w, h = cv2.boundingRect(mask_f)
    # cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    bgcolor = np.round(
        (img[:, :, 0][mask_f[:] < 10].mean(), img[:, :, 1][mask_f < 10].mean(),
         img[:, :, 2][mask_f < 10].mean()))

    x, y, w, h = rect
    cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

    return out_img, mask_f, bgcolor[::-1]


# 1. Resolution   in range min_height, max_height, min_width, max_width
def checkResolution(image,
                    min_height=480,
                    max_height=1080,
                    min_width=720,
                    max_width=1400,
                    **kwargs):
    correct = (image.shape[1] <= max_width) and (
        image.shape[1] >= min_width) and (image.shape[0] <= max_height) and (
            image.shape[0] >= min_height)
    return correct


def correctResolution(image, **kwargs):
    return image


# 2. Image size and resolution mean the same thing?? aspect ratio ?
def checkAspectRatio(image, min_aspect=0.5, max_aspect=2.0, **kwargs):
    correct = (float(image.shape[1]) /
               float(image.shape[0])) < max_aspect and (
                   float(image.shape[1]) / float(image.shape[0])) > min_aspect
    return correct


def correctAspectRatio(image, **kwargs):

    return image


def getImageMetatData(image_path, **kwargs):
    img = Image.open(image_path)
    try:
        exif = {
            ExifTags.TAGS[k]: v
            for k, v in img._getexif().items() if k in ExifTags.TAGS
        }
    except:
        exif = None
    return exif


# 3. Dpi should be more than certain value
def checkDpi(image_file, min_dpi=120, max_dpi=600, **kwargs):
    metadata = getImageMetatData(image_file)
    correct = False
    if metadata:
        if 'XResolution' in metadata:
            correct = (metadata['XResolution'][0] < max_dpi) and (
                metadata['XResolution'][0] >
                min_dpi) and (metadata['YResolution'][0] > min_dpi) and (
                    metadata['YResolution'][0] < max_dpi)
    return correct


def checkExactDpi(image_file, out_dpi=96, **kwargs):
    metadata = getImageMetatData(image_file)
    correct = False
    if metadata:
        if 'XResolution' in metadata:
            try:
                correct = (
                    (metadata['XResolution'][0] / metadata['XResolution'][1])
                    == out_dpi) and ((metadata['YResolution'][0] /
                                      metadata['YResolution'][1]) == out_dpi)
            except:
                correct = (metadata['XResolution']
                           == out_dpi) and (metadata['YResolution'] == out_dpi)
    return correct


def copyMissingExif(image_file, out_file_path):
    print("entered ")
    image_file_data = open(image_file, "rb")
    image1 = Image(image_file_data.read())
    print(image1.copyright)
    image_file_data.close()
    image_file_data = open(out_file_path, "rb")
    image2 = Image(image_file_data)
    image_file_data.close()
    print(image2.copyright)

    with open(out_file_path, 'wb') as updated_file:
        image2.copyright = image1.copyright
        print(image1.copyright)
        updated_file.write(image2.get_file())


def correctDpi(image_file, out_file_path, out_dpi, **kwargs):
    im = Image.open(image_file)
    # oim = Image.open(out_file_path)
    exif_dict = piexif.load(im.info["exif"])
    num = exif_dict["0th"][piexif.ImageIFD.XResolution][1]
    try:
        exif_dict["0th"][piexif.ImageIFD.XResolution] = (out_dpi * num,
                                                         1 * num)
        exif_dict["0th"][piexif.ImageIFD.YResolution] = (out_dpi * num,
                                                         1 * num)
    except Exception() as e:
        print(e)
        exif_dict["0th"][piexif.ImageIFD.XResolution] = out_dpi
        exif_dict["0th"][piexif.ImageIFD.YResolution] = out_dpi
    exif_bytes = piexif.dump(exif_dict)

    # return piexif.insert(exif_bytes, out_file_path)
    # oim.save(out_file_path, exif=exif_bytes)
    # copyMissingExif(image_file, out_file_path)
    return exif_bytes


def copyExifData(image_file, out_file_path, **kwargs):
    im = Image.open(image_file)
    # oim = Image.open(out_file_path)
    exif_dict = piexif.load(im.info["exif"])
    exif_bytes = piexif.dump(exif_dict)
    # piexif.transplant(image_file, out_file_path)
    # piexif.insert(exif_bytes, out_file_path)
    # copyMissingExif(image_file, out_file_path)
    return exif_bytes


# 4. Object size (zooming in and out) area coverage


def getEdgeBoundingBox(image, sigma=0.98, **kwargs):

    edges = getEdgeMask(image, sigma)

    rect = cv2.boundingRect(edges)

    # imshow(edges)

    if rect == (0, 0, image.shape[1], image.shape[0]):
        rect = (1, 1, image.shape[1] - 2, image.shape[0] - 2)

    return rect


def getAreaCoverage(image, square_area=True, **kwargs):
    rect = getEdgeBoundingBox(image)

    x, y, w, h = rect

    nom = np.max([w, h])**2 if square_area else w * h

    normalised_area = float(nom) / float((image.shape[0]) * (image.shape[1]))

    return normalised_area


def checkAreaCoverage(image,
                      min_area_coverage=0.6,
                      max_area_coverag=0.8,
                      **kwargs):
    normalised_area = getAreaCoverage(image)
    correct = normalised_area < max_area_coverag and normalised_area > min_area_coverage
    return correct


def correctAreaCoverage(image,
                        output_resolution=None,
                        min_area_coverage=0.6,
                        max_area_coverag=0.6,
                        do_center_object=False,
                        aspectratio=1.0,
                        pararmeters=None,
                        **kwargs):

    rect = getEdgeBoundingBox(image)

    avg_coverage_area = (min_area_coverage + max_area_coverag) / 2.0

    x, y, w, h = rect

    # normalised_area = float((w) * (h)) / float(
    #     (image.shape[0]) * (image.shape[1]))

    # resMatSizeFactor = np.array(
    #     image.shape[:2]) * normalised_area / avg_coverage_area

    # if (resMatSizeFactor[1] / resMatSizeFactor[0]) > aspectratio:
    #     resMatSizeFactor[0] = resMatSizeFactor[1] / aspectratio
    # else:
    #     resMatSizeFactor[1] = resMatSizeFactor[0] * aspectratio

    # resMatSize = output_resolution or resMatSizeFactor

    resMatSize, out_resolution, correction_log = getFinalPararmetersFixed(
        pararmeters, rect, image.shape[:2][::-1])
    resMatSize = np.round(resMatSize[::-1])

    res = image

    # imshow(res, scale_down=4, rect=rect, color=(0, 0, 255))

    center = [image.shape[0] / 2.0, image.shape[1] / 2.0]

    if (do_center_object):
        center = [
            (rect[3] / 2.0 + rect[1]),
            (rect[2] / 2.0 + rect[0]),
        ]

    rect = [
        round(center[1] - resMatSize[1] / 2.0),
        round(center[0] - resMatSize[0] / 2.0), resMatSize[1], resMatSize[0]
    ]
    newMatSize = [
        -rect[1] if rect[1] < 0 else 0,  #// h0
        rect[1] + rect[3] - image.shape[0] if
        (rect[1] + rect[3]) > image.shape[0] else 0,  #// h1
        -rect[0] if rect[0] < 0 else 0,  #// w0
        (rect[0] + rect[2] -
         image.shape[1]) if rect[0] + rect[2] > image.shape[1] else 0  #, // w1
    ]

    # newMatSize = center
    color = getBackgroundColorCorners(image, **pararmeters['background'])

    BORDER_TYPE = cv2.BORDER_CONSTANT  #cv2.BORDER_REPLICATE  #

    # res = np.array(object)

    # imshow(res, scale_down=4, rect=rect, color=(255, 0, 0))
    print(newMatSize)
    res = cv2.copyMakeBorder(image,
                             top=round(newMatSize[0]),
                             bottom=round(newMatSize[1]),
                             left=round(newMatSize[2]),
                             right=round(newMatSize[3]),
                             borderType=BORDER_TYPE,
                             value=color)

    if (res.shape[0] != resMatSize[0] or res.shape[1] != resMatSize[1]):
        rect[0] += newMatSize[2]
        rect[1] += newMatSize[0]

        # imshow(res, scale_down=4, rect=rect, color=(0, 255, 0))

        res = res[int(rect[1]):int(rect[1] + rect[3]),
                  int(rect[0]):int(rect[0] + rect[2])]

        # imshow(res, scale_down=4)

    res = cv2.resize(
        res,
        [out_resolution[0], out_resolution[1]],
        #  fx=scaling_factor,
        #  fy=scaling_factor,
        interpolation=cv2.INTER_CUBIC)
    return res, correction_log

    # out_mask_unit = grabCutMask(img, edges)


# 5. Object should be centred in all images
def checkObjectCentering(image,
                         min_center_range=0.45,
                         max_center_range=0.55,
                         **kwargs):

    rect = getEdgeBoundingBox(image)

    x, y, w, h = rect

    # print("center", (float(w + x) / 2.0), (float(h + y) / 2.0))

    center_factor = (float(w + x) / 2.0) / float(
        image.shape[1]), (float(h + y) / 2.0) / float(image.shape[0])

    correct = center_factor[0] < max_center_range and center_factor[
        0] > min_center_range and center_factor[
            1] < max_center_range and center_factor[1] > min_center_range
    return correct


def correctObjectCentering(image_file,
                           min_area_coverage=0.6,
                           max_area_coverag=1.0,
                           **kwargs):

    im = Image.open(image_file)
    exif_dict = piexif.load(im.info["exif"])
    exif_dict["0th"][piexif.ImageIFD.XResolution] = (width_dpi, 1)
    exif_dict["0th"][piexif.ImageIFD.YResolution] = (height_dpi, 1)
    exif_bytes = piexif.dump(exif_dict)
    return im.save(image_file, "jpeg", exif=exif_bytes)


# 6. Image should not blurred
def checkBlurring(image, blur_threshold=10.0, **kwargs):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    # print('sharpness', sharpness)

    correct = sharpness > blur_threshold
    return correct


def correctBlurring(image_file,
                    min_area_coverage=0.6,
                    max_area_coverag=1.0,
                    **kwargs):

    im = Image.open(image_file)
    exif_dict = piexif.load(im.info["exif"])
    exif_dict["0th"][piexif.ImageIFD.XResolution] = (width_dpi, 1)
    exif_dict["0th"][piexif.ImageIFD.YResolution] = (height_dpi, 1)
    exif_bytes = piexif.dump(exif_dict)
    return im.save(image_file, "jpeg", exif=exif_bytes)


# 7. Image should not contain Hard shadow and lighting
def checkShadow(image, value=0.3, **kwargs):

    # if(V>0.3 && V<0.85 && H<85 && S<0.15)
    # if(V>0.5 && V<0.95 &&  S<0.2)
    # if(V>0.3 && V<0.95 &&  S<0.2)

    lower_hsv = (0, round(0.2 * 255), round(0.3 * 255))
    higher_hsv = (255, 255, round(0.95 * 255))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # imshow(hsv)

    # shadow = cv2.inRange(hsv, lower_hsv, higher_hsv)

    shadow = cv2.adaptiveThreshold(hsv[:, :,
                                       1], 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 401, -10)

    imshow(shadow)

    return True


def correctShadow(image,
                  min_area_coverage=0.6,
                  max_area_coverag=1.0,
                  **kwargs):

    res = shadow.checkShadow(image)

    return res


# 8. Image should not contain Hard shadow and lighting
# def checkHardLighting(image, min_center_range=0.45, max_center_range=0.55):

#     return True

# def correctHardLighting(image_file,
#                         min_area_coverage=0.6,
#                         max_area_coverag=1.0):

#     im = Image.open(image_file)
#     exif_dict = piexif.load(im.info["exif"])
#     exif_dict["0th"][piexif.ImageIFD.XResolution] = (width_dpi, 1)
#     exif_dict["0th"][piexif.ImageIFD.YResolution] = (height_dpi, 1)
#     exif_bytes = piexif.dump(exif_dict)
#     return im.save(image_file, "jpeg", exif=exif_bytes)


def getBackgroundColorCorners(img,
                              sigma=0.98,
                              colors=[[255, 255, 255]],
                              variance=8,
                              **kwargs):
    edges = getEdgeMask(img, sigma)

    rect = cv2.boundingRect(edges)
    bg_colors = [
        img[rect[1] + 1, rect[0] + 1], img[rect[1] + 1, rect[2] - 1],
        img[rect[3] - 1, rect[1] + 1], img[rect[3] - 1, rect[2] - 1, ]
    ]

    selected_colors = []
    for col in bg_colors:
        for d_color in colors:
            if isColorInRange(col, [d_color], variance=variance):
                selected_colors.append(col)
    # colorsb = np.array(bg_colors, dtype=np.uint8)
    # colorsb = colorsb.reshape((2, 2, 3))

    # hsv = cv2.cvtColor(colorsb, cv2.COLOR_BGR2HSV)

    # select = np.median(colors, axis=0)
    # colorsb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if len(selected_colors) < 1:
        selected_colors = bg_colors
    # print("selected_colors", selected_colors)
    color = np.round(np.median(selected_colors, axis=0))

    return color


def getBackgroundColor(image, sigma, **kwargs):

    res, edge, bgcolor = processImage(img, 1.0)

    return bgcolor


# class bcolors:
HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'


def printGreen(*args):
    return
    print(OKGREEN, end="")
    print(*args, end="")
    print(ENDC)


def printRed(*args):
    return
    print(FAIL, end="")
    print(*args, end="")
    print(ENDC)


def printColor(*args, color=OKBLUE):
    return
    print(color, end="")
    print(*args, end="")
    print(ENDC)


CORRECTION_LOG_SEGREGATION = {
    CORRECTION_UNREQUIRED: 'correct',
    CORRECTION_BOUNDS_SATISFIED: 'corrected',
    CORRECTION_BOUNDS_SCALED: 'scaled',
    CORRECTION_BOUNDS_UNSATISFIED: 'non_correct'
}


def isColorInRange(color, colors=[[255, 255, 255]], variance=8, **kwargs):
    color = np.array(color)
    colorInRange = False

    for ref in colors:
        np_ref = np.array(ref)
        colorInRange = colorInRange or (np.greater(color,
                                                   (np_ref - variance)).all()
                                        and np.greater(
                                            (np_ref + variance), color).all())

    return colorInRange


class CustomJSONizer(json.JSONEncoder):
    def default(self, obj):
        return super().encode(bool(obj)) \
            if isinstance(obj, np.bool_) \
            else super().default(obj)


def validateImage(image, image_file, pararmeters):

    results = {'image_file': os.path.basename(image_file)}
    printColor(image_file, ":")

    res = image

    # 1. Resolution should be more than eg (1000,720)

    is_resolution_correct = checkResolution(image, **pararmeters['resolution'])

    results['resolution'] = is_resolution_correct

    if is_resolution_correct:
        printGreen("1: Resolution in Correct Range")
    else:
        printRed("1: Resolution is out of Range")

    # 2. Image size and resolution mean the same thing??
    is_aspect_correct = checkAspectRatio(image, **pararmeters['aspectratio'])
    results['aspectratio'] = is_aspect_correct
    if is_aspect_correct:
        printGreen("2: AspectRatio in Correct Range")
    else:
        printRed("2: AspectRatio is out of Range")

    # 4. Object size (zooming in and out) area coverage
    is_area_correct = checkAreaCoverage(image, **pararmeters['areacoverage'])
    results['areacoverage'] = is_area_correct
    if is_area_correct:
        printGreen("4: AreaCoverage in Correct Range")
    else:
        printRed("4: AreaCoverage is out of Range")

    # 5. Object should be centred in all images
    is_object_centered = checkObjectCentering(image,
                                              **pararmeters['objectcentering'])
    results['objectcentering'] = is_object_centered
    if is_object_centered:
        printGreen("5: ObjectCentering in Correct Range")
    else:
        printRed("5: ObjectCentering is out of Range")

    correction_log = CORRECTION_UNREQUIRED

    if not (is_area_correct and is_aspect_correct and is_area_correct
            and is_object_centered):
        # area coverage / centering / aspect / resolution correction in same function
        res, correction_log = correctAreaCoverage(
            image,
            do_center_object=not is_object_centered,
            aspectratio=1.0,
            pararmeters=pararmeters,
            **pararmeters['areacoverage'],
        )
        results['correction_log'] = CORRECTION_LOG_SEGREGATION[correction_log]

    # 6. Image should not blurred
    if checkBlurring(image, **pararmeters['blurring']):
        printGreen("6: Image is Not blurred ")
        results['blurring'] = True
    else:
        printRed("6:Image is blurred ")
        results['blurring'] = False

    # # 7.  Image should not contain Hard shadow and lighting
    # if checkShadow(image, **pararmeters['shadow']):
    #     printGreen("7: Image does not Contain Hard shadow / light")
    # else:
    #     printRed("7:Image Contains Hard shadow / light")

    # 8.  getBackgroundColor
    bgcolor = getBackgroundColorCorners(image, **pararmeters['background'])
    printGreen("8: Background Color :", bgcolor)
    is_color_in_range = isColorInRange(bgcolor, **pararmeters['background'])
    results['background'] = {
        'color': rgb_to_hex(*list(bgcolor)[::-1]),
        'color_match': is_color_in_range
    }

    # out_file_path = image_file.replace(
    #     'images/',
    #     'proc_images/{}/'.format(CORRECTION_LOG_SEGREGATION[correction_log]))

    out_file_path = image_file.replace('images', 'proc_images')
    # out_file_path = image_file
    # out_json_path = out_file_path.replace('.jpg', '.json')
    # out_json_path = out_json_path.replace('.jpeg', '.json')

    exif_bytes = None

    # 3. Dpi should be more than certain value
    if checkExactDpi(image_file, **pararmeters['dpi']):
        printGreen("3: Dpi in Correct Range")
        exif_bytes = copyExifData(image_file, out_file_path)
        results['dpi'] = True
    else:
        printRed("3: Dpi is out of Range or Does not exist in metadata")
        try:
            exif_bytes = correctDpi(image_file, out_file_path,
                                    **pararmeters['dpi'])
            results['dpi'] = False
        except:
            pass
            # results['aspectratio'] = is_aspect_correct
        # printGreen("2: Dpi Corrected to ({} ,{})".format(240, 240))

    quality, size = imwrite(out_file_path,
                            res,
                            **pararmeters['size'],
                            exif_bytes=exif_bytes)

    results['size'] = {'quality': quality, 'size': size}

    return results, out_file_path


pararmeters = {
    'resolution': {
        'min_height': 1079,
        'max_height': 1081,
        'min_width': 1399,
        'max_width': 1401
    },
    'output_resolution': {
        'height': 800,
        'width': 1000,
    },
    'aspectratio': {
        'min_aspect': 1.0 / 1.4,
        'max_aspect': 1.4
    },
    'dpi': {
        'min_dpi': 120,
        'max_dpi': 600,
        'out_dpi': 240,
    },
    'areacoverage': {
        'min_area_coverage': 0.4,
        'max_area_coverage': 0.5,
        # 'method': 'corners',
        'method': 'boundary',
        # 'output_resolution': [1600, 1600]
    },
    'objectcentering': {
        'min_center_range': 0.45,
        'max_center_range': 0.55
    },
    'size': {
        'max_out_size': 500000,
        'max_quality': 100,
        'min_quality': 5,
    },
    'blurring': {
        'blur_threshold': 10.0
    },
    'background': {
        'method': 'corners',
        # 'method': 'grabcut',
        # 'method': 'grabcut',
        'background': [[255, 255, 255], [220, 220, 220]],
        'variance': 8,
        'sigma': 0.98
    },
    'edge': {
        'sigma': 0.98
    },
    'shadow': {
        'value': 0.5
    }
}


def imageJob(image_file, pararmeters):
    img = cv2.imread(image_file)

    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # print(image_file)
    # isNonWhite = 'non_whites' in image_file

    results, out_file_path = validateImage(img, image_file, pararmeters)

    extension = os.path.splitext(out_file_path)[-1].replace('.', '')

    results['data'] = "data:image/{};base64,".format(extension.lower(
    )) + FileHandler.loadFile(out_file_path).decode('utf-8')
    return results
    # print(results)


def processFolder(directory, pararmeters):
    files = getFilesWithExtensions(
        directory, ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'])

    # fig = figure()
    # ax = fig.add_subplot(projection='3d')
    # ax2 = fig.add_subplot(projection='3d')

    # files = ['images/whites/shoe.jpg']
    # files = ['images/All Images/59347 -TAN/59347 -TAN -BACK 8.jpg']

    # files = [
    #     'images/non_whites/poco-c3-mzb07rhin-original-imafw8zkfygmfgfr.jpeg'
    # ]
    # files = [
    #     'images/whites/white-women-s-leather-sneakers-sun-shadows-background-top-view-stylish-youth-sports-shoes-genuine-footwear-minimalistic-212439057.jpg'
    # ]

    # print("processing {} images".format(len(files)))

    start = time.time()

    processes = []
    results = {}

    # for i, image_file in enumerate(files):
    #     re = imageJob(image_file, pararmeters)
    #     results[re['image_file']] = re

    # with Pool() as pool:
    #     pool.starmap(imageJob, zip(files, repeat(pararmeters)))

    with ThreadPoolExecutor() as executor:
        for re in executor.map(imageJob, files, [pararmeters] * len(files)):
            results[re['image_file']] = re

    #     executor.shutdown(wait=True)

    # out_json_path = os.path.join(directory, 'response.json')

    # with open(out_json_path, 'w', encoding='utf-8') as f:
    #     json.dump(results, f, ensure_ascii=False, indent=4, cls=CustomJSONizer)

    end = time.time()
    print("time:", end - start)
    return results


if '__main__' == __name__:

    import matplotlib
    matplotlib.use('GTK3Agg')
    from matplotlib import pyplot as plt

    processFolder('images', pararmeters)  #/All Images/59347 -TAN

    cv2.destroyAllWindows()

# res = correctAreaCoverage(img, do_center_object=True)

# rect = getEdgeBoundingBox(img)
# bgcolor = getBackgroundColorCorners(img, **pararmeters['edge'])
# printGreen("8: Background Color :", bgcolor)

# res, edge = processImage(img, 1.0)

# x, y, w, h = rect
# cv2.rectangle(res, (x, y), (x + w, y + h), (255, 0, 0), 1)

# cv2.imwrite(
#     out_file_path.replace('.jpeg',
#                           '_edge.jpeg').replace('.jpg', '_edge.jpg'),
#     edge)

# # img = cv2.imread(image_file)

# var = [
#     np.var(hsv[:, :, 0]),
#     np.var(hsv[:, :, 1]),
#     np.var(hsv[:, :, 2])
# ]
# med = [
#     np.median(hsv[:, :, 0]),
#     np.median(hsv[:, :, 1]),
#     np.median(hsv[:, :, 2])
# ]

# # print("variance",var)
# # print("median",med)
# color = 'b' if isNonWhite else 'r'

# ax.scatter(var[0], var[1], var[2], color=color)
# ax.text(var[0],
#         var[1],
#         var[2],
#         '%s' % (str(i)),
#         size=20,
#         zorder=1,
#         color='k')
# print(i, " : ", image_file.split('/')[-1])
# # color =  'g' if isNonWhite else 'y'
# ax2.scatter(med[0], med[1], med[2], color=color)
# ax2.text(med[0],
#          med[1],
#          med[2],
#          '%s' % (str(i)),
#          size=20,
#          zorder=1,
#          color='k')

# break

# imshow(res)
# plt.show()