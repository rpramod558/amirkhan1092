import numpy as np
import pyclipper

# import matplotlib.pyplot as plt


def getCurvePoint(xy, aspect):
    y = np.sqrt(xy / aspect)
    x = aspect * y
    return [x, y]


def getAspectAreaPolygon(pararmeters, rect, num=5):
    x, y, rect_width, rect_height = rect

    xymin = rect_width * rect_height / pararmeters['areacoverage'][
        'min_area_coverage']
    xymax = rect_width * rect_height / pararmeters['areacoverage'][
        'max_area_coverage']

    x_by_y_min = pararmeters['aspectratio']['min_aspect']
    x_by_y_max = pararmeters['aspectratio']['max_aspect']

    aspect_area_polygon = []

    for aspect in np.geomspace(x_by_y_max, x_by_y_min, num):
        # print('aspect', aspect)
        aspect_area_polygon.append(getCurvePoint(xymin, aspect))
    for aspect in np.geomspace(x_by_y_min, x_by_y_max, num):
        # print('aspect', aspect)
        aspect_area_polygon.append(getCurvePoint(xymax, aspect))

    aspect_area_polygon = np.array(aspect_area_polygon)

    return aspect_area_polygon


def getAspectAreaLine(pararmeters, rect, num=5):
    x, y, rect_width, rect_height = rect

    common_aspect = pararmeters['output_resolution']['width'] / pararmeters[
        'output_resolution']['height']

    xymin = np.max([rect_width, rect_height
                    ])**2 / pararmeters['areacoverage']['min_area_coverage']
    xymax = np.max([rect_width, rect_height
                    ])**2 / pararmeters['areacoverage']['max_area_coverage']

    # x_by_y = common_aspect
    # x_by_y_max = pararmeters['aspectratio']['max_aspect']

    aspect_area_line = []

    aspect_area_line.append(getCurvePoint(xymin, common_aspect))
    aspect_area_line.append(getCurvePoint(xymax, common_aspect))

    aspect_area_line = np.array(aspect_area_line)

    return aspect_area_line


def getResolutionPolygon(pararmeters, rect):
    x, y, rect_width, rect_height = rect
    resolution_polygon = np.array([
        [
            pararmeters['resolution']['min_width'],
            pararmeters['resolution']['min_height']
        ],
        [
            pararmeters['resolution']['min_width'],
            pararmeters['resolution']['max_height']
        ],
        [
            pararmeters['resolution']['max_width'],
            pararmeters['resolution']['max_height']
        ],
        [
            pararmeters['resolution']['max_width'],
            pararmeters['resolution']['min_height'],
        ],
    ])

    return resolution_polygon


def findCentroid(v):
    ans = [0, 0]

    n = len(v)
    signedArea = 0

    # For all vertices
    for i in range(len(v)):

        x0 = v[i][0]
        y0 = v[i][1]
        x1 = v[(i + 1) % n][0]
        y1 = v[(i + 1) % n][1]

        # Calculate value of A
        # using shoelace formula
        A = (x0 * y1) - (x1 * y0)
        signedArea += A

        # Calculating coordinates of
        # centroid of polygon
        ans[0] += (x0 + x1) * A
        ans[1] += (y0 + y1) * A

    signedArea *= 0.5
    ans[0] = (ans[0]) / (6 * signedArea)
    ans[1] = (ans[1]) / (6 * signedArea)

    return ans


def interset(aspect_area_polygon, resolution_polygon):
    pc = pyclipper.Pyclipper()
    pc.AddPath(aspect_area_polygon, pyclipper.PT_CLIP, True)
    pc.AddPath(resolution_polygon, pyclipper.PT_SUBJECT, True)
    solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD,
                          pyclipper.PFT_EVENODD)
    if len(solution) > 0:
        return np.array(solution[0])
    else:
        return None


CORRECTION_UNREQUIRED = 0
CORRECTION_BOUNDS_SATISFIED = 1
CORRECTION_BOUNDS_SCALED = 2
CORRECTION_BOUNDS_UNSATISFIED = 3


def getFinalPararmeters(pararmeters, rect, original_resolution):
    aspect_area_polygon = getAspectAreaPolygon(pararmeters, rect)
    resolution_polygon = getResolutionPolygon(pararmeters, rect)

    solution = interset(aspect_area_polygon, resolution_polygon)

    if solution is not None:
        centroid = findCentroid(solution)
        # plotPolys(resolution_polygon, aspect_area_polygon, solution, centroid,
        #           original_resolution)
        # print('image correction bounds satisfied')
        return centroid, 1, CORRECTION_BOUNDS_SATISFIED
    else:
        # scaling required
        resolution_polygon_centroid = findCentroid(resolution_polygon)
        aspect_area_polygon_centroid = findCentroid(aspect_area_polygon)

        scaling_factor = np.linalg.norm(
            aspect_area_polygon_centroid) / np.linalg.norm(
                resolution_polygon_centroid)

        pararmeters['resolution']['min_width'] *= scaling_factor
        pararmeters['resolution']['min_height'] *= scaling_factor
        pararmeters['resolution']['max_width'] *= scaling_factor
        pararmeters['resolution']['max_height'] *= scaling_factor

        resolution_polygon = getResolutionPolygon(pararmeters, rect)

        solution = interset(aspect_area_polygon, resolution_polygon)

        if solution is not None:
            centroid = findCentroid(solution)
            # plotPolys(resolution_polygon, aspect_area_polygon, solution,
            #           centroid, original_resolution)
            # print('image correction bounds satisfied after scaling')
            return centroid, 1.0 / scaling_factor, CORRECTION_BOUNDS_SCALED
        else:
            # print(
            #     'image correction bounds not satisfied even after scaling (forcing resolution that works)'
            # )
            return aspect_area_polygon_centroid, 1.0 / scaling_factor, CORRECTION_BOUNDS_UNSATISFIED


def getFinalPararmetersFixed(pararmeters, rect, original_resolution):
    aspect_area_line = getAspectAreaLine(pararmeters, rect)

    output_resolution = np.array([
        pararmeters['output_resolution']['width'],
        pararmeters['output_resolution']['height']
    ])

    common_aspect = pararmeters['output_resolution']['width'] / pararmeters[
        'output_resolution']['height']

    pararmeters['aspectratio']['min_aspect'] = common_aspect * 0.9
    pararmeters['aspectratio']['max_aspect'] = common_aspect / 0.9

    # aspect_area_polygon = getAspectAreaPolygon(pararmeters, rect)
    # resolution_polygon = getResolutionPolygon(pararmeters, rect)

    scaling_factor_max = np.linalg.norm(
        aspect_area_line[0]) / np.linalg.norm(output_resolution)
    scaling_factor_min = np.linalg.norm(
        aspect_area_line[1]) / np.linalg.norm(output_resolution)

    scaling_factor = (scaling_factor_max + scaling_factor_min) / 2.0

    # plotPoints([
    #     output_resolution, output_resolution * scaling_factor,
    #     aspect_area_line[0], aspect_area_line[1]
    # ])

    if scaling_factor_min < 1.0 and scaling_factor_max > 1.0:
        return output_resolution, output_resolution, CORRECTION_BOUNDS_SATISFIED
    else:
        return output_resolution * scaling_factor, output_resolution, CORRECTION_BOUNDS_SCALED

    # scaling_factor = np.linalg.norm(
    #     aspect_area_polygon_centroid) / np.linalg.norm(
    #         resolution_polygon_centroid)

    # solution = interset(aspect_area_polygon, resolution_polygon)

    # if solution is not None:
    #     centroid = findCentroid(solution)
    #     # plotPolys(resolution_polygon, aspect_area_polygon, solution, centroid,
    #     #           original_resolution)
    #     # print('image correction bounds satisfied')
    #     return output_resolution, 1, CORRECTION_BOUNDS_SATISFIED
    # else:
    #     # scaling required
    #     resolution_polygon_centroid = findCentroid(resolution_polygon)
    #     aspect_area_polygon_centroid = findCentroid(aspect_area_polygon)

    #     scaling_factor = np.linalg.norm(
    #         aspect_area_polygon_centroid) / np.linalg.norm(
    #             resolution_polygon_centroid)

    #     pararmeters['resolution']['min_width'] *= scaling_factor
    #     pararmeters['resolution']['min_height'] *= scaling_factor
    #     pararmeters['resolution']['max_width'] *= scaling_factor
    #     pararmeters['resolution']['max_height'] *= scaling_factor

    #     resolution_polygon = getResolutionPolygon(pararmeters, rect)

    #     solution = interset(aspect_area_polygon, resolution_polygon)

    #     if solution is not None:
    #         centroid = findCentroid(solution)
    #         # plotPolys(resolution_polygon, aspect_area_polygon, solution,
    #         #           centroid, original_resolution)
    #         # print('image correction bounds satisfied after scaling')
    #         return output_resolution, 1.0 / scaling_factor, CORRECTION_BOUNDS_SCALED
    #     else:
    #         # print(
    #         #     'image correction bounds not satisfied even after scaling (forcing resolution that works)'
    #         # )
    #         return output_resolution, 1.0 / scaling_factor, CORRECTION_BOUNDS_UNSATISFIED


def plotPolys(resolution_polygon, aspect_area_polygon, solution, centroid,
              original_resolution):
    x = resolution_polygon[:, 0]
    y = resolution_polygon[:, 1]
    plt.figure(figsize=(8, 8))
    plt.axis('equal')
    plt.fill(x, y)

    x = aspect_area_polygon[:, 0]
    y = aspect_area_polygon[:, 1]
    plt.fill(x, y)

    x = solution[:, 0]
    y = solution[:, 1]
    # print(solution)
    plt.fill(x, y)

    plt.plot(centroid[0], centroid[1], 'ro')
    plt.plot(original_resolution[0], original_resolution[1], 'ro')
    plt.show()


def plotPoints(points):
    plt.figure(figsize=(8, 8))
    plt.axis('equal')
    for point in points:
        plt.plot(point[0], point[1], 'ro')
    plt.show()


if '__main__' == __name__:

    pararmeters = {
        'resolution': {
            'min_height': 480,
            'max_height': 1080,
            'min_width': 720,
            'max_width': 1400
        },
        'output_resolution': {
            'height': 1600,
            'width': 1600,
        },
        'aspectratio': {
            'min_aspect': 0.5,
            'max_aspect': 2.0
        },
        'dpi': {
            'min_dpi': 120,
            'max_dpi': 600,
            'out_dpi': 240,
        },
        'areacoverage': {
            'min_area_coverage': 0.5,
            'max_area_coverage': 0.7,
            # 'method': 'corners',
            'method': 'boundary',
            # 'output_resolution': [1600, 1600]
        },
        'objectcentering': {
            'min_center_range': 0.45,
            'max_center_range': 0.55
        },
        'blurring': {
            'blur_threshold': 10.0
        },
        'background': {
            'method': 'corners',
            # 'method': 'grabcut',
            # 'method': 'grabcut',
            'color': [255, 255, 255],
            'sigma': 0.98
        },
        'edge': {
            'sigma': 0.98
        }
    }

    print(
        getFinalPararmetersFixed(pararmeters, [40, 40, 700, 480], [1080, 720]))
