from flask import Flask, request
import base64
import os
from flask import jsonify
import json
from flask import send_file, send_from_directory
from process import *
from pathlib import Path
import shutil
import FileHandler

app = Flask(__name__)


def saveFile(filename, data):
    Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as fh:
        fh.write(data)


def loadFile(filename):
    with open(filename, "rb") as fh:
        out_data = base64.b64encode(fh.read())


TEMP_DIRECTORY = '/tmp/Temp/data'


def getPararmeters(request, parameters):
    parameters['output_resolution']['width'] = max(
        int(request.values['width']), parameters['resolution']['min_width'])
    parameters['output_resolution']['height'] = max(
        int(request.values['height']), parameters['resolution']['min_height'])

    parameters['resolution']['min_width'] = parameters['output_resolution'][
        'width']
    parameters['resolution']['min_height'] = parameters['output_resolution'][
        'height']

    parameters['resolution']['max_width'] = parameters['output_resolution'][
        'width']
    parameters['resolution']['max_height'] = parameters['output_resolution'][
        'height']

    if 'min_area_coverage' in request.values:
        parameters['areacoverage']['min_area_coverage'] = float(
            request.values['min_area_coverage'])

    if 'max_area_coverage' in request.values:
        parameters['areacoverage']['max_area_coverage'] = float(
            request.values['max_area_coverage'])

    if 'min_center_range' in request.values:
        parameters['objectcentering']['min_center_range'] = float(
            request.values['min_center_range'])

    if 'max_center_range' in request.values:
        parameters['objectcentering']['max_center_range'] = float(
            request.values['max_center_range'])

    if 'max_out_size' in request.values:
        parameters['size']['max_out_size'] = int(
            request.values['max_out_size'])

    if 'min_out_size' in request.values:
        parameters['size']['min_out_size'] = int(
            request.values['min_out_size'])

    if 'min_in_size' in request.values:
        parameters['size']['min_in_size'] = int(request.values['min_in_size'])

    if 'max_in_size' in request.values:
        parameters['size']['max_in_size'] = int(request.values['max_in_size'])

    if 'max_quality' in request.values:
        parameters['size']['max_quality'] = int(request.values['max_quality'])

    if 'min_quality' in request.values:
        parameters['size']['min_quality'] = int(request.values['min_quality'])

    if 'out_dpi' in request.values:
        parameters['dpi']['out_dpi'] = int(request.values['out_dpi'])

    colors_dict = [
        hex_to_rgb('#' + v)[::-1] for k, v in request.values.items()
        if k.startswith('background_color')
    ]

    parameters['background']['colors'] = colors_dict

    # if 'out_dpi' in request.values:
    #     parameters['dpi']['out_dpi'] = int(request.values['out_dpi'])

    return parameters


@app.route('/hello/', methods=['GET', 'POST', 'OPTIONS'])
def welcome():

    #prepare input data
    body = request.data

    parameters = getPararmeters(request, global_parameters.copy())

    Path(os.path.join(TEMP_DIRECTORY, 'images')).mkdir(parents=True,
                                                       exist_ok=True)

    files = request.files.getlist("file")
    for x in files:
        x.save(os.path.join(TEMP_DIRECTORY, 'images', x.filename))

    # urls = [
    #     hex_to_rgb('#' + v)[::-1] for k, v in request.values.items()
    #     if k.startswith('background_color')
    # ]

    urls = [
        value for key, value in request.form.items()
        if key.startswith('links[')
    ]

    FileHandler.getAllUrlImage(urls, os.path.join(TEMP_DIRECTORY, 'images'))

    #process
    results = processFolder(os.path.join(TEMP_DIRECTORY, 'images'), parameters)

    #prepare responce
    response = app.response_class(response=json.dumps(results,
                                                      ensure_ascii=False,
                                                      indent=4,
                                                      cls=CustomJSONizer),
                                  status=200,
                                  mimetype='application/json')

    # json_filename = os.path.join(TEMP_DIRECTORY, 'response.json')

    # zip_filename = os.path.dirname(TEMP_DIRECTORY) + '.zip'

    # FileHandler.compressZipFile(zip_filename, TEMP_DIRECTORY)

    # resp = send_file(json_filename, mimetype='application/json')
    # shutil.rmtree(TEMP_DIRECTORY)
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
