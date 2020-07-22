import os

CWD = os.path.dirname(os.path.abspath(__file__))


def predict_mask(chart):

    # Absolute paths for darknet model
    mask_path =  os.path.join(CWD,'..', '..', chart.filename.replace('.png', chart._prefix + '-mask'))
    img_path =  os.path.join(CWD, '..', '..', chart.filename)
    text_detection_model_path = os.path.join(CWD, '..', '..', 'models', 'darknet_text_detection')
    darknet_lib_path = '/home/joao/Documents/repos/darknet/./darknet'

    # Darknet text-detection command line
    command_line = '"{darknet_lib}" writing test "{config}" "{model}" "{img}" "{mask}"'.format(
        darknet_lib = darknet_lib_path,
        config = os.path.join(text_detection_model_path, 'writing.cfg'),
        model = os.path.join(text_detection_model_path, 'writing_18500.weights'),
        img = img_path,
        mask = mask_path
    )
    # Execute please!
    os.system(command_line)