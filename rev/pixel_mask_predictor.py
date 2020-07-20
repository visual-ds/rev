import os

CWD = os.path.dirname(os.path.abspath(__file__))


def predict_mask(chart):
    mask_name = chart.predicted_mask_name().replace('.png', '')

    darknet_folder = os.path.join(CWD, '../../label_generator/darknet')
    cmd = 'echo \"{0}\" | {1}/darknet writing test {1}/cfg/writing.cfg {1}/../writing_backup/writing_18500.weights \"{2}\"'\
        .format(chart.image_name(), darknet_folder, mask_name)
    os.system(cmd)
