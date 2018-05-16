import os



def _get_base_path():
    if not os.environ.get('OPENPOSE_MODEL', ''):
        return './models'
    return os.environ.get('OPENPOSE_MODEL')



def get_graph_path(model_name):
    dyn_graph_path = {
        'mobilenet_thin': \
            'models/graph/mobilenet_thin/graph_opt.pb'
            # './models/graph/mobilenet_thin/mobilenet_0.75_0.50_model-388003/graph_opt.pb'
    }
    graph_path = dyn_graph_path[model_name]
    for path in (graph_path, os.path.join(os.path.dirname(os.path.abspath(__file__)), graph_path), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', graph_path)):
        if not os.path.isfile(path):
            continue
        return path
    raise Exception('Graph file doesn\'t exist, path=%s' % graph_path)


def model_wh(resolution_str):
    width, height = map(int, resolution_str.split('x'))
    if width % 16 != 0 or height % 16 != 0:
        raise Exception('Width and height should be multiples of 16. w=%d, h=%d' % (width, height))
    return int(width), int(height)
