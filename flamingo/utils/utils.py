import json
import yaml

try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError("No tensorboardX package is found. Please install with the command: \npip install tensorboardX")


def read_config(file_path):
    with open(file_path, 'r') as f:
        json_object = json.load(f)
    return json_object


def dump_json(file_path, data):
    with open(file_path, 'w') as fp:
        json.dump(data, fp)
    return


def load_json(file_path):
    with open(file_path, 'r') as f:
        json_object = json.load(f)
    return json_object


def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        x = yaml.load(f)
    return x


def create_summary_writer(model, data_loader, log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    x, y = next(data_loader_iter)
    # try:
    #     writer.add_graph(model, x)
    # except Exception as e:
    #     print("Failed to save model graph: {}".format(e))
    return writer
