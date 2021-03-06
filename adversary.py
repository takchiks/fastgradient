import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models

from utils import compute_gradient, read_image, to_array

from ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default='2022-04-22_14-46', help='Experiment root')
    # parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()

def func(inp, net=None, target=None):

    out, _ = net(inp)
    loss = torch.nn.functional.nll_loss(out, target=torch.LongTensor([target]))

    print(f"Loss: {loss.item()}")
    return loss

def attack(tensor, net, step, eps=0.005, n_iter=5, orig_class="car", filename="output"):
    args = parse_args()
    new_tensor = tensor.detach().clone()
    old_tenor = tensor.detach().clone()
    # orig_prediction, _ = net(tensor)
    orig_prediction, _ = net(tensor)
    num_itr = 0

    data_path = 'data/modelnet40_normal_resampled/'

    if parse_args().num_category == 10:
        catfile = os.path.join(data_path, 'modelnet10_shape_names.txt')
    else:
        catfile = os.path.join(data_path, 'modelnet40_shape_names.txt')

    cat = [line.rstrip() for line in open(catfile)]
    classes = dict(zip(cat, range(len(cat))))

    # log_string(orig_prediction)
    orig_prediction = orig_prediction.argmax()
    new_prediction = orig_prediction.argmax()

    log_string(f"{tensor.size()}")
    print(f"Original class: {orig_class}")

    for i in range(n_iter):

        if cat.index(orig_class) != orig_prediction:
            num_itr = 500
            print(f"We fooled the network after {i+1} iterations!")
            print(f"New prediction: {cat[orig_prediction]}")
            # log_string(new_tensor.transpose(1,2).detach().numpy())
            break

        net.zero_grad()
        # log_string(orig_prediction.item())
        grad = compute_gradient(
                func, new_tensor, net=net, target=orig_prediction
                )
        new_tensor = torch.clamp(new_tensor + eps * grad.sign(), -2, 2)
        new_tensor = new_tensor
        new_prediction, _ = net(new_tensor)
        # new_prediction, _ = net(new_tensor)
        new_prediction = new_prediction.argmax()
        num_itr = i
        if cat.index(orig_class) != new_prediction:
            num_itr = i
            print(f"We fooled the network after {i+1} iterations!")
            print(f"New prediction: {cat[new_prediction]}")
            print(f"{filename} \n")
            # log_string(new_tensor.transpose(1,2).detach().numpy())
            break


    new_prediction, _ = net(new_tensor)
    new_prediction = new_prediction.argmax()

    diff_tenor = torch.clamp(new_tensor - tensor.detach().clone(),-2,2)
    tensor_numpy = new_tensor.transpose(1, 2).detach().numpy()
    diff_tenor_numpy = diff_tenor.transpose(1, 2).detach().numpy()
    tensor_string = ""
    tenor_string = ""
    # log_string(old_tenor)
    # log_string(new_tensor)
    # log_string(diff_tenor)

    # import numpy as np

    # # data = np.loadtxt("../examples/airplane/0_output.txt", delimiter = "\n")
    # point_list = []
    # i = 0
    # with open('../examples/airplane/airplane_0628.txt', 'r') as data:
    #     for line in data:
    #         point_list.append([])
    #         point_list[i] = [n for n in line.split(',')]
    #         i += 1
    #
    data_path= f"examples/{eps}and{n_iter}"
    # np.append(tensor_numpy[0],point_list[:,3],axis=1)
    # np.append(tensor_numpy[0],point_list[:,4],axis=1)
    # np.append(tensor_numpy[0],point_list[:,5],axis=1)

    for a in tensor_numpy[0]:
        tensor_string =f"{tensor_string}{a[0]},{a[1]},{a[2]}\n"
        # tensor_string = tensor_string + ','.join(str(v) for v in tensor_numpy) + "\n"

    for a in diff_tenor_numpy[0]:
        tenor_string =f"{tenor_string}{a[0]},{a[1]},{a[2]}\n"
        # tensor_string = tensor_string + ','.join(str(v) for v in tensor_numpy) + "\n"

    filenaming = os. path. join(data_path,f"{orig_class}", f"{filename}.txt")
    text_file = open(filenaming, "w")
    text_file.write(tensor_string)
    text_file.close()

    filenaming = os. path. join(data_path,f"{orig_class}", f"{filename}_diff.txt")
    text_file = open(filenaming, "w")
    text_file.write(tenor_string)
    text_file.close()
    # ','.join(map(str, a))

    if cat.index(orig_class) == new_prediction:
        print(f"After {n_iter} epochs the model could not be fooled! ")
        # log_string(new_tensor)
        # log_string(f"{new_tensor.size()}")

    return diff_tenor ,new_tensor , orig_prediction.item(), new_prediction.item(), num_itr


if __name__ == "__main__":

    def log_string(str):
        logger.info(str)
        print(str)


    args = parse_args()
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10)
    shape_names = test_dataset.getPointCloudFiles()
    fileshape = test_dataset.getFileShape()

    if parse_args().num_category == 10:
        catfile = os.path.join(data_path, 'modelnet10_shape_names.txt')
    else:
        catfile = os.path.join(data_path, 'modelnet40_shape_names.txt')

    cat = [line.rstrip() for line in open(catfile)]

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cpu()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth', map_location=torch.device('cpu'))
    classifier.load_state_dict(checkpoint['model_state_dict'])

    # net = models.resnet18(pretrained=True)
    # net.eval()
    #
    # tensor = read_image("img.jpg")
    # log_string(tensor)

    net = classifier
    net.eval()

    # for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
    # log_string(test_dataset)
    total = 0
    countdiff=0
    class_total = [0 for i in range(40)]
    class_countdiff = [0 for i in range(40)]
    real_adv = []
    epsilon = 0.001
    epochs = 5
    logging_string = ""

    for step, (x, y) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
        x = x.transpose(2, 1)
        # x_t = torch.Tensor(x)
        # log_string(x_t)
        # y_t = torch.Tensor(y)
        tensor = x
        diff ,new_tensor, orig_prediction, new_prediction, num_itr = attack(
            tensor, net, step, eps=epsilon, n_iter=epochs, orig_class=shape_names[step], filename=fileshape[step]
            )
        total += 1
        class_total[cat.index(shape_names[step])] += 1
        if cat.index(shape_names[step]) != new_prediction:
            countdiff += 1
            class_countdiff[cat.index(shape_names[step])] += 1
            if cat.index(shape_names[step]) == orig_prediction:
                real_adv.append([fileshape[step], num_itr, shape_names[step], cat[new_prediction]])

    for i in range(len(class_total)):
        accuracy = 1 - (class_countdiff[i]/class_total[i])
        logging_string+= f"The CLASS accuracy after adding a perturbation of {cat[i]} = {accuracy*100}% \n"
        log_string(f"The CLASS accuracy after adding a perturbation of {cat[i]} = {accuracy*100}% ")

    log_string(f"{real_adv}")

    accuracy = 1 - (countdiff/total)
    log_string(f"\n E = {epsilon} ep = {epochs} \n The overall accuracy of the model after adding a perturbation is now {accuracy*100}%")
    logging_string+=f"\n E = {epsilon} ep = {epochs} \n The overall accuracy of the model after adding a perturbation is now {accuracy*100}%\n"
    logging_string+=f"{real_adv}"
    data_path= f"examples/{epsilon}and{epochs}"
    filenaming = os. path. join(data_path,f"accuracies.txt")
    text_file = open(filenaming, "w")
    text_file.write(logging_string)
    text_file.close()
        # arr = to_array(new_tensor)

    # _, (ax_orig, ax_new, ax_diff) = plt.subplots(1, 3, figsize=(19.20,10.80))
    # arr = to_array(tensor)
    # new_arr = to_array(new_tensor)
    # diff_arr = np.abs(arr - new_arr).mean(axis=-1)
    # diff_arr = diff_arr / diff_arr.max()
    #
    # ax_orig.imshow(arr)
    # ax_new.imshow(new_arr)
    # ax_diff.imshow(diff_arr, cmap="gray")
    #
    # ax_orig.axis("off")
    # ax_new.axis("off")
    # ax_diff.axis("off")
    #
    # ax_orig.set_title(f"Original: {orig_prediction}")
    # ax_new.set_title(f"Modified: {new_prediction}")
    # ax_diff.set_title("Difference")
    #
    # plt.savefig("res_1.png")
