from Forward_Dataloader import *
from Model import *
from torch.autograd import Variable
import os
from scipy import io as sio
import matplotlib


# Create data loader
def set_data_sets(data_transforms, batch_size, dirpath=None):
    forward_dataset = DA_jp2000_ForwardDataloader(R=100, transform=data_transforms, use_cuda=use_cuda, dirpath=dirpath)
    forwardloader = DataLoader(forward_dataset, batch_size=batch_size, shuffle=False)
    print('Loaded %d of Forward Images' % forward_dataset.__len__())
    return forwardloader

def create_dir_safely(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

print('Forward pass through CNN')

use_cuda = torch.cuda.is_available()

# create new instance
net = NetStride5()
path = './Forward_Results/jp2/Net5_lossUxUy/'
net_file = 'net_files/081218_jp2_UxUy/4_NetStride5_Adam_32_53p43.pkl'
images_src_dir = 'Forward_Images/jp2/'

# load the pretrained state_dict & filter for not existing keys
pretrained_dict = torch.load(net_file)
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in net.state_dict()}

# update the net
net.load_state_dict(pretrained_dict)
net = tensor_to_gpu(net, use_cuda)

print('Forward-path begin')
i = 0

forwardloader = set_data_sets(data_transforms=None, batch_size=1, dirpath=images_src_dir)
net.eval()
for data in forwardloader:
    image, deformed = data
    # for stride2 & 4 together
    #if image.size(2)%4 > 0 or image.size(3)%4:
    #    i += 1
    #    continue
    image, deformed = tensor_to_gpu(Variable(image),use_cuda), tensor_to_gpu(Variable(deformed),use_cuda)
    output_img, output_uxuy = net(image)
    print(i)
    dir_name = path + str(i)
    create_dir_safely(dir_name)
    matplotlib.pyplot.imsave(dir_name + '/' + 'original.png', image[0].data.permute(1, 2, 0).type('torch.ByteTensor'))
    matplotlib.pyplot.imsave(dir_name + '/' + 'y_iterative.png', deformed[0].data.permute(1, 2, 0).type('torch.ByteTensor'))
    matplotlib.pyplot.imsave(dir_name + '/' + 'net_deformed.png', output_img[0].data.permute(1, 2, 0).type('torch.ByteTensor'))
    ux_dict = {'ux': tensor_to_cpu(output_uxuy[0][0],use_cuda).data.numpy()}
    uy_dict = {'uy': tensor_to_cpu(output_uxuy[0][1],use_cuda).data.numpy()}
    sio.savemat(dir_name + '/' + 'new_ux.mat', ux_dict)
    sio.savemat(dir_name + '/' + 'new_uy.mat', uy_dict)
    i += 1

print('Finished')
