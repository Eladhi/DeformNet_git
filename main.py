from DataLoader import *
from Model import *
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
import datetime
# imports for send_email
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


# General preferences
use_cuda = torch.cuda.is_available()
wanted_img_size = [128, 128]
epochs = 500
iterations = 30
log_file = 'logs/log.log'
#

# configuration options
def set_cfg(i):
    cfg = {'architecture':None, 'optimizer':None, 'batch_size':None, 'scheduler_gamma':None}
    res = i % 7
    if res == 0:
        cfg['architecture'] = 'NetStride1'
    elif res == 1:
        cfg['architecture'] = 'NetBasicCNN'
    #elif res == 2:
    #    cfg['architecture'] = 'NetStride2'
    #elif res == 3:
    #    cfg['architecture'] = 'NetStride2_resnet'
    elif res == 2:
        cfg['architecture'] = 'NetStride1_resnet'
    elif res == 3:
        cfg['architecture'] = 'NetStride3'
    #elif res == 6:
    #    cfg['architecture'] = 'NetStride4'
    elif res == 4:
        cfg['architecture'] = 'NetStride5'
    elif res == 5:
        cfg['architecture'] = 'NetStride6'
    else:
        cfg['architecture'] = 'NetStride2_resnet_ker5'
    #cfg['architecture'] = random.choice(['NetBasicCNN', 'NetStride1', 'NetStride2', 'NetStride3', 'NetStride4',
    #                        'NetStride5', 'NetStride1_resnet', 'NetStride2_resnet'])
    #cfg['architecture'] = random.choice(['NetStride2_resnet'])
    cfg['optimizer'] = random.choice(['Adam'])
    cfg['batch_size'] = random.choice([32])
    #cfg['batch_size'] = random.choice([64])
    cfg['learning_rate'] = random.choice([0.001])
    cfg['scheduler_gamma'] = random.choice([0.1])
    return cfg


# Create train&test dataloaders
def set_data_sets(data_transforms, batch_size):
    train_dataset = DA_jp2000_dataloader(R=100, wanted_img_size=wanted_img_size, Train=True, transform=data_transforms, use_cuda=use_cuda)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print('Loaded %d of Training Images' % train_dataset.__len__())
    test_dataset = DA_jp2000_dataloader(R=100, wanted_img_size=wanted_img_size, Train=False, transform=data_transforms, use_cuda=use_cuda)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print('Loaded %d of Test Images' % test_dataset.__len__())
    return trainloader, testloader


# create the configured net
def init_net(cfg):
    if (cfg['architecture']=='NetBasicCNN'):
        net = tensor_to_gpu(NetBasicCNN(), use_cuda)
    elif (cfg['architecture']=='NetStride1'):
        net = tensor_to_gpu(NetStride1(), use_cuda)
    elif (cfg['architecture']=='NetStride2'):
        net = tensor_to_gpu(NetStride2(), use_cuda)
    elif (cfg['architecture']=='NetStride3'):
        net = tensor_to_gpu(NetStride3(), use_cuda)
    elif (cfg['architecture']=='NetStride4'):
        net = tensor_to_gpu(NetStride4(), use_cuda)
    elif (cfg['architecture']=='NetStride5'):
        net = tensor_to_gpu(NetStride5(), use_cuda)
    elif (cfg['architecture'] == 'NetStride1_resnet'):
        net = tensor_to_gpu(NetStride1_resnet(), use_cuda)
    elif (cfg['architecture'] == 'NetStride6'):
        net = tensor_to_gpu(NetStride6(), use_cuda)
    elif (cfg['architecture'] == 'NetStride2_resnet_ker5'):
        net = tensor_to_gpu(NetStride2_resnet_ker5(), use_cuda)
    else:
        net = tensor_to_gpu(NetStride1(), use_cuda)
    return net


# create the configured optimizer
def set_optimizer(cfg, net):
    if (cfg['optimizer']=='Adam'):
        optimizer = optim.Adam(net.parameters(), lr=cfg['learning_rate'])
    else:
        optimizer = optim.Adam(net.parameters(), lr=cfg['learning_rate'])
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=cfg['scheduler_gamma'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250, 425], gamma=cfg['scheduler_gamma'])
    return optimizer, scheduler


# Perform training
def train_net(net, criterion, optimizer, scheduler, trainloader, testloader):
    err_vec = []
    err_test_vec = []
    uxuy_err_vec = []
    uxuy_err_test_vec = []
    identity_err_vec = []
    epoch_idx = 0

    print('Start Training')
    for epoch in range(epochs):
        net.train()
        scheduler.step()
        running_loss = 0.0
        running_loss_test = 0.0
        running_uxuy_loss = 0.0
        running_uxuy_loss_test = 0.0
        running_loss_identity = 0.0


        for i, data in enumerate(trainloader, 0):
            # get inputs
            image, uxuy, deformed = data
            image, uxuy, deformed = tensor_to_gpu(Variable(image),use_cuda), tensor_to_gpu(Variable(uxuy),use_cuda), tensor_to_gpu(Variable(deformed),use_cuda)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            output_img, output_uxuy = net(image)
            # exclude 10 border pixels for the loss calculation
            output_img = output_img[:, :, 10:-10, 10:-10]
            deformed = deformed[:, :, 10:-10, 10:-10]
            output_uxuy = output_uxuy[:, :, 10:-10, 10:-10]
            uxuy = uxuy[:, :, 10:-10, 10:-10]
            # backward + optimize
            loss = criterion(output_img, deformed)
            loss_uxuy = criterion(output_uxuy,uxuy)
            #loss.backward()
            loss_uxuy.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data
            running_uxuy_loss += loss_uxuy.data

        # epoch statistics
        # training performance
        err_vec += [running_loss / trainloader.__len__()]
        uxuy_err_vec += [running_uxuy_loss / trainloader.__len__()]
        running_loss = 0
        running_uxuy_loss = 0
        epoch_idx += 1
        print('[%d] loss train: %.3f' % (epoch_idx, err_vec[-1]))
        print('[%d] loss flow train: %.3f' % (epoch_idx, uxuy_err_vec[-1]))
        # test performance
        net.eval()
        for data in testloader:
            image, uxuy, deformed = data
            image, uxuy, deformed = tensor_to_gpu(Variable(image),use_cuda), tensor_to_gpu(Variable(uxuy),use_cuda), tensor_to_gpu(Variable(deformed),use_cuda)
            # forward
            output_img, output_uxuy = net(image)
            loss = criterion(output_img, deformed)
            loss_uxuy = criterion(output_uxuy, uxuy)
            loss_identity = criterion(image, deformed)
            # calc statistics
            running_loss_test += loss.data
            running_uxuy_loss_test += loss_uxuy.data
            running_loss_identity += loss_identity.data
        err_test_vec += [running_loss_test / testloader.__len__()]
        uxuy_err_test_vec += [running_uxuy_loss_test / testloader.__len__()]
        identity_err_vec += [running_loss_identity / testloader.__len__()]
        print('[%d] loss test: %.3f' % (epoch_idx, err_test_vec[-1]))
        print('[%d] loss flow test: %.3f' % (epoch_idx, uxuy_err_test_vec[-1]))

    print('Finished Training')
    return net, err_vec, err_test_vec, uxuy_err_vec, uxuy_err_test_vec, identity_err_vec

def save_error_plot(cfg, err_vec, err_test_vec, uxuy_err_vec, uxuy_err_test_vec, identity_err_vec, i):
    plt.subplot(211)
    plt.plot(err_vec, label='Training Data')
    plt.plot(err_test_vec, label='Test Data')
    plt.plot(identity_err_vec, label='Test Identity')
    plt.ylabel('MSE loss')
    plt.xlabel('Epoch')
    plt.title('Deformed image error')
    plt.legend(loc='upper right')
    plt.subplot(212)
    plt.plot(uxuy_err_vec, label='Training Data')
    plt.plot(uxuy_err_test_vec, label='Test Data')
    plt.ylabel('MSE loss')
    plt.xlabel('Epoch')
    plt.title('Ux&Uy flow error')
    plt.legend(loc='upper right')
    plt.suptitle('Arch: %s | Optimizer: %s %f | Batch S: %d' % (cfg['architecture'], cfg['optimizer'], cfg['learning_rate'], cfg['batch_size']))
    file_path = 'figs/%d_%s_%s_%d' % (i, cfg['architecture'], cfg['optimizer'], cfg['batch_size'])
    plt.savefig(file_path)
    plt.clf()
    return (file_path + '.png')


def save_net(net, cfg, acc, i):
    str_acc = ('%.2f' % acc).replace('.', 'p')
    net_to_save = tensor_to_cpu(net,use_cuda)
    filename = 'net_files/' + str(i) + '_' + cfg['architecture'] + '_' + cfg['optimizer'] + '_' + str(cfg['batch_size']) + '_' + str_acc + '.pkl'
    torch.save(net_to_save.state_dict(), filename)


def writelog(i, cfg, acc_train, acc_test, acc_flow_train, acc_flow_test, time_dif):
    str_acc_train = ('%.2f' % acc_train)
    str_acc_test = ('%.2f' % acc_test)
    str_acc_flow_train = ('%.2f' % acc_flow_train)
    str_acc_flow_test = ('%.2f' % acc_flow_test)
    line2write = (str(i) + ' ' + cfg['architecture'] + ', ' + cfg['optimizer'] + ', ' + str(cfg['batch_size']) + ', ' +
                   'acc training: ' + str_acc_train + ', acc test: ' + str_acc_test + ', acc flow training: ' +
                   str_acc_flow_train + ', acc flow test: ' + str_acc_flow_test + ', time elapsed: ' +
                  str(time_dif))
    log = open(log_file,'a')
    log.write(line2write+'\n')
    log.close()


def send_mail(subject, file_path=None):
    fromaddr = "vm.cgm.project@gmail.com"
    toaddr = "eladhirsh90@gmail.com"
    # instance of MIMEMultipart
    msg = MIMEMultipart()
    # storing the senders email address
    msg['From'] = fromaddr
    # storing the receivers email address
    msg['To'] = toaddr
    # storing the subject
    msg['Subject'] = subject
    # string to store the body of the mail
    body = "GO GIRLL"
    # attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))
    # open the file to be sent
    attachment = open(file_path, "rb")
    # instance of MIMEBase and named as p
    p = MIMEBase('application', 'octet-stream')
    # To change the payload into encoded form
    p.set_payload((attachment).read())
    # encode into base64
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', "attachment; filename= %s" % file_path)
    # attach the instance 'p' to instance 'msg'
    msg.attach(p)
    # creates SMTP session
    s = smtplib.SMTP('smtp.gmail.com', 587)
    # start TLS for security
    s.starttls()
    # Authentication
    s.login(fromaddr, "summer17")
    # Converts the Multipart msg into a string
    text = msg.as_string()
    # sending the mail
    s.sendmail(fromaddr, toaddr, text)
    # terminating the session
    s.quit()


## Main Loop
for i in range(iterations):
    i += 6 # skip first net
    cfg = set_cfg(i)
    data_transforms = None
    trainloader, testloader = set_data_sets(data_transforms=data_transforms, batch_size=cfg['batch_size'])
    net = init_net(cfg)
    optimizer, scheduler = set_optimizer(cfg, net)
    criterion = tensor_to_gpu(nn.MSELoss(), use_cuda)
    start_time = datetime.datetime.now()
    net, err_vec, err_test_vec, uxuy_err_vec, uxuy_err_test_vec, identity_err_vec = train_net(net, criterion, optimizer, scheduler, trainloader, testloader)
    end_time = datetime.datetime.now()
    fig_path = save_error_plot(cfg, err_vec, err_test_vec, uxuy_err_vec, uxuy_err_test_vec, identity_err_vec, i)
    save_net(net, cfg, err_test_vec[-1], i)
    writelog(i, cfg, err_vec[-1], err_test_vec[-1], uxuy_err_vec[-1], uxuy_err_test_vec[-1], end_time-start_time)
    send_mail('(gpu7) ProjB: iter #%d' % i, fig_path)