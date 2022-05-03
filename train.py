import os
import torch
import pickle
import csv
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.utils as utils
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from data import Radpath, Radpath_test
from CNN import RNet
#from CNN_glore import RNet
from loss import FocalLoss
from utilities import metrics_all
import argparse

parser = argparse.ArgumentParser(description="Rad")

# batch_size = 10
# num_epochs = 50
# learning_rate = 0.0001
# initialize = "kaimingNormal"
# # fold is not used
# use_weights_in_loss = True
loss_weights = [0.31, 0.32, 0.36]
# use_focal_loss = True
# show_images = True

parser.add_argument("--batch_size", type=int, default=10, help="batch size")
parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--initialize", type=str, default="kaimingNormal", help='kaimingNormal or kaimingUniform or xavierNormal or xavierUniform')
parser.add_argument("--use_weights_in_loss", action='store_true', help='turn on to use weighted loss or else use equal weighted loss')
parser.add_argument("--use_focal_loss", action='store_true', help='turn on focal loss or else use default cross entropy loss')
parser.add_argument("--use_oversampling", action='store_true', help='offline oversampling')
parser.add_argument("--show_images", action='store_true', help='log images of slices for debugging')

parser.add_argument('--log_folder_name', type=str, default='temp', help='name of log file')
opt = parser.parse_args()

exp_params = {'batch size': opt.batch_size, 'num_epochs': opt.num_epochs, 'lr': opt.lr, 'init': opt.initialize,
              'weighted loss': opt.use_weights_in_loss, 'loss_weights': loss_weights, 'focal loss': opt.use_focal_loss,
              'show_images': opt.show_images, 'use_oversampling': opt.use_oversampling, 'log_folder_name': opt.log_folder_name}

# just focusssing on the Radiology images
train_folder = '../../CPM-RadPath_2020_Training_Data/Radiology/'
val_folder = '../../CPM-RadPath_2020_Training_Data/Radiology/'
label_folder = '../../CPM-RadPath_2020_Training_Data/'
log_folder = os.path.join('./log/', opt.log_folder_name)
os.mkdir(log_folder)
# log_folder = os.path.join('./log/', opt.log_folder_name)
# print(log_folder)
current_folder = './'
test_folder = '../../CPM-RadPath_2020_Validation_Data/test/'
test_csv = '../../CPM-RadPath_2020_Validation_Data/test.csv'
label_dict = {'G': 0, 'O': 1, 'A': 2}
inv_label_dict = {0: 'G', 1: 'O', 2: 'A'}

def read_data_mean(trainset):
    pickle_name = os.path.join(current_folder, 'mean')
    try:
        Mean, Std, Max = pickle.load(open(pickle_name, "rb"))
    except OSError as e:
        Mean = torch.zeros(4)
        Std = torch.zeros(4)
        Max = torch.zeros(4)
        kkk = 0
        for i in range(len(trainset)):
            I, L = trainset[i]
            C, D, W, H = I.size()
            Mean += I.view(C, -1).mean(1)
            Std += I.view(C, -1).std(1)
            MM = torch.max(I.view(C, -1), dim=1)[0]
            # verify the variable change
            for j in range(4):
                if MM[j] > Max[j]:
                    Max[j] = MM[j]
            kkk += 1
            print(kkk, end=" ")
        Mean /= len(trainset)
        Std /= len(trainset)
        pickle.dump([Mean, Std, Max], open(pickle_name, "wb"))
    print('\n mean: '), print(Mean.numpy())
    print('std: '), print(Std.numpy())
    print('max: '), print(Max.numpy())
    return Mean, Std, Max

def train(log_file):
    print('\n loading the data... \n')

    if opt.use_oversampling:
        print('\n Minority class oversampled offline \n')
        train_file = label_folder+'train_oversample_shuffle.csv'
    else:
        print('\n No oversampling... \n')
        train_file = label_folder+'train.csv'

    val_file = label_folder+'val.csv'

    # no augmentation yet
    trainset = Radpath(csv_file=train_file, data_path=train_folder, shuffle=True)
    valset = Radpath(csv_file=val_file, data_path=val_folder, shuffle=False)
    testset = Radpath_test(csv_file=test_csv, data_path=test_folder)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    Mean, Std, Max = read_data_mean(trainset)
    print('\ndone reading data mean std max\n')

    # load network
    print('\nloading the model ...\n')
    net = RNet(in_features=4, num_class=3, init=opt.initialize)
    print(net)
    print('\n net done\n')

    # move to GPU
    print('/n moving models to GPU... \n')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, 'chosen')
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).to(device)


    # Loss
    if opt.use_weights_in_loss:
        weighted_loss = torch.tensor(loss_weights).to(device)
        print('using weighted loss with weights- ', loss_weights)
    else:
        weighted_loss = None
        print('using no loss weights')

    if opt.use_focal_loss:
        print('\n using focal loss... \n')
        criterion = FocalLoss(gama=2., size_average=True, weight=weighted_loss)
    else:
        print('\n using normal crossentropy loss... \n')
        criterion = nn.CrossEntropyLoss(weight=weighted_loss)

    criterion.to(device)
    print('\n loss code done')


    # optimizer
    # Try using ADAM optimizer?
    lr_lambda = lambda epoch: np.power(0.5, epoch//10)
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4, nesterov=False)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # training
    print('\n starting the training... \n')
    step = 0
    running_avg_accuracy = 0
    writer = SummaryWriter(os.path.join(log_folder, 'tensorboard_RNet'))

    # epoch_loss = []
    # epoch_acc = []

    for epoch in range(opt.num_epochs):
        print("\n epoch number %d learning rate %f" % (epoch, optimizer.param_groups[0]['lr']))
        for i, data in enumerate(trainloader, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            inputs, labels = data
            inputs = (inputs - Mean.view(1, 4, 1, 1, 1))/Std.view(1, 4, 1, 1, 1)
            inputs, labels = inputs.to(device), labels.to(device)

            # forward pass
            pred = model.forward(inputs)
            # backward pass
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            model.eval()
            pred = model.forward(inputs)
            predict = torch.argmax(pred, 1)
            total = labels.size(0)
            correct = torch.eq(predict, labels).sum().double().item()
            accuracy = correct/total
            running_avg_accuracy = 0.9*running_avg_accuracy + 0.1*accuracy

            # write to tensorboard
            writer.add_scalar('train/loss', loss.item(), step)
            writer.add_scalar('train/accuracy', accuracy, step)
            writer.add_scalar('train/running_avg_accuracy', running_avg_accuracy, step)
            print("[epoch %d/%d][%d/%d] loss %.4f accuracy %.2f%% running avg accuracy %.2f%%"
                  % (epoch, opt.num_epochs, i, len(trainloader)-1, loss.item(), (100 * accuracy),
                     (100 * running_avg_accuracy)))
            step += 1

        # save model checkpoints
        print('\n one epoch done, saving checkpoints ...\n')
        torch.save(model.state_dict(), os.path.join(log_folder, 'net.pth'))
        # if epoch == opt.epochs / 2:
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(log_folder, 'net_{}.pth'.format(epoch)))

        # validation phase
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            val_pred_csv = os.path.join(log_folder, 'val_pred.csv')
            with open(val_pred_csv, 'wt', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',')
                for i, data in enumerate(valloader, 0):
                    images_val, labels_val = data
                    images_val = (images_val-Mean.view(1, 4, 1, 1, 1))/Std.view(1, 4, 1, 1, 1)
                    images_val, labels_val = images_val.to(device), labels_val.to(device)
                    pred_val = model.forward(images_val)
                    predict = torch.argmax(pred_val, 1)
                    total += labels_val.size(0)
                    correct += torch.eq(predict, labels_val).sum().double().item()

                    # record val prediction responses
                    responses = F.softmax(pred_val, dim=1).squeeze().cpu().numpy()
                    responses = [responses[j] for j in range(responses.shape[0])]
                    csv_writer.writerow(responses)

            #precision, recall, f1, auroc, cm = metrics_all(val_file, val_pred_csv)
            precision, recall, f1, cm = metrics_all(val_file, val_pred_csv)
            writer.add_scalar('val/accuracy', correct / total, epoch)
            writer.add_scalar('val/avg_precision', np.mean(precision), epoch)
            writer.add_scalar('val/avg_recall', np.mean(recall), epoch)
            print("\n[epoch %d] val result: accuracy %.2f%% \navg_precision %.2f%% avg_recall %.2f%%\n" %
                  (epoch, 100 * correct / total, 100 * np.mean(precision), 100 * np.mean(recall)))
            print('precision:', precision)
            print('recall:', recall)
            print('confusion matrix:', cm)
            print('f1:', f1)

            if epoch+1 == opt.num_epochs:
                val_results = dict()
                val_results['precision'] = precision.tolist()
                val_results['recall'] = recall.tolist()
                val_results['cm'] = cm.tolist()
                val_results['f1'] = f1
                #val_results['auroc'] = auroc
                val_results['acc'] = correct / total
                with open(os.path.join(log_folder, 'val_metrics_output.json'), 'w') as js:
                        json.dump(val_results, js)

                # log_file.write('Fold {} cross validation\n'.format(fold))
                log_file.write(str(100 * correct / total)), log_file.write('\n\n')  # accuracy
                np.savetxt(log_file, precision), log_file.write('\n')  # precision
                np.savetxt(log_file, recall), log_file.write('\n')  # recall
                np.savetxt(log_file, cm), log_file.write('\n')
                log_file.write(str(f1)), log_file.write('\n')
                #log_file.write(str(auroc)), log_file.write('\n')
                # sensitivity, specificity, confusion matrix
                log_file.write('\n')

        # display images 
        if opt.show_images:
            I_T1 = utils.make_grid(inputs[:, 0, 64, :, :].unsqueeze(1), nrow=4, normalize=True, scale_each=True)
            I_T1Gd = utils.make_grid(inputs[:, 1, 64, :, :].unsqueeze(1), nrow=4, normalize=True, scale_each=True)
            I_T2 = utils.make_grid(inputs[:, 2, 64, :, :].unsqueeze(1), nrow=4, normalize=True, scale_each=True)
            I_FLAIR = utils.make_grid(inputs[:, 3, 64, :, :].unsqueeze(1), nrow=4, normalize=True, scale_each=True)
            writer.add_image('Image/T1', I_T1, epoch)
            writer.add_image('Image/T1Gd', I_T1Gd, epoch)
            writer.add_image('Image/T2', I_T2, epoch)
            writer.add_image('Image/FLAIR', I_FLAIR, epoch)


        # test
        if epoch+1 == opt.num_epochs:
            model.eval()
            with torch.no_grad():
                test_pred_csv = os.path.join(log_folder, 'test_pred.csv')
                with open(test_pred_csv, 'wt', newline='') as csv_file2:
                    csv_writer2 = csv.writer(csv_file2, delimiter=',')
                    #for i, data in enumerate(testloader, 0):
                    for data in testloader:
                        test_image, dataID = data
                        test_image = (test_image-Mean.view(1, 4, 1, 1, 1))/Std.view(1, 4, 1, 1, 1)
                        test_image.to(device)
                        pred_test = model.forward(test_image)
                        predict_test = torch.argmax(pred_test, 1)
                        predict_label = inv_label_dict[predict_test.item()]
                        
                        # record predictions for uploading
                        output = [dataID[0], predict_label]
                        csv_writer2.writerow(output)


        # adjust learning rate
        scheduler.step()


def main():
    log_file = open(os.path.join(log_folder, "log_file.txt"), "w")
    train(log_file)
    log_file.close()
    with open(os.path.join(log_folder, 'exp_parameters.json'), 'w') as js:
        json.dump(exp_params, js, indent=2)

if __name__=="__main__":
    main()
