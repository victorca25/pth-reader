import torch
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-pretrained', '-p', type=str, required=False, help='Path to pretrained model.')
parser.add_argument('-new', '-n', type=str, required=False, help='Path to new model.')
args = parser.parse_args()


print(args.pretrained)
print(args.new)

if args.pretrained:
    pretrained_net = torch.load(args.pretrained)
    print("Loaded model: " + args.pretrained)

else:
    pretrained_net = torch.load('../../experiments/pretrained_models/RRDB_ESRGAN_x4.pth')
    print("Loaded default RRDB_ESRGAN_x4.pth model")

if args.new:
    crt_net = torch.load(args.new)
    print("Loaded model: " + args.new)

else:
    crt_net = torch.load('../../experiments/pretrained_models/RRDB_ESRGAN_x4.pth')
    print("Loaded default RRDB_ESRGAN_x4.pth model")

layers_pretrain = []

for k, v in pretrained_net.items():
    layers_pretrain.append(k)

if 'model.3.weight' in layers_pretrain:
    if 'model.6.weight' in layers_pretrain:
        if 'model.8.weight' in layers_pretrain:
            if 'model.10.weight' in layers_pretrain:
                print('x4 or x5 scale model')
        elif 'model.9.weight' in layers_pretrain:
            if 'model.11.weight' in layers_pretrain:
                if 'model.13.weight' in layers_pretrain:
                    print('x7 or x8 scale model')
    elif 'model.5.weight' in layers_pretrain:
        if 'model.7.weight' in layers_pretrain:
            print('x2 scale model')
elif 'model.2.weight' in layers_pretrain:
    print('x1 scale model')

