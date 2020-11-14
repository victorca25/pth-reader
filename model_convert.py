import torch
import argparse


# def get_network_description(network):
#     '''Get the string and total parameters of the network'''
#     if isinstance(network, torch.nn.DataParallel):
#         network = network.module
#     s = str(network)
#     n = sum(map(lambda x: x.numel(), network.parameters()))
#     return s, n


# def print_net(network):
#     # Generator
#     s, n = get_network_description(network)
#     if isinstance(network, torch.nn.DataParallel):
#         net_struc_str = '{} - {}'.format(network.__class__.__name__,
#                                             network.module.__class__.__name__)
#     else:
#         net_struc_str = '{}'.format(network.__class__.__name__)

#     print('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
#     print(s)


# def loadnewempty(arch = 'RRDBNet'):
#     if arch == 'RRDBNet':
#         crt_model = arch.RRDBNet(3, 3, 64, 23, gc=32)
#     else:
#         print('TBD')
#         exit()
#     crt_net = crt_model.state_dict()
#    
#     load_net_clean = {}
#     for k, v in state_dict.items():
#         if k.startswith('module.'):
#             load_net_clean[k[7:]] = v
#         else:
#             load_net_clean[k] = v
#     state_dict = load_net_clean
#     return state_dict


def normal2mod(state_dict):
    if 'model.0.weight' in state_dict:
        print('Attempting to convert and load an original RRDB model\n')
        crt_net = {}
        items = []

        for k, v in state_dict.items():
            items.append(k)

        # # directly copy
        # for k, v in crt_net.items():
        #     if k in state_dict and state_dict[k].size() == v.size():
        #         crt_net[k] = state_dict[k]
        #         items.remove(k)

        crt_net['conv_first.weight'] = state_dict['model.0.weight']
        crt_net['conv_first.bias'] = state_dict['model.0.bias']

        for k in items.copy():
            if 'RDB' in k:
                ori_k = k.replace('model.1.sub.', 'RRDB_trunk.')
                if '.0.weight' in k:
                    ori_k = ori_k.replace('.0.weight', '.weight')
                elif '.0.bias' in k:
                    ori_k = ori_k.replace('.0.bias', '.bias')
                crt_net[ori_k] = state_dict[k]
                items.remove(k)

        crt_net['trunk_conv.weight'] = state_dict['model.1.sub.23.weight']
        crt_net['trunk_conv.bias'] = state_dict['model.1.sub.23.bias']
        crt_net['upconv1.weight'] = state_dict['model.3.weight']
        crt_net['upconv1.bias'] = state_dict['model.3.bias']
        crt_net['upconv2.weight'] = state_dict['model.6.weight']
        crt_net['upconv2.bias'] = state_dict['model.6.bias']
        crt_net['HRconv.weight'] = state_dict['model.8.weight']
        crt_net['HRconv.bias'] = state_dict['model.8.bias']
        crt_net['conv_last.weight'] = state_dict['model.10.weight']
        crt_net['conv_last.bias'] = state_dict['model.10.bias']
        state_dict = crt_net

    return state_dict



def mod2normal(state_dict):
    if 'conv_first.weight' in state_dict:
        print('Attempting to convert and load a modified RRDB model')
        crt_net = {}
        items = []
        for k, v in state_dict.items():
            items.append(k)

        crt_net['model.0.weight'] = state_dict['conv_first.weight']
        crt_net['model.0.bias'] = state_dict['conv_first.bias']

        for k in items.copy():
            if 'RDB' in k:
                ori_k = k.replace('RRDB_trunk.', 'model.1.sub.')
                if '.weight' in k:
                    ori_k = ori_k.replace('.weight', '.0.weight')
                elif '.bias' in k:
                    ori_k = ori_k.replace('.bias', '.0.bias')
                crt_net[ori_k] = state_dict[k]
                items.remove(k)

        crt_net['model.1.sub.23.weight'] = state_dict['trunk_conv.weight']
        crt_net['model.1.sub.23.bias'] = state_dict['trunk_conv.bias']
        crt_net['model.3.weight'] = state_dict['upconv1.weight']
        crt_net['model.3.bias'] = state_dict['upconv1.bias']
        crt_net['model.6.weight'] = state_dict['upconv2.weight']
        crt_net['model.6.bias'] = state_dict['upconv2.bias']
        crt_net['model.8.weight'] = state_dict['HRconv.weight']
        crt_net['model.8.bias'] = state_dict['HRconv.bias']
        crt_net['model.10.weight'] = state_dict['conv_last.weight']
        crt_net['model.10.bias'] = state_dict['conv_last.bias']
        state_dict = crt_net
    return state_dict


# def oldread(state_dict):
#     layers = []

#     for k, v in state_dict.items():
#         layers.append(k)

#     if 'model.3.weight' in layers:
#         if 'model.6.weight' in layers:
#             if 'model.8.weight' in layers:
#                 if 'model.10.weight' in layers:
#                     print('x4 or x5 scale model')
#             elif 'model.9.weight' in layers:
#                 if 'model.11.weight' in layers:
#                     if 'model.13.weight' in layers:
#                         print('x7 or x8 scale model')
#         elif 'model.5.weight' in layers:
#             if 'model.7.weight' in layers:
#                 print('x2 scale model')
#     elif 'model.2.weight' in layers:
#         print('x1 scale model')

def newread(state_dict):
    # extract model information
    scale2 = 0
    max_part = 0
    kind = 'ESRGAN'
    scalemin = 6
    
    for part in list(state_dict):
        parts = part.split('.')
        n_parts = len(parts)
        if n_parts == 5 and parts[2] == 'sub':
            nb = int(parts[3])
        elif n_parts == 3:
            part_num = int(parts[1])
            if part_num > scalemin and parts[0] == 'model' and parts[2] == 'weight':
                scale2 += 1
            if part_num > max_part:
                max_part = part_num
                out_nc = state_dict[part].shape[0]
    upscale = 2 ** scale2
    in_nc = state_dict['model.0.weight'].shape[1]
    nf = state_dict['model.0.weight'].shape[0]
    print("X{} upscale factor".format(upscale))

    # model = arch.RRDB_Net(in_nc, out_nc, nf, nb, gc=32, upscale=upscale, norm_type=None, act_type='leakyrelu',
    #                                   mode='CNA', res_scale=1, upsample_mode='upconv')


def save_model(state_dict, save_path="./model.pth"):
    try: #save model in the pre-1.4.0 non-zipped format
        torch.save(state_dict, save_path, _use_new_zipfile_serialization=False)
    except: #pre 1.4.0, normal torch.save
        torch.save(state_dict, save_path)
    print('Saving to ', save_path)


def print_layers(state_dict):
    # layers = []
    for k, v in state_dict.items():
        # layers.append(k)
        print(k)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', '-m', type=str, required=True, help='Path to original model.')
    parser.add_argument('-arch', '-a', type=str, required=False, default='orig', help='Target architecture (orig or mod).')
    parser.add_argument('-dest', '-d', type=str, required=False, help='Path to save converted model.')
    args = parser.parse_args()

    print(args.model)

    if args.model:
        state_dict = torch.load(args.model)
        print("Loaded model: " + args.model)
    # else:
    #     state_dict = torch.load('../../experiments/pretrained_models/RRDB_ESRGAN_x4.pth')
    #     print("Loaded default RRDB_ESRGAN_x4.pth model")

    # print_layers(state_dict)
    if args.arch == 'mod':
        converted_state = normal2mod(state_dict)
    else: #elif args.arch == 'orig':
        converted_state = mod2normal(state_dict)
    # print_layers(converted_state)

    # oldread(converted_state)
    newread(converted_state)

    if args.dest:
        save_path = args.dest
    else:
        save_path="./model.pth"
    save_model(converted_state, save_path)

if __name__ == '__main__':
    main()
