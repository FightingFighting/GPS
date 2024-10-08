import numpy as np
import torch
import torch.nn as nn




def prune_by_percentile_gradient_perCell(model, time_para=1):
    statistic = {}
    new_masks = {}

    for name, param in model.named_parameters():
        if "norm" in name or "pos_embed" in name or "cls_token" in name:
            new_mask = np.ones_like(param.data.cpu().numpy())
        elif 'head' in name or "bias" in name or "gamma" in name:
            new_mask = np.zeros_like(param.data.cpu().numpy())
        else:
            if "patch_embed" in name or "conv" in name or "stem.proj.weight" in name or "downsample.proj.weight" in name:
                tensor = param.grad.data.cpu().numpy()
                B,C,H,W = tensor.shape
                tensor = np.reshape(tensor,[B,-1])
            else:
                tensor = param.grad.data.cpu().numpy()

            new_mask=np.ones_like(tensor)
            for ind in range(time_para):
                max_index = abs(tensor).argsort(1)[:, -(ind + 1)]
                one_hot_temp = ~np.eye(max(tensor.shape))[max_index][:, :tensor.shape[1]].astype(np.bool)
                new_mask_temp = one_hot_temp.astype(np.float32)
                new_mask = new_mask.astype(int) & new_mask_temp.astype(int)
                new_mask = new_mask.astype(np.float32)
                # print(f"88888888888888888888888 time_para: {time_para} 888888888888888888888888888")


            if "patch_embed" in name or "conv" in name or "stem.proj.weight" in name or "downsample.proj.weight" in name:
                new_mask = np.reshape(new_mask, (B,C,H,W))

        trainable_param = len(new_mask.reshape(-1))-len(np.nonzero(new_mask)[0])
        total_para = len(new_mask.reshape(-1))
        statistic[name]=[trainable_param, total_para]
        print(name, ": ", trainable_param, "/", total_para, "(",np.round((trainable_param/total_para)*100, 4), "%)", new_mask.shape   )


        new_masks[name] = torch.from_numpy(new_mask).cuda()


    print("---------------------------------------------------------------")
    trainable_withouthead = 0
    total_withouthead = 0
    trainable_head = 0
    total_head = 0
    for na, [trainable_p, t_p] in statistic.items():
        # print(na, ":", trainable_p, "/", t_p,"(",np.round((trainable_p/t_p)*100, 4), "%)")
        if "head" not in na:
            trainable_withouthead = trainable_withouthead + trainable_p
            total_withouthead = total_withouthead + t_p
        else:
            trainable_head = trainable_head + trainable_p
            total_head = total_head + t_p
    print("---------------------------------------------------------------")

    print("---------------------------------------------------------------")
    print("Trainable parameter / Total (without head): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4), "%)")
    print("Trainable parameter / Total (head): ", trainable_head, "/", total_head, "(", np.round((trainable_head/total_head)*100,4), "%)")
    print("Trainable parameter / Total (total): ", trainable_head+trainable_withouthead, "/", total_head+total_withouthead, "(", np.round(((trainable_head+trainable_withouthead)/(total_head+total_withouthead))*100,4), "%)")

    print("#######################################################################")
    return new_masks

def prune_by_percentile_gradient_perCell_magnitude(model, time_para=1):
    statistic = {}
    new_masks = {}

    for name, param in model.named_parameters():
        if "norm" in name or "pos_embed" in name or "cls_token" in name:
            new_mask = np.ones_like(param.data.cpu().numpy())
        elif 'head' in name or "bias" in name or "gamma" in name:
            new_mask = np.zeros_like(param.data.cpu().numpy())
        else:
            if "patch_embed" in name or "conv" in name or "stem.proj.weight" in name or "downsample.proj.weight" in name:
                tensor = param.data.cpu().numpy()
                B,C,H,W = tensor.shape
                tensor = np.reshape(tensor,[B,-1])
            else:
                tensor = param.data.cpu().numpy()

            new_mask=np.ones_like(tensor)
            for ind in range(time_para):
                max_index = abs(tensor).argsort(1)[:, -(ind + 1)]
                one_hot_temp = ~np.eye(max(tensor.shape))[max_index][:, :tensor.shape[1]].astype(np.bool)
                new_mask_temp = one_hot_temp.astype(np.float32)
                new_mask = new_mask.astype(int) & new_mask_temp.astype(int)
                new_mask = new_mask.astype(np.float32)
                # print(f"88888888888888888888888 time_para: {time_para} 888888888888888888888888888")


            if "patch_embed" in name or "conv" in name or "stem.proj.weight" in name or "downsample.proj.weight" in name:
                new_mask = np.reshape(new_mask, (B,C,H,W))

        trainable_param = len(new_mask.reshape(-1))-len(np.nonzero(new_mask)[0])
        total_para = len(new_mask.reshape(-1))
        statistic[name]=[trainable_param, total_para]
        print(name, ": ", trainable_param, "/", total_para, "(",np.round((trainable_param/total_para)*100, 4), "%)", new_mask.shape   )


        new_masks[name] = torch.from_numpy(new_mask).cuda()


    print("---------------------------------------------------------------")
    trainable_withouthead = 0
    total_withouthead = 0
    trainable_head = 0
    total_head = 0
    for na, [trainable_p, t_p] in statistic.items():
        # print(na, ":", trainable_p, "/", t_p,"(",np.round((trainable_p/t_p)*100, 4), "%)")
        if "head" not in na:
            trainable_withouthead = trainable_withouthead + trainable_p
            total_withouthead = total_withouthead + t_p
        else:
            trainable_head = trainable_head + trainable_p
            total_head = total_head + t_p
    print("---------------------------------------------------------------")

    print("---------------------------------------------------------------")
    print("Trainable parameter / Total (without head): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4), "%)")
    print("Trainable parameter / Total (head): ", trainable_head, "/", total_head, "(", np.round((trainable_head/total_head)*100,4), "%)")
    print("Trainable parameter / Total (total): ", trainable_head+trainable_withouthead, "/", total_head+total_withouthead, "(", np.round(((trainable_head+trainable_withouthead)/(total_head+total_withouthead))*100,4), "%)")

    print("#######################################################################")
    return new_masks

def prune_by_random_perCell(model, time_para=1):
    statistic = {}
    new_masks = {}

    for name, param in model.named_parameters():
        if "norm" in name or "pos_embed" in name or "cls_token" in name:
            new_mask = np.ones_like(param.data.cpu().numpy())
        elif 'head' in name or "bias" in name or "gamma" in name:
            new_mask = np.zeros_like(param.data.cpu().numpy())
        else:
            if "patch_embed" in name or "conv" in name or "stem.proj.weight" in name or "downsample.proj.weight" in name:
                tensor = param.grad.data.cpu().numpy()
                B,C,H,W = tensor.shape
                tensor = np.reshape(tensor,[B,-1])
            else:
                tensor = param.grad.data.cpu().numpy()

            new_mask=np.ones_like(tensor)
            for ind in range(time_para):
                max_index = abs(tensor).argsort(1)[:, -(ind + 1)]
                one_hot_temp = ~np.eye(max(tensor.shape))[max_index][:, :tensor.shape[1]].astype(np.bool)
                new_mask_temp = one_hot_temp.astype(np.float32)
                new_mask = new_mask.astype(int) & new_mask_temp.astype(int)
                new_mask = new_mask.astype(np.float32)
                # print(f"88888888888888888888888 time_para: {time_para} 888888888888888888888888888")


            if "patch_embed" in name or "conv" in name or "stem.proj.weight" in name or "downsample.proj.weight" in name:
                new_mask = np.reshape(new_mask, (B,C,H,W))

            np.random.shuffle(new_mask)

        trainable_param = len(new_mask.reshape(-1))-len(np.nonzero(new_mask)[0])
        total_para = len(new_mask.reshape(-1))
        statistic[name]=[trainable_param, total_para]
        print(name, ": ", trainable_param, "/", total_para, "(",np.round((trainable_param/total_para)*100, 4), "%)", new_mask.shape   )


        new_masks[name] = torch.from_numpy(new_mask).cuda()


    print("---------------------------------------------------------------")
    trainable_withouthead = 0
    total_withouthead = 0
    trainable_head = 0
    total_head = 0
    for na, [trainable_p, t_p] in statistic.items():
        # print(na, ":", trainable_p, "/", t_p,"(",np.round((trainable_p/t_p)*100, 4), "%)")
        if "head" not in na:
            trainable_withouthead = trainable_withouthead + trainable_p
            total_withouthead = total_withouthead + t_p
        else:
            trainable_head = trainable_head + trainable_p
            total_head = total_head + t_p
    print("---------------------------------------------------------------")

    print("---------------------------------------------------------------")
    print("Trainable parameter / Total (without head): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4), "%)")
    print("Trainable parameter / Total (head): ", trainable_head, "/", total_head, "(", np.round((trainable_head/total_head)*100,4), "%)")
    print("Trainable parameter / Total (total): ", trainable_head+trainable_withouthead, "/", total_head+total_withouthead, "(", np.round(((trainable_head+trainable_withouthead)/(total_head+total_withouthead))*100,4), "%)")

    print("#######################################################################")
    return new_masks


def prune_by_percentile_gradient_perCell_part(model, time_para=1, block_num=None):
    statistic = {}
    new_masks = {}

    for name, param in model.named_parameters():
        new_mask = []
        if "blocks" in name:
            for bn in range(block_num+1):
                if f"blocks.{str(bn)}" in name:
                    new_mask = np.ones_like(param.data.cpu().numpy())
        if new_mask == []:
            if "norm" in name or "pos_embed" in name or "cls_token" in name or "patch_embed" in name:
                new_mask = np.ones_like(param.data.cpu().numpy())
            elif 'head' in name or "bias" in name or "gamma" in name:
                new_mask = np.zeros_like(param.data.cpu().numpy())
            else:
                if "patch_embed" in name or "conv" in name or "stem.proj.weight" in name or "downsample.proj.weight" in name:
                    tensor = param.grad.data.cpu().numpy()
                    B,C,H,W = tensor.shape
                    tensor = np.reshape(tensor,[B,-1])
                else:
                    tensor = param.grad.data.cpu().numpy()

                new_mask=np.ones_like(tensor)
                for ind in range(time_para):
                    max_index = abs(tensor).argsort(1)[:, -(ind + 1)]
                    one_hot_temp = ~np.eye(max(tensor.shape))[max_index][:, :tensor.shape[1]].astype(np.bool)
                    new_mask_temp = one_hot_temp.astype(np.float32)
                    new_mask = new_mask.astype(int) & new_mask_temp.astype(int)
                    new_mask = new_mask.astype(np.float32)
                    # print(f"88888888888888888888888 time_para: {time_para} 888888888888888888888888888")


                if "patch_embed" in name or "conv" in name or "stem.proj.weight" in name or "downsample.proj.weight" in name:
                    new_mask = np.reshape(new_mask, (B,C,H,W))

        trainable_param = len(new_mask.reshape(-1))-len(np.nonzero(new_mask)[0])
        total_para = len(new_mask.reshape(-1))
        statistic[name]=[trainable_param, total_para]
        print(name, ": ", trainable_param, "/", total_para, "(",np.round((trainable_param/total_para)*100, 4), "%)", new_mask.shape   )

        new_masks[name] = torch.from_numpy(new_mask).cuda()


    print("---------------------------------------------------------------")
    trainable_withouthead = 0
    total_withouthead = 0
    trainable_head = 0
    total_head = 0
    for na, [trainable_p, t_p] in statistic.items():
        # print(na, ":", trainable_p, "/", t_p,"(",np.round((trainable_p/t_p)*100, 4), "%)")
        if "head" not in na:
            trainable_withouthead = trainable_withouthead + trainable_p
            total_withouthead = total_withouthead + t_p
        else:
            trainable_head = trainable_head + trainable_p
            total_head = total_head + t_p
    print("---------------------------------------------------------------")

    print("---------------------------------------------------------------")
    print("Trainable parameter / Total (without head): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4), "%)")
    print("Trainable parameter / Total (head): ", trainable_head, "/", total_head, "(", np.round((trainable_head/total_head)*100,4), "%)")
    print("Trainable parameter / Total (total): ", trainable_head+trainable_withouthead, "/", total_head+total_withouthead, "(", np.round(((trainable_head+trainable_withouthead)/(total_head+total_withouthead))*100,4), "%)")

    print("#######################################################################")
    return new_masks

# def prune_by_percentile_gradient_perCell_part(model, time_para=1, block_num=None):
#     statistic = {}
#     new_masks = {}
#
#     for name, param in model.named_parameters():
#         new_mask = []
#
#         if f"blocks.{str(block_num)}" in name:
#             new_mask = np.ones_like(param.data.cpu().numpy())
#
#         if new_mask == []:
#             if "norm" in name or "pos_embed" in name or "cls_token" in name or "patch_embed" in name:
#                 new_mask = np.ones_like(param.data.cpu().numpy())
#             elif 'head' in name or "bias" in name or "gamma" in name:
#                 new_mask = np.zeros_like(param.data.cpu().numpy())
#             else:
#                 if "patch_embed" in name or "conv" in name or "stem.proj.weight" in name or "downsample.proj.weight" in name:
#                     tensor = param.grad.data.cpu().numpy()
#                     B,C,H,W = tensor.shape
#                     tensor = np.reshape(tensor,[B,-1])
#                 else:
#                     tensor = param.grad.data.cpu().numpy()
#
#                 new_mask=np.ones_like(tensor)
#                 for ind in range(time_para):
#                     max_index = abs(tensor).argsort(1)[:, -(ind + 1)]
#                     one_hot_temp = ~np.eye(max(tensor.shape))[max_index][:, :tensor.shape[1]].astype(np.bool)
#                     new_mask_temp = one_hot_temp.astype(np.float32)
#                     new_mask = new_mask.astype(int) & new_mask_temp.astype(int)
#                     new_mask = new_mask.astype(np.float32)
#                     # print(f"88888888888888888888888 time_para: {time_para} 888888888888888888888888888")
#
#
#                 if "patch_embed" in name or "conv" in name or "stem.proj.weight" in name or "downsample.proj.weight" in name:
#                     new_mask = np.reshape(new_mask, (B,C,H,W))
#
#         trainable_param = len(new_mask.reshape(-1))-len(np.nonzero(new_mask)[0])
#         total_para = len(new_mask.reshape(-1))
#         statistic[name]=[trainable_param, total_para]
#         print(name, ": ", trainable_param, "/", total_para, "(",np.round((trainable_param/total_para)*100, 4), "%)", new_mask.shape   )
#
#         new_masks[name] = torch.from_numpy(new_mask).cuda()
#
#
#     print("---------------------------------------------------------------")
#     trainable_withouthead = 0
#     total_withouthead = 0
#     trainable_head = 0
#     total_head = 0
#     for na, [trainable_p, t_p] in statistic.items():
#         # print(na, ":", trainable_p, "/", t_p,"(",np.round((trainable_p/t_p)*100, 4), "%)")
#         if "head" not in na:
#             trainable_withouthead = trainable_withouthead + trainable_p
#             total_withouthead = total_withouthead + t_p
#         else:
#             trainable_head = trainable_head + trainable_p
#             total_head = total_head + t_p
#     print("---------------------------------------------------------------")
#
#     print("---------------------------------------------------------------")
#     print("Trainable parameter / Total (without head): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4), "%)")
#     print("Trainable parameter / Total (head): ", trainable_head, "/", total_head, "(", np.round((trainable_head/total_head)*100,4), "%)")
#     print("Trainable parameter / Total (total): ", trainable_head+trainable_withouthead, "/", total_head+total_withouthead, "(", np.round(((trainable_head+trainable_withouthead)/(total_head+total_withouthead))*100,4), "%)")
#
#     print("#######################################################################")
#     return new_masks

def prune_by_percentile_gradient_perLayer(model, percent_pruning_min, percent_pruning_max):
    statistic = {}
    new_masks = {}

    for name, param in model.named_parameters():
        if "norm" in name or "pos_embed" in name or "cls_token" in name:
            new_mask = np.ones_like(param.data.cpu().numpy())
        elif 'head' in name or "bias" in name or "gamma" in name:
            new_mask = np.zeros_like(param.data.cpu().numpy())
        else:
            tensor = param.grad.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values

            percentile_value_min = np.percentile(abs(alive), percent_pruning_min)
            percentile_value_max = np.percentile(abs(alive), percent_pruning_max)

            old_mask = np.ones_like(param.data.cpu().numpy())
            new_mask = np.where((abs(tensor) > percentile_value_min) & (abs(tensor) <= percentile_value_max), 0, old_mask)

        trainable_param = len(new_mask.reshape(-1))-len(np.nonzero(new_mask)[0])
        total_para = len(new_mask.reshape(-1))
        statistic[name]=[trainable_param, total_para]
        print(name, ": ", trainable_param, "/", total_para, "(",np.round((trainable_param/total_para)*100, 4), "%)", new_mask.shape   )


        new_masks[name] = torch.from_numpy(new_mask).cuda()


    print("---------------------------------------------------------------")
    trainable_withouthead = 0
    total_withouthead = 0
    trainable_head = 0
    total_head = 0
    for na, [trainable_p, t_p] in statistic.items():
        # print(na, ":", trainable_p, "/", t_p,"(",np.round((trainable_p/t_p)*100, 4), "%)")
        if "head" not in na:
            trainable_withouthead = trainable_withouthead + trainable_p
            total_withouthead = total_withouthead + t_p
        else:
            trainable_head = trainable_head + trainable_p
            total_head = total_head + t_p
    print("---------------------------------------------------------------")

    print("---------------------------------------------------------------")
    print("Trainable parameter / Total (without head): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4), "%)")
    print("Trainable parameter / Total (head): ", trainable_head, "/", total_head, "(", np.round((trainable_head/total_head)*100,4), "%)")
    print("Trainable parameter / Total (total): ", trainable_head+trainable_withouthead, "/", total_head+total_withouthead, "(", np.round(((trainable_head+trainable_withouthead)/(total_head+total_withouthead))*100,4), "%)")

    print("#######################################################################")
    return new_masks

def prune_by_percentile_gradient_allLayer(model, percent_pruning_min, percent_pruning_max):
    # Calculate percentile value
    alive_all = np.array([])
    for name, param in model.named_parameters():
        if "head" in name: continue
        tensor = param.grad.data.cpu().numpy()
        alive = tensor[np.nonzero(tensor)]  # flattened array of nonzero values
        alive_all = np.concatenate([alive_all, alive])

    percentile_value_min = np.percentile(abs(alive_all), percent_pruning_min)
    percentile_value_max = np.percentile(abs(alive_all), percent_pruning_max)


    statistic = {}
    new_masks = {}

    for name, param in model.named_parameters():
        if "norm" in name or "pos_embed" in name or "cls_token" in name:
            new_mask = np.ones_like(param.data.cpu().numpy())
        elif 'head' in name or "bias" in name or "gamma" in name:
            new_mask = np.zeros_like(param.data.cpu().numpy())
        else:
            tensor = param.grad.data.cpu().numpy()
            old_mask = np.ones_like(param.data.cpu().numpy())
            new_mask = np.where((abs(tensor) > percentile_value_min) & (abs(tensor) <= percentile_value_max), 0, old_mask)

        trainable_param = len(new_mask.reshape(-1))-len(np.nonzero(new_mask)[0])
        total_para = len(new_mask.reshape(-1))
        statistic[name]=[trainable_param, total_para]
        print(name, ": ", trainable_param, "/", total_para, "(",np.round((trainable_param/total_para)*100, 4), "%)", new_mask.shape   )


        new_masks[name] = torch.from_numpy(new_mask).cuda()


    print("---------------------------------------------------------------")
    trainable_withouthead = 0
    total_withouthead = 0
    trainable_head = 0
    total_head = 0
    for na, [trainable_p, t_p] in statistic.items():
        # print(na, ":", trainable_p, "/", t_p,"(",np.round((trainable_p/t_p)*100, 4), "%)")
        if "head" not in na:
            trainable_withouthead = trainable_withouthead + trainable_p
            total_withouthead = total_withouthead + t_p
        else:
            trainable_head = trainable_head + trainable_p
            total_head = total_head + t_p
    print("---------------------------------------------------------------")

    print("---------------------------------------------------------------")
    print("Trainable parameter / Total (without head): ", trainable_withouthead, "/", total_withouthead, "(", np.round((trainable_withouthead/total_withouthead)*100,4), "%)")
    print("Trainable parameter / Total (head): ", trainable_head, "/", total_head, "(", np.round((trainable_head/total_head)*100,4), "%)")
    print("Trainable parameter / Total (total): ", trainable_head+trainable_withouthead, "/", total_head+total_withouthead, "(", np.round(((trainable_head+trainable_withouthead)/(total_head+total_withouthead))*100,4), "%)")

    print("#######################################################################")
    return new_masks




def make_mask(model):
    masks={}
    for name, param in model.named_parameters():
        masks[name] = np.ones_like(param.data.cpu().numpy())
    return masks
