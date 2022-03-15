from EIVideo.eivideo.utils.log import Logging

import os
import timeit
import paddle
from PIL import Image
from davisinteractive.utils.scribbles import scribbles2mask, annotated_frames
from paddle import nn

from EIVideo.eivideo.utils.manet import load
from EIVideo.eivideo.utils.manet import float_, _palette, write_dict, rough_ROI
from EIVideo.eivideo.utils.manet import load_video, get_scribbles, submit_masks
from EIVideo.eivideo.utils.manet import Resize_manet, ToTensor_manet, ToTensor_manet2


class Manet(nn.Layer):
    def __init__(self, backbone=None, head=None, **cfg):
        super().__init__()

    def forward(self, data_batch, mode='infer', **kwargs):
        """
        1. Define how the model is going to run, from input to output.
        2. Console of train, valid, test or infer step
        3. Set mode='infer' is used for saving inference model, refer to tools/export_model.py
        """
        return self.test_step(data_batch, **kwargs)

    def test_step(self, weights, parallel=True, is_save_image=True, json_scribbles=None, video_path=None, **cfg):
        # 1. Construct model.
        # if parallel:
        #     model = paddle.DataParallel(model)

        from ..models.manet.networks import DeepLab
        from ..models.manet.networks import IntVOS
        feature_extracter = DeepLab(backbone='resnet', freeze_bn=False)
        model = IntVOS(feature_extracter, **cfg['HEAD'])

        # 2. Construct data.
        sequence = video_path.split('/')[-1].split('.')[0]
        output_dir = cfg['TEST']['output_dir']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        obj_nums = 1
        video_path = "data/uploads/video/" + video_path
        from EIVideo import join_root_path
        video_path = join_root_path(video_path)
        Logging.debug(video_path)
        images, _ = load_video(video_path, 480)
        Logging.debug("stage1 load_video success")
        # [195, 389, 238, 47, 244, 374, 175, 399]
        # .shape: (502, 480, 600, 3)
            # Configuration used in the challenges
        max_nb_interactions = 8  # Maximum number of interactions
        # Interactive parameters

        state_dicts_ = load(weights)['state_dict']
        state_dicts = {}
        for k, v in state_dicts_.items():
            if 'num_batches_tracked' not in k:
                state_dicts[k] = v
                if (k) not in model.state_dict().keys():
                    Logging.info(f'pretrained -----{k} -------is not in model')
                    pass
        write_dict(state_dicts, 'model_for_infer.txt', **cfg)
        model.set_state_dict(state_dicts)

        model.eval()
        seen_seq = False

        with paddle.no_grad():

            # Get the current iteration scribbles
            for scribbles, first_scribble in get_scribbles(json_scribbles):
                t_total = timeit.default_timer()
                f, h, w = images.shape[:3]
                if 'prev_label_storage' not in locals().keys():
                    prev_label_storage = paddle.zeros([f, h, w])
                if len(annotated_frames(scribbles)) == 0:
                    final_masks = prev_label_storage
                    submit_masks(cfg['TEST']["output_dir"], final_masks.numpy(), images)
                    continue

                # if no scribbles return, keep masks in previous round
                start_annotated_frame = annotated_frames(scribbles)[0]
                pred_masks = []
                pred_masks_reverse = []

                if first_scribble:  # If in the first round, initialize memories
                    n_interaction = 1
                    eval_global_map_tmp_dic = {}
                    local_map_dics = ({}, {})
                    total_frame_num = f

                else:
                    n_interaction += 1

                if first_scribble:  # if in the first round, extract pixel embbedings.
                    if not seen_seq:
                        seen_seq = True
                        inter_turn = 1
                        embedding_memory = []

                        list = []
                        list.append(int(cfg['TEST']['output_high']))
                        list.append(int(cfg['TEST']['output_width']))
                        for imgs in images:
                            if cfg['TEST']:
                                imgs = Resize_manet(list)(imgs)
                                imgs = ToTensor_manet()(imgs)
                            else:
                                imgs = paddle.to_tensor([imgs])

                            if parallel:
                                for c in model.children():
                                    frame_embedding = c.extract_feature(
                                        imgs)
                            else:
                                frame_embedding = model.extract_feature(
                                    imgs)
                            embedding_memory.append(frame_embedding)

                        del frame_embedding

                        embedding_memory = paddle.concat(embedding_memory, 0)
                        _, _, emb_h, emb_w = embedding_memory.shape
                        ref_frame_embedding = embedding_memory[
                            start_annotated_frame]
                        ref_frame_embedding = ref_frame_embedding.unsqueeze(0)
                    else:
                        inter_turn += 1
                        ref_frame_embedding = embedding_memory[
                            start_annotated_frame]
                        ref_frame_embedding = ref_frame_embedding.unsqueeze(0)

                else:
                    ref_frame_embedding = embedding_memory[
                        start_annotated_frame]
                    ref_frame_embedding = ref_frame_embedding.unsqueeze(0)
                ########
                scribble_masks = scribbles2mask(scribbles, (emb_h, emb_w))
                scribble_label = scribble_masks[start_annotated_frame]
                scribble_sample = {'scribble_label': scribble_label}

                scribble_sample = ToTensor_manet2()(scribble_sample)
                #                     print(ref_frame_embedding, ref_frame_embedding.shape)
                scribble_label = scribble_sample['scribble_label']

                scribble_label = scribble_label.unsqueeze(0)
                model_name = cfg['MODEL']['model_name']
                output_dir = os.path.join(cfg['TEST']['output_dir'], cfg['MODEL']['model_name'])
                inter_file_path = os.path.join(
                    output_dir, sequence, 'interactive' + str(n_interaction),
                                          'turn' + str(inter_turn))
                if is_save_image:
                    ref_scribble_to_show = scribble_label.squeeze().numpy()
                    im_ = Image.fromarray(
                        ref_scribble_to_show.astype('uint8')).convert('P', )
                    im_.putpalette(_palette)
                    ref_img_name = str(start_annotated_frame)

                    if not os.path.exists(inter_file_path):
                        os.makedirs(inter_file_path)
                    im_.save(
                        os.path.join(inter_file_path,
                                     'inter_' + ref_img_name + '.png'))
                if first_scribble:
                    prev_label = None
                    prev_label_storage = paddle.zeros([f, h, w])
                else:
                    prev_label = prev_label_storage[start_annotated_frame]
                    prev_label = prev_label.unsqueeze(0).unsqueeze(0)
                # check if no scribbles.
                if not first_scribble and paddle.unique(
                        scribble_label).shape[0] == 1:
                    print(
                        'not first_scribble and paddle.unique(scribble_label).shape[0] == 1'
                    )
                    final_masks = prev_label_storage
                    submit_masks(cfg['MODEL']['model_name'], final_masks.numpy(), images)
                    # submit_masks(cfg["save_path"], final_masks.numpy(), images)
                    continue

                ###inteaction segmentation head
                if parallel:
                    for c in model.children():
                        tmp_dic, local_map_dics = c.int_seghead(
                            ref_frame_embedding=ref_frame_embedding,
                            ref_scribble_label=scribble_label,
                            prev_round_label=prev_label,
                            global_map_tmp_dic=eval_global_map_tmp_dic,
                            local_map_dics=local_map_dics,
                            interaction_num=n_interaction,
                            seq_names=[sequence],
                            gt_ids=paddle.to_tensor([obj_nums]),
                            frame_num=[start_annotated_frame],
                            first_inter=first_scribble)
                else:
                    tmp_dic, local_map_dics = model.int_seghead(
                        ref_frame_embedding=ref_frame_embedding,
                        ref_scribble_label=scribble_label,
                        prev_round_label=prev_label,
                        global_map_tmp_dic=eval_global_map_tmp_dic,
                        local_map_dics=local_map_dics,
                        interaction_num=n_interaction,
                        seq_names=[sequence],
                        gt_ids=paddle.to_tensor([obj_nums]),
                        frame_num=[start_annotated_frame],
                        first_inter=first_scribble)
                pred_label = tmp_dic[sequence]
                pred_label = nn.functional.interpolate(pred_label,
                                                       size=(h, w),
                                                       mode='bilinear',
                                                       align_corners=True)
                pred_label = paddle.argmax(pred_label, axis=1)
                pred_masks.append(float_(pred_label))
                # np.unique(pred_label)
                # array([0], dtype=int64)
                prev_label_storage[start_annotated_frame] = float_(
                    pred_label[0])

                if is_save_image:  # save image
                    pred_label_to_save = pred_label.squeeze(0).numpy()
                    im = Image.fromarray(
                        pred_label_to_save.astype('uint8')).convert('P', )
                    im.putpalette(_palette)
                    imgname = str(start_annotated_frame)
                    while len(imgname) < 5:
                        imgname = '0' + imgname
                    if not os.path.exists(inter_file_path):
                        os.makedirs(inter_file_path)
                    im.save(os.path.join(inter_file_path, imgname + '.png'))
                #######################################
                if first_scribble:
                    scribble_label = rough_ROI(scribble_label)

                ##############################
                ref_prev_label = pred_label.unsqueeze(0)
                prev_label = pred_label.unsqueeze(0)
                prev_embedding = ref_frame_embedding
                for ii in range(start_annotated_frame + 1, total_frame_num):
                    current_embedding = embedding_memory[ii]
                    current_embedding = current_embedding.unsqueeze(0)
                    prev_label = prev_label
                    if parallel:
                        for c in model.children():
                            tmp_dic, eval_global_map_tmp_dic, local_map_dics = c.prop_seghead(
                                ref_frame_embedding,
                                prev_embedding,
                                current_embedding,
                                scribble_label,
                                prev_label,
                                normalize_nearest_neighbor_distances=True,
                                use_local_map=True,
                                seq_names=[sequence],
                                gt_ids=paddle.to_tensor([obj_nums]),
                                k_nearest_neighbors=cfg['knns'],
                                global_map_tmp_dic=eval_global_map_tmp_dic,
                                local_map_dics=local_map_dics,
                                interaction_num=n_interaction,
                                start_annotated_frame=start_annotated_frame,
                                frame_num=[ii],
                                dynamic_seghead=c.dynamic_seghead)
                    else:
                        tmp_dic, eval_global_map_tmp_dic, local_map_dics = model.prop_seghead(
                            ref_frame_embedding,
                            prev_embedding,
                            current_embedding,
                            scribble_label,
                            prev_label,
                            normalize_nearest_neighbor_distances=True,
                            use_local_map=True,
                            seq_names=[sequence],
                            gt_ids=paddle.to_tensor([obj_nums]),
                            k_nearest_neighbors=cfg['knns'],
                            global_map_tmp_dic=eval_global_map_tmp_dic,
                            local_map_dics=local_map_dics,
                            interaction_num=n_interaction,
                            start_annotated_frame=start_annotated_frame,
                            frame_num=[ii],
                            dynamic_seghead=model.dynamic_seghead)
                    pred_label = tmp_dic[sequence]
                    pred_label = nn.functional.interpolate(pred_label,
                                                           size=(h, w),
                                                           mode='bilinear',
                                                           align_corners=True)
                    pred_label = paddle.argmax(pred_label, axis=1)
                    pred_masks.append(float_(pred_label))
                    prev_label = pred_label.unsqueeze(0)
                    prev_embedding = current_embedding
                    prev_label_storage[ii] = float_(pred_label[0])
                    if is_save_image:
                        pred_label_to_save = pred_label.squeeze(0).numpy()
                        im = Image.fromarray(
                            pred_label_to_save.astype('uint8')).convert('P', )
                        im.putpalette(_palette)
                        imgname = str(ii)
                        while len(imgname) < 5:
                            imgname = '0' + imgname
                        if not os.path.exists(inter_file_path):
                            os.makedirs(inter_file_path)
                        im.save(os.path.join(inter_file_path,
                                             imgname + '.png'))
                #######################################
                prev_label = ref_prev_label
                prev_embedding = ref_frame_embedding
                #######
                # Propagation <-
                for ii in range(start_annotated_frame):
                    current_frame_num = start_annotated_frame - 1 - ii
                    current_embedding = embedding_memory[current_frame_num]
                    current_embedding = current_embedding.unsqueeze(0)
                    prev_label = prev_label
                    if parallel:
                        for c in model.children():
                            tmp_dic, eval_global_map_tmp_dic, local_map_dics = c.head.prop_seghead(
                                ref_frame_embedding,
                                prev_embedding,
                                current_embedding,
                                scribble_label,
                                prev_label,
                                normalize_nearest_neighbor_distances=True,
                                use_local_map=True,
                                seq_names=[sequence],
                                gt_ids=paddle.to_tensor([obj_nums]),
                                k_nearest_neighbors=cfg['knns'],
                                global_map_tmp_dic=eval_global_map_tmp_dic,
                                local_map_dics=local_map_dics,
                                interaction_num=n_interaction,
                                start_annotated_frame=start_annotated_frame,
                                frame_num=[current_frame_num],
                                dynamic_seghead=c.head.dynamic_seghead)
                    else:
                        tmp_dic, eval_global_map_tmp_dic, local_map_dics = model.prop_seghead(
                            ref_frame_embedding,
                            prev_embedding,
                            current_embedding,
                            scribble_label,
                            prev_label,
                            normalize_nearest_neighbor_distances=True,
                            use_local_map=True,
                            seq_names=[sequence],
                            gt_ids=paddle.to_tensor([obj_nums]),
                            k_nearest_neighbors=cfg['knns'],
                            global_map_tmp_dic=eval_global_map_tmp_dic,
                            local_map_dics=local_map_dics,
                            interaction_num=n_interaction,
                            start_annotated_frame=start_annotated_frame,
                            frame_num=[current_frame_num],
                            dynamic_seghead=model.dynamic_seghead)
                    pred_label = tmp_dic[sequence]
                    pred_label = nn.functional.interpolate(pred_label,
                                                           size=(h, w),
                                                           mode='bilinear',
                                                           align_corners=True)

                    pred_label = paddle.argmax(pred_label, axis=1)
                    pred_masks_reverse.append(float_(pred_label))
                    prev_label = pred_label.unsqueeze(0)
                    prev_embedding = current_embedding
                    ####
                    prev_label_storage[current_frame_num] = float_(
                        pred_label[0])
                    ###
                    if is_save_image:
                        pred_label_to_save = pred_label.squeeze(0).numpy()
                        im = Image.fromarray(
                            pred_label_to_save.astype('uint8')).convert('P', )
                        im.putpalette(_palette)
                        imgname = str(current_frame_num)
                        while len(imgname) < 5:
                            imgname = '0' + imgname
                        if not os.path.exists(inter_file_path):
                            os.makedirs(inter_file_path)
                        im.save(os.path.join(inter_file_path,
                                             imgname + '.png'))
                pred_masks_reverse.reverse()
                pred_masks_reverse.extend(pred_masks)
                final_masks = paddle.concat(pred_masks_reverse, 0)
                result_dir = cfg['TEST']['output_dir']+'/'+cfg['MODEL']['model_name']+'/'+sequence+'/results'
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                submit_masks(result_dir, final_masks.numpy(), images)

                t_end = timeit.default_timer()
                Logging.info('Total time for single interaction: ' +
                             str(t_end - t_total))
        return None
