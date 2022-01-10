# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from resources.backend.paddlevideo.loader.builder import build_pipeline

from resources.backend.paddlevideo.loader.pipelines import ToTensor_manet

import os
import timeit
import paddle
from PIL import Image
from davisinteractive.utils.scribbles import scribbles2mask, annotated_frames
from paddle import nn


from resources.backend.paddlevideo.utils import load
from resources.backend.paddlevideo.utils.manet_utils import float_, _palette, damage_masks, int_, long_, write_dict, rough_ROI
from EIVideo.api import load_video, get_images, submit_masks, get_scribbles
from ...builder import build_model
from ...registry import SEGMENT
from .base import BaseSegment

# if cfg.MODEL.framework == "ManetSegment_Stage1":
#     cfg_helper = {"knns": 1,
#                   "is_save_image": True}
#     cfg.update(cfg_helper)
#     build_model(cfg['MODEL']).test_step(**cfg,
#                                         weights=weights,
#                                         parallel=False)
#     return


@SEGMENT.register()
class ManetSegment_Stage1(BaseSegment):
    def __init__(self, backbone=None, head=None, **cfg):
        super().__init__(backbone, head, **cfg)

    def train_step(self, data_batch, step, **cfg):
        """Define how the model is going to train, from input to output.
        返回任何你想打印到日志中的东西
        """
        ref_imgs = data_batch['ref_img']  # batch_size * 3 * h * w
        img1s = data_batch['img1']
        img2s = data_batch['img2']
        ref_scribble_labels = data_batch[
            'ref_scribble_label']  # batch_size * 1 * h * w
        label1s = data_batch['label1']
        label2s = data_batch['label2']
        seq_names = data_batch['meta']['seq_name']
        obj_nums = data_batch['meta']['obj_num']

        bs, _, h, w = img2s.shape
        inputs = paddle.concat((ref_imgs, img1s, img2s), 0)
        if cfg['damage_initial_previous_frame_mask']:
            try:
                label1s = damage_masks(label1s)
            except:
                label1s = label1s
                print('damage_error')

        tmp_dic = self.head(inputs,
                            ref_scribble_labels,
                            label1s,
                            use_local_map=True,
                            seq_names=seq_names,
                            gt_ids=obj_nums,
                            k_nearest_neighbors=cfg['knns'])
        label_and_obj_dic = {}
        label_dic = {}
        obj_dict = {}
        for i, seq_ in enumerate(seq_names):
            label_and_obj_dic[seq_] = (label2s[i], obj_nums[i])
        for seq_ in tmp_dic.keys():
            tmp_pred_logits = tmp_dic[seq_]
            tmp_pred_logits = nn.functional.interpolate(tmp_pred_logits,
                                                        size=(h, w),
                                                        mode='bilinear',
                                                        align_corners=True)
            tmp_dic[seq_] = tmp_pred_logits
            label_tmp, obj_num = label_and_obj_dic[seq_]
            label_dic[seq_] = long_(label_tmp)
        loss_metrics = {
            'loss':
            self.head.loss(dic_tmp=tmp_dic,
                           label_dic=label_dic,
                           step=step,
                           obj_dict=obj_dict) / bs
        }
        return loss_metrics

    def val_step(self, data_batch, **kwargs):
        pass

    def infer_step(self, data_batch, **kwargs):
        """Define how the model is going to test, from input to output."""
        pass

    def test_step(self, weights, parallel=True, is_save_image=True, **cfg):
        # 1. Construct model.
        cfg['MODEL'].head.pretrained = ''
        cfg['MODEL'].head.test_mode = True
        model = build_model(cfg['MODEL'])
        if parallel:
            model = paddle.DataParallel(model)

        # 2. Construct data.
        # sequence = 'drone'
        sequence = 'bike-packing'
        obj_nums = 1
        if sequence == 'drone':
            video = 'data/1.mp4'
            images = load_video(video, 480)
        elif sequence == 'bike-packing':
            images = get_images(sequence='bike-packing')
            obj_nums = 2
        # [195, 389, 238, 47, 244, 374, 175, 399]
        # .shape: (502, 480, 600, 3)
        report_save_dir = cfg.get("output_dir",
                                  f"./output/{cfg['model_name']}")
        if not os.path.exists(report_save_dir):
            os.makedirs(report_save_dir)
            # Configuration used in the challenges
        max_nb_interactions = 8  # Maximum number of interactions
        # Interactive parameters
        model.eval()

        state_dicts_ = load(weights)['state_dict']
        state_dicts = {}
        for k, v in state_dicts_.items():
            if 'num_batches_tracked' not in k:
                state_dicts['head.' + k] = v
                if ('head.' + k) not in model.state_dict().keys():
                    print(f'pretrained -----{k} -------is not in model')
        write_dict(state_dicts, 'model_for_infer.txt', **cfg)
        model.set_state_dict(state_dicts)
        inter_file = open(
            os.path.join(
                cfg.get("output_dir", f"./output/{cfg['model_name']}"),
                'inter_file.txt'), 'w')
        seen_seq = False

        with paddle.no_grad():

            # Get the current iteration scribbles
            for scribbles, first_scribble in get_scribbles():
                t_total = timeit.default_timer()
                f, h, w = images.shape[:3]
                if 'prev_label_storage' not in locals().keys():
                    prev_label_storage = paddle.zeros([f, h, w])
                if len(annotated_frames(scribbles)) == 0:
                    final_masks = prev_label_storage
                    submit_masks(final_masks.numpy(), images)
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
                inter_file.write(sequence + ' ' + 'interaction' +
                                 str(n_interaction) + ' ' + 'frame' +
                                 str(start_annotated_frame) + '\n')

                if first_scribble:  # if in the first round, extract pixel embbedings.
                    if not seen_seq:
                        seen_seq = True
                        inter_turn = 1
                        embedding_memory = []
                        places = paddle.set_device('gpu')

                        for imgs in images:
                            if cfg['PIPELINE'].get('test'):
                                imgs = paddle.to_tensor([
                                    build_pipeline(cfg['PIPELINE'].test)({
                                        'img1':
                                        imgs
                                    })['img1']
                                ])
                            else:
                                imgs = paddle.to_tensor([imgs])
                            if parallel:
                                for c in model.children():
                                    frame_embedding = c.head.extract_feature(
                                        imgs)
                            else:
                                frame_embedding = model.head.extract_feature(
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
                scribble_sample = ToTensor_manet()(scribble_sample)
                #                     print(ref_frame_embedding, ref_frame_embedding.shape)
                scribble_label = scribble_sample['scribble_label']

                scribble_label = scribble_label.unsqueeze(0)
                model_name = cfg['model_name']
                output_dir = cfg.get("output_dir", f"./output/{model_name}")
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
                    print(paddle.unique(scribble_label))
                    final_masks = prev_label_storage
                    submit_masks(final_masks.numpy(), images, inter_file_path)
                    continue

                    ###inteaction segmentation head
                if parallel:
                    for c in model.children():
                        tmp_dic, local_map_dics = c.head.int_seghead(
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
                    tmp_dic, local_map_dics = model.head.int_seghead(
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
                                frame_num=[ii],
                                dynamic_seghead=c.head.dynamic_seghead)
                    else:
                        tmp_dic, eval_global_map_tmp_dic, local_map_dics = model.head.prop_seghead(
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
                            dynamic_seghead=model.head.dynamic_seghead)
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
                        tmp_dic, eval_global_map_tmp_dic, local_map_dics = model.head.prop_seghead(
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
                            dynamic_seghead=model.head.dynamic_seghead)
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
                submit_masks(final_masks.numpy(), images, inter_file_path)

                t_end = timeit.default_timer()
                print('Total time for single interaction: ' +
                      str(t_end - t_total))
        inter_file.close()