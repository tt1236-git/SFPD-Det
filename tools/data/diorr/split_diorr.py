# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import math
import os
import os.path as osp
import shutil
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import mmcv
import numpy as np
from mmrotate.core.evaluation import eval_rbbox_map


def parse_args():
    parser = argparse.ArgumentParser(description='Split DIOR-R dataset')
    parser.add_argument(
        '--data-root',
        default='data/diorr',
        type=str,
        help='DIOR-R dataset root')
    parser.add_argument(
        '--out-dir',
        default='data',
        type=str,
        help='output path of processed dataset')
    parser.add_argument(
        '--trainval-rate',
        default=1.0,
        type=float,
        help='the split rate of trainval dataset')
    parser.add_argument(
        '--val-rate',
        default=0.3,
        type=float,
        help='the split rate of val dataset')
    parser.add_argument(
        '--nproc', default=10, type=int, help='the procession number')
    args = parser.parse_args()
    return args


def load_diorr_txt(txt_file):
    """Load DIOR-R dataset from txt file."""
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    data_list = []
    for line in lines:
        data_list.append(line.strip())
    return data_list


def load_diorr_xml(xml_file):
    """Load DIOR-R dataset from xml file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        difficult = int(obj.find('difficult').text)
        bnd_box = obj.find('robndbox')
        bbox = [
            float(bnd_box.find('cx').text),
            float(bnd_box.find('cy').text),
            float(bnd_box.find('w').text),
            float(bnd_box.find('h').text),
            float(bnd_box.find('angle').text)
        ]
        objects.append({
            'name': name,
            'difficult': difficult,
            'bbox': bbox,
        })
    return objects


def generate_txt_files(src_path, dst_path, rate):
    """Generate txt files for split datasets."""
    os.makedirs(dst_path, exist_ok=True)
    files = glob.glob(osp.join(src_path, '*.xml'))
    files = [osp.basename(f)[:-4] for f in files]
    np.random.shuffle(files)
    trainval_files = files[:int(rate * len(files))]
    test_files = files[int(rate * len(files)):]
    with open(osp.join(dst_path, 'trainval.txt'), 'w') as f:
        for filename in trainval_files:
            f.write(f'{filename}\n')
    with open(osp.join(dst_path, 'test.txt'), 'w') as f:
        for filename in test_files:
            f.write(f'{filename}\n')
    return trainval_files, test_files


def generate_file_list(data_root, out_dir, trainval_rate, val_rate):
    """Generate file lists for trainval and test datasets."""
    trainval_files, test_files = generate_txt_files(
        osp.join(data_root, 'annotations'),
        out_dir,
        trainval_rate)
    np.random.shuffle(trainval_files)
    train_files = trainval_files[:int((1 - val_rate) * len(trainval_files))]
    val_files = trainval_files[int((1 - val_rate) * len(trainval_files)):]
    with open(osp.join(out_dir, 'train.txt'), 'w') as f:
        for filename in train_files:
            f.write(f'{filename}\n')
    with open(osp.join(out_dir, 'val.txt'), 'w') as f:
        for filename in val_files:
            f.write(f'{filename}\n')
    return train_files, val_files, test_files


def convert_diorr_to_mmrotate(src_path, dst_path, img_files, ann_files):
    """Convert DIOR-R dataset to MMRotate format."""
    img_dir = osp.join(dst_path, 'images')
    os.makedirs(img_dir, exist_ok=True)
    label_dir = osp.join(dst_path, 'annfiles')
    os.makedirs(label_dir, exist_ok=True)

    for img_file in img_files:
        img_src = osp.join(src_path, 'images', f'{img_file}.jpg')
        img_dst = osp.join(img_dir, f'{img_file}.jpg')
        shutil.copy(img_src, img_dst)

    for ann_file in ann_files:
        ann_src = osp.join(src_path, 'annotations', f'{ann_file}.xml')
        if not osp.exists(ann_src):
            continue
        objects = load_diorr_xml(ann_src)
        label_dst = osp.join(label_dir, f'{ann_file}.txt')
        with open(label_dst, 'w') as f:
            for obj in objects:
                bbox = obj['bbox']
                difficult = obj['difficult']
                label = obj['name']
                line = f'{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]} {label} {difficult}\n'
                f.write(line)


def single_scale_process(data_root, out_dir, files, split):
    """Process single scale dataset."""
    dst_path = osp.join(out_dir, f'split_ss_diorr/{split}')
    convert_diorr_to_mmrotate(data_root, dst_path, files, files)


def multi_scale_process(data_root, out_dir, files, split):
    """Process multi scale dataset."""
    dst_path = osp.join(out_dir, f'split_ms_diorr/{split}')
    convert_diorr_to_mmrotate(data_root, dst_path, files, files)


def main():
    args = parse_args()
    data_root = args.data_root
    out_dir = args.out_dir
    trainval_rate = args.trainval_rate
    val_rate = args.val_rate
    nproc = args.nproc

    print('Generating file lists...')
    train_files, val_files, test_files = generate_file_list(
        data_root, out_dir, trainval_rate, val_rate)
    trainval_files = train_files + val_files

    pool = Pool(nproc)
    print('Processing single scale dataset...')
    pool.apply_async(
        single_scale_process,
        args=(data_root, out_dir, trainval_files, 'trainval'))
    pool.apply_async(
        single_scale_process, args=(data_root, out_dir, val_files, 'val'))
    pool.apply_async(
        single_scale_process, args=(data_root, out_dir, test_files, 'test'))

    print('Processing multi scale dataset...')
    pool.apply_async(
        multi_scale_process,
        args=(data_root, out_dir, trainval_files, 'trainval'))
    pool.apply_async(
        multi_scale_process, args=(data_root, out_dir, val_files, 'val'))
    pool.apply_async(
        multi_scale_process, args=(data_root, out_dir, test_files, 'test'))

    pool.close()
    pool.join()
    print('Done!')


if __name__ == '__main__':
    main()