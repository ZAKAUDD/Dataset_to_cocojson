#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
import pycococreatortools

ROOT_DIR = '2525'
IMAGE_DIR = os.path.join(ROOT_DIR, "SuaKIT_Original_20200425_082946/Train/Labeled")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "SuaKIT_Label_20200425_082946/Train/Labeled")
INFO = {
    "description": "Diangan Dataset",
    "url": "https://github.com",
    "version": "0.1.0",
    "year": 2020,
    "contributor": "Gao",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}
LICENSES = [
    {
        "id": 1,
        "name": "Gao License",
        "url": "http://www.baidu.com"
    }
]
CATEGORIES = [
    {
        "id": 1,
        "name": "Dingbukongdong",
        "supercategory": "diangan",
    },
    {
        "id": 2,
        "name": "Suipian",
        "supercategory": "diangan",
    },
    {
        "id": 3,
        "name": "Liewen",
        "supercategory": "diangan",
    },
    {
        "id": 4,
        "name": "Loutong",
        "supercategory": "diangan",
    },
    {
        "id": 5,
        "name": "Dingbuyiwu",
        "supercategory": "diangan",
    },
    {
        "id": 6,
        "name": "Dingbuyouwu",
        "supercategory": "diangan",
    },
]


def main():
    coco_output = {
        'info': INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    image_id = 1
    segmentation_id = 1
    # filter for jpeg images
    image_files = os.listdir(IMAGE_DIR)
    annotation_filenames = os.listdir(ANNOTATION_DIR)
    # go through each image
    for image_filename, annotation_filename in zip(image_files, annotation_filenames):
        image = Image.open(os.path.join(IMAGE_DIR, image_filename))
        image_info = pycococreatortools.create_image_info(
            image_id, image_filename, image.size)
        coco_output["images"].append(image_info)
        print(image_filename, annotation_filename)
        binary_mask = np.asarray(Image.open(os.path.join(ANNOTATION_DIR, annotation_filename)).convert('1')).astype(np.uint8)
        mask = np.unique(binary_mask)
        for i in mask:
            if i < 255:
                class_id = i
                category_info = {"id": class_id, "is_crowd": "crowd" in image_filename}
                annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id, image_id, category_info, binary_mask,
                    image.size, tolerance=2)
                coco_output["annotations"].append(annotation_info)
        segmentation_id = segmentation_id + 1
        image_id = image_id + 1

    class NumpyEncoder(json.JSONEncoder):
        """ Special json encoder for numpy types """
        def default(self, obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32,
                                  np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    # operation = json.dumps(coco_output, cls=NumpyEncoder)
    with open("{}/instances_diangan_train2020.json".format(ROOT_DIR), "w", encoding="utf-8") as output_json_file:
        json.dump(coco_output, output_json_file, ensure_ascii=False, indent=1, cls=NumpyEncoder)


if __name__ == "__main__":
    main()
