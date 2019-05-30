from pycocotools.mask import encode
import json
import numpy as np
from test import pred_to_mask

def make_submit(image_name, preds):
    '''
    Convert the prediction of each image to the required submit format
    :param image_name: image file name
    :param preds: 5 class prediction mask in numpy array
    :return:
    '''

    submit=dict()
    submit['image_name']= image_name
    submit['size']=(preds.shape[1],preds.shape[2])  #(height,width)
    submit['mask']=dict()

    for cls_id in range(0, 5):      # 5 classes in this competition

        mask=preds[cls_id, :, :]
        cls_id_str=str(cls_id+1)   # class index from 1 to 5,convert to str
        fortran_mask = np.asfortranarray(mask)
        rle = encode(fortran_mask) #encode the mask into rle, for detail see: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py
        submit['mask'][cls_id_str]=rle

    return submit


def dump_2_json(submits,save_p):
    '''
    :param submits: submits dict
    :param save_p: json dst save path
    :return:
    '''
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, bytes):
                return str(obj, encoding='utf-8')
            return json.JSONEncoder.default(self, obj)

    print('save submission', save_p)
    file = open(save_p, 'w', encoding='utf-8')
    file.write(json.dumps(submits, cls=MyEncoder, indent=4))
    file.close()


def export(predictions, output):
    submits_dict=dict()
    num_normal = 0
    num_restricted = 0
    for img_id, pred in predictions:
        masks = pred_to_mask(pred)
        if masks.any():
            num_restricted += 1
        else:
            num_normal += 1
        submit=make_submit(img_id, masks)
        submits_dict[img_id] = submit
    dump_2_json(submits_dict, output)
    print(f'num_normal={num_normal}, num_restricted={num_restricted}')


if __name__ == "__main__":
    from trainer import Predictions
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     fromfile_prefix_chars='@')
    parser.add_argument('predictions', type=str, help='pickle file of inference results')
    parser.add_argument('-o', '--output', default='', type=str,
                        help='path for output (default the same directory as input)')
    args = parser.parse_args()

    predictions = Predictions.open(args.predictions)

    output = args.output if args.output else Path(args.predictions).with_name('submission.json')
    export(predictions, output)
