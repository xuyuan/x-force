

import pickle
import shelve
import time
from pathlib import Path

import torch
from tqdm import tqdm
from .utils import choose_device, get_num_workers
from .predictions import Predictions
from .data import split_dataset


def inference(net, samples, output_root=None, process_id=None, disable_tqdm=None):
    assert callable(net)
    desc = "inference"
    if process_id is not None:
        desc += " #{}".format(process_id)
    pbar = tqdm(samples, unit="images", desc=desc, position=process_id, disable=disable_tqdm,
                mininterval=60, miniters=len(samples)//100, smoothing=1)

    detections = {}
    output_detection = None

    if output_root is not None:
        output_detection = str(output_root / "detections")
        if process_id is not None:
            output_detection += str(process_id)
        detections = shelve.open(output_detection, flag='n', protocol=pickle.HIGHEST_PROTOCOL)

    with torch.no_grad():
        start_time = time.time()
        for sample in pbar:
            image_id = sample['image_id']
            if isinstance(image_id, list):
                image_id = image_id[0]  # assume batch_size = 1

            d = net(sample)
            detections[image_id] = d
        finish_time = time.time()
    inference_time = finish_time - start_time

    if output_detection is not None:
        detections.close()  # close shelve
        # return filename
        return output_detection, inference_time
    else:
        return detections, inference_time


class Tester(object):
    def __init__(self, create_model=None, device='auto', jobs=1, disable_tqdm=None):
        self.create_model = create_model if create_model else Tester.default_create_model
        self.device = device
        self.jobs = jobs
        self.disable_tqdm = disable_tqdm

    @staticmethod
    def default_create_model(filename, use_cuda):
        model = torch.load(filename)
        model.eval()
        if use_cuda:
            model.cuda()
        classes = None
        return model, classes

    def inference(self, model_file, dataset, output_root=None, process_id=None, cuda_device_id=None):
        if cuda_device_id is not None:
            #torch.backends.cudnn.benchmark = True
            #torch.backends.cudnn.deterministic = True
            torch.cuda.set_device(cuda_device_id)
            use_cuda = True
        else:
            use_cuda = False

        model, classes = self.create_model(model_file, use_cuda)
        det, inference_time = inference(model, dataset, output_root, process_id, self.disable_tqdm)
        return classes, det

    def inference_(self, kwargs):
        process_id = kwargs.get('process_id', 0)
        if process_id >= torch.cuda.device_count():
            # wait to avoid GPU OOM due to cudnn.benchmark
            time.sleep((process_id - torch.cuda.device_count() + 1) * 10)

        return self.inference(**kwargs)

    def test(self, model_file, dataset, output=None):
        device = choose_device(self.device)
        use_cuda = device.type == 'cuda'

        output_root = None
        if output:
            output_root = Path(output)
            output_root.mkdir(parents=True, exist_ok=True)

        detections = {}
        detection_files = []
        classnames = None

        num_processes = get_num_workers(self.jobs)
        if num_processes == 1:
            cuda_device_id = 0 if use_cuda else None
            classnames, r = self.inference(model_file, dataset, output_root, cuda_device_id=cuda_device_id)
            if isinstance(r, str):
                detection_files.append(r)
            else:
                detections = r
        else:
            assert (num_processes > 1)
            if not output_root:
                raise RuntimeError("multi-processes test only possible with output as cache")
            print('start', num_processes, 'processes')
            dataset_splits = split_dataset(dataset, num_processes)
            worker_args = [dict(model_file=model_file, dataset=subset, process_id=i, output_root=output_root)
                           for i, subset in enumerate(dataset_splits)]

            import torch.multiprocessing as multiprocessing
            if use_cuda:
                multiprocessing = multiprocessing.get_context('spawn')  # necessary for sharing cuda data
                n_gpu = torch.cuda.device_count()
                # distribute to GPUs
                if n_gpu > 0:
                    print('with {} GPUs'.format(n_gpu))
                    for i, arg in enumerate(worker_args):
                        arg['cuda_device_id'] = i % n_gpu

            with multiprocessing.Pool(processes=num_processes) as pool:
                sub_ret = pool.map_async(self.inference_, worker_args)
                sub_ret.wait()
                for cls, r in sub_ret.get():
                    if classnames is None:
                        classnames = cls
                    else:
                        assert classnames == cls

                    if isinstance(r, str):
                        detection_files.append(r)
                    else:
                        detections.update(r)

        predictions = Predictions([detections] + detection_files, classnames)
        if output_root:
            output_detection = output_root / "detections.pkl"
            print('saving', output_detection)
            predictions.save(output_detection)
            print('saved', output_detection)

        return predictions