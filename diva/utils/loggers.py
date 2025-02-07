import datetime
import io
import json
import os

import wandb

os.environ['WANDB_LOG_LEVEL'] = 'DEBUG'

from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from diva.utils.torch import DeviceConfig
from diva.wandb_config import ENTITY, PROJECT


class TBLogger:
    def __init__(self, args, exp_label):
        self.output_name = exp_label + '_' + str(args.seed) + '_' + datetime.datetime.now().strftime('_%d:%m_%H:%M:%S')
        try:
            log_dir = args.domain.results_log_dir
        except AttributeError:
            log_dir = args['results_log_dir']

        if log_dir is None:
            dir_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
            dir_path = os.path.join(dir_path, '_logs/tb')
        else:
            dir_path = log_dir

        if not os.path.exists(dir_path):
            try:
                os.mkdir(dir_path)
            except:
                dir_path_head, dir_path_tail = os.path.split(dir_path)
                if len(dir_path_tail) == 0:
                    dir_path_head, dir_path_tail = os.path.split(dir_path_head)
                os.mkdir(dir_path_head)
                os.mkdir(dir_path)

        self.full_output_folder = os.path.join(os.path.join(dir_path, 'logs_{}'.format(args.domain.env_name)),
                                               self.output_name)

        self.writer = SummaryWriter(log_dir=self.full_output_folder)

        print('logging under', self.full_output_folder)

        if not os.path.exists(self.full_output_folder):
            os.makedirs(self.full_output_folder)
        with open(os.path.join(self.full_output_folder, 'config.json'), 'w') as f:
            config = OmegaConf.to_container(args, resolve=True)
            config.update(device=DeviceConfig.DEVICE.type)
            print(type(config))
            
            json.dump(config, f, indent=2)

    def add(self, name, value, x_pos):
        return
        # self.writer.add_scalar(name, value, x_pos)

    def lazy_add(self, name, value):
        return
    
    def push_metrics(self, x_pos):
        return
    
    def add_video(self, name, video_data, iter_idx, fps=30):
        return  # TODO: We only generate videos for W&B runs currently

    def add_image(self, name, image_data, iter_idx):
        return  # TODO: We only generate images for W&B runs currently
    
    def add_histogram(self, name, value, x_pos):
        return  # TODO: We only generate histograms for W&B runs currently
    
    def close(self):
        self.writer.close()


class WandBLogger(TBLogger):
    """Logger that logs to WandB (and still uses TBLogger as backup)."""
    def __init__(self, args, exp_label, notes='', tags=list()):
        config = OmegaConf.to_container(args, resolve=True)
        dir_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
        dir_path = os.path.join(dir_path, '_logs/_wandb')
        # Initialize W&B rn
        wandb.init(
            # set the wandb project where this run will be logged
            dir=dir_path,
            entity=ENTITY,
            project=PROJECT,
            name=exp_label,
            notes=notes,
            tags=tags,
            # track hyperparameters and run metadata
            config=config
        )
        super().__init__(args, exp_label)
        self.prepared_metrics = dict()
    
    def lazy_add(self, name, value):
        self.prepared_metrics[name] = value

    def push_metrics(self, x_pos):
        self.prepared_metrics['x_pos'] = x_pos
        wandb.log(self.prepared_metrics)
        self.prepared_metrics = dict()
    
    def add(self, name, value, x_pos, skip_tb=False):
        wandb.log({name: value, 'x_pos': x_pos})
        if not skip_tb:
            super().add(name, value, x_pos)
    
    def add_dict(self, name, dict_data, x_pos):
        json_str = json.dumps(dict(dict_data))  # Serialize the dictionary to a JSON string
        json_file = io.StringIO(json_str)  # Create an in-memory text stream
        artifact = wandb.Artifact(name, type='dataset')  # Create a new artifact
        # Add the in-memory file to the artifact, specifying the desired file name within the artifact
        artifact.add_file(json_file, name=f'{name}.json')
        wandb.log_artifact(artifact)  # Log the artifact to W&B
        json_file.close()  # Close the in-memory file
        # NOTE: We don't log dicts to TBLogger currently
    
    def add_histogram(self, name, value, x_pos):
        wandb.log({name: wandb.Histogram(value)})
        # NOTE: We don't log histograms to TBLogger currently
    
    def add_video(self, name, video_data, iter_idx, fps=30):
        wandb.log({name: wandb.Video(video_data, fps=fps, format='mp4')})
        # NOTE: We don't log videos to TBLogger currently
    
    def add_image(self, name, image_data, iter_idx):
        if len(image_data.shape) == 4:
            image_data = image_data[:, :, :3]
        wandb.log({name: wandb.Image(image_data, mode='RGB')})
        # NOTE: We don't log images to TBLogger currently
    
    def close(self):
        # Finish W&B run
        wandb.finish()
