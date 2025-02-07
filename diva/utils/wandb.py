import base64
import hashlib
import io
import json
import os
import tempfile

import numpy as np
import wandb


def hash_name(long_name):
    """Used to condense W&B labels (`long_name`) to artifact name of appropriate length.
    
    We can later pass in the same `long_name` to find the relevant artifact. 
    """
    # Create a SHA-256 hash of the long name
    hash_object = hashlib.sha256(long_name.encode('utf-8'))
    # Get the hexadecimal digest of the hash
    hash_hex = hash_object.hexdigest()
    return hash_hex


def solutions_from_artifact(artifact):
    artifact_name = artifact.name
    assert artifact.version in artifact_name
    with tempfile.TemporaryDirectory() as temp_dir:
        _artifact_dir = artifact.download(root=temp_dir)
        file_path = os.path.join(_artifact_dir, 'solutions.json')
        with open(file_path) as f:
            data = f.read()
        if not data: 
            return None
        json_data = json.loads(data)
        solutions = np.array(load_and_decompress_matrix(json_data))
    return solutions


def download_archive_data(run_name, run_index=0):
    """ For a given run, defined by name and index, download wandb data for heatmap plotting. """
    from diva.wandb_config import ENTITY, PROJECT
    # Select relevant runs
    api = wandb.Api(timeout=120) 
    entity = ENTITY
    project = PROJECT
    
    # run_index = 0
    raw_run_stages = ['ws1', 'ws2']
    
    runs = api.runs(f'{entity}/{project}', {"config.wandb_label": run_name}, order='created_at')  # Remove dict to get all runs
    run = runs[run_index] # Get specific run from defined index
    archive_dims = run._attrs['config']['dist']['qd']['archive_dims']
    measures = run._attrs['config']['dist']['qd']['measures']
    gt_type = run._attrs['config']['domain']['gt_type']
    del gt_type  # Not used

    if raw_run_stages is None:
        raw_run_stages = ['ws1', 'ws2']
    run_stages = []

    all_solutions = []
    for stage in raw_run_stages:
        all_artifacts = run.logged_artifacts()
        filtered_artifacts = [artifact for artifact in all_artifacts if f'{stage}' in artifact.name and 'QD' in artifact.name]

        print(f'Filtered down to {len(filtered_artifacts)} artifacts for {stage}')
        if len(filtered_artifacts) == 0:
            print(f'No artifacts for {stage}')
            continue

        artifact = sorted(filtered_artifacts, key=lambda a: int(a.version[1:]), reverse=True)[0]  # Newest one
        solutions = solutions_from_artifact(artifact)
        if solutions is None:
            print(f'No solutions for {stage}')
            continue
        run_stages.append(stage)
        all_solutions.append(solutions)

    solutions_to_use = all_solutions[-1]
    print('Using solutions from stage: ', run_stages[-1])

    return solutions_to_use, archive_dims, measures

class MatrixWrapper:
    def __init__(self, matrix):
        if isinstance(matrix, np.ndarray):
            self.matrix = matrix.tolist()
        else:
            self.matrix = matrix

    def to_json(self):
        return json.dumps(self.matrix)  # Convert list of lists into JSON string


def compress_and_encode_matrix(matrix):
    buf = io.BytesIO()
    np.savez_compressed(buf, matrix=matrix)
    buf.seek(0)
    compressed_data = buf.read()
    encoded_data = base64.b64encode(compressed_data).decode('utf-8')
    return encoded_data


def load_and_decompress_matrix(encoded_data):
    compressed_data = base64.b64decode(encoded_data)
    data_buffer = io.BytesIO(compressed_data)
    with np.load(data_buffer, allow_pickle=True) as data:
        matrix = data['matrix']
    return matrix