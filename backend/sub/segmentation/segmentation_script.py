import torch
import numpy as np
import nibabel as nib
from skimage.transform import resize
from skimage import measure

# import trimesh
import logging
from .segmentation_model import UNet
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Get the directory of the current script
script_dir = Path(__file__).resolve().parent

# Define paths relative to the script directory
TRAIN_DATASET_PATH = script_dir / "BraTS2020_TrainingData/BraTS20_Training_002"
TRAIN_DATASET_PATH.mkdir(parents=True, exist_ok=True)

MODEL_CHECKPOINT_PATH = script_dir / "final_model/model.pth"
FILES_DIRECTORY = script_dir.parent.parent / "files"
FILES_DIRECTORY.mkdir(parents=True, exist_ok=True)


def write_obj(filename, vertices, faces):
    with open(filename, "w") as file:
        for v in vertices:
            file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for f in faces:
            # OBJ faces are 1-indexed
            file.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")


def segment_and_export_mesh(input_filename):
    """
    Perform segmentation and return the mesh object without exporting it.

    Args:
    input_filename (str): The filename of the input image file.

    Returns:
    trimesh.Trimesh: The mesh object generated from the segmentation.
    """
    logging.info("Starting segmentation script.")
    image_path = input_filename
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device for computation: {device}")

    try:
        model = UNet(num_classes=1).to(device)
        if MODEL_CHECKPOINT_PATH.exists():
            checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint)
            logging.info("Model loaded successfully.")
        else:
            raise FileNotFoundError(f"Checkpoint '{MODEL_CHECKPOINT_PATH}' not found.")

        image_nii = nib.load(str(image_path)).get_fdata()
        image_resized = resize(image_nii, (240, 240, 155), anti_aliasing=True)
        predictions_3d = image_resized

        # predictions_3d = np.zeros(image_resized.shape)

        # for slice_idx in range(image_resized.shape[2]):
        #     slice_tensor = (
        #         torch.from_numpy(image_resized[:, :, slice_idx])
        #         .unsqueeze(0)
        #         .unsqueeze(0)
        #         .float()
        #     )
        #     slice_tensor = slice_tensor.to(device)
        #     with torch.no_grad():
        #         output = model(slice_tensor)
        #         probs = torch.sigmoid(output)
        #         preds = (probs > 0.5).float()
        #     predictions_3d[:, :, slice_idx] = preds.squeeze().cpu().numpy()

        return measure.marching_cubes(predictions_3d, 0)
    # verts, faces, normals, values

    except Exception as e:
        logging.error(f"Error in segmentation: {e}")
        raise


def main_segmentation(input_filename, USE_PREPROCESSING):
    mesh_brain = segment_and_export_mesh(
        FILES_DIRECTORY / input_filename
    )  # actual brain part (grey)

    mesh_lesion = nib.load(
        TRAIN_DATASET_PATH / "BraTS20_Training_002_seg.nii"
    ).get_fdata()  # the lesion part (yellow)
    mesh_lesion = measure.marching_cubes(mesh_lesion, 0)

    verts_brain, faces_brain, _, _ = mesh_brain
    verts_lesion, faces_lesion, _, _ = mesh_lesion

    write_obj(
        FILES_DIRECTORY
        / "obj_output"
        / (
            input_filename.replace(
                "_normalized.nii.gz" if USE_PREPROCESSING else ".nii", ""
            )
            + "_brain.obj"
        ),
        verts_brain,
        faces_brain,
    )

    write_obj(
        FILES_DIRECTORY
        / "obj_output"
        / (
            input_filename.replace(
                "_normalized.nii.gz" if USE_PREPROCESSING else ".nii", ""
            )
            + "_lesion.obj"
        ),
        verts_lesion,
        faces_lesion,
    )
    return True
