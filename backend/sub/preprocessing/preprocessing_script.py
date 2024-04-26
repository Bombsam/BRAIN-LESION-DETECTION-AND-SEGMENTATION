import os
import SimpleITK as sitk
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Get the directory of the current script
script_dir = Path(__file__).resolve().parent

FILES_DIRECTORY = script_dir.parent / "files"
FILES_DIRECTORY.mkdir(parents=True, exist_ok=True)


def setup_environment():
    logging.info("Setting up environment for NeuroDesk and FSL")
    os.environ["LD_PRELOAD"] = ""
    os.environ["APPTAINER_BINDPATH"] = "/content"
    os.environ["LMOD_CMD"] = "/usr/share/lmod/lmod/libexec/lmod"
    os.environ["MODULEPATH"] = ":".join(
        map(
            str,
            list(
                map(
                    lambda x: os.path.join(
                        os.path.abspath(
                            "/cvmfs/neurodesk.ardc.edu.au/neurodesk-modules/"
                        ),
                        x,
                    ),
                    os.listdir("/cvmfs/neurodesk.ardc.edu.au/neurodesk-modules/"),
                )
            ),
        )
    )
    os.environ["SM_FRAMEWORK"] = "tf.keras"
    logging.info("Environment setup complete")


def intensity_normalization(input_path, output_path):
    logging.info(f"Attempting to read from: {input_path}")
    image = sitk.ReadImage(str(input_path))
    rescaleFilter = sitk.RescaleIntensityImageFilter()
    rescaleFilter.SetOutputMaximum(255)
    rescaleFilter.SetOutputMinimum(0)
    image = rescaleFilter.Execute(image)
    print({output_path})
    sitk.WriteImage(image, str(output_path))
    logging.info(f"Intensity normalization complete. Output saved to {output_path}")
    return str(output_path)


def preprocess_mri(input_filename):
    setup_environment()
    input_path = FILES_DIRECTORY / input_filename
    normalized_file = FILES_DIRECTORY / input_filename.replace(
        ".nii", "_normalized.nii.gz"
    )
    return intensity_normalization(input_path, normalized_file)
