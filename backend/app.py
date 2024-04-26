import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from sub.preprocessing.preprocessing_script import preprocess_mri
from sub.segmentation.segmentation_script import main_segmentation
import json
from pydantic import BaseModel
import nibabel as nib
from numpy import rot90

app = FastAPI()

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

script_dir = Path(__file__).resolve().parent

# Folder to store the files
FILES_DIRECTORY = script_dir / "files"
FILES_DIRECTORY.mkdir(parents=True, exist_ok=True)

USE_PREPROCESSING = True


@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    if not (file.filename.endswith(".nii.gz") or file.filename.endswith(".nii")):
        logging.error("Invalid file extension.")
        raise HTTPException(status_code=400, detail="Invalid file extension.")

    file_location = FILES_DIRECTORY / file.filename
    with open(file_location, "wb") as buffer:
        data = await file.read()
        buffer.write(data)
    logging.info(f"File {file.filename} saved at {file_location}")

    try:
        if USE_PREPROCESSING:
            preprocess_mri(file.filename)
            logging.info(
                f"Preprocessed file saved to: {FILES_DIRECTORY / file.filename}"
            )

        main_segmentation(
            (
                file.filename.replace(".nii", "_normalized.nii.gz")
                if USE_PREPROCESSING
                else file.filename
            ),
            USE_PREPROCESSING,
        )
        return JSONResponse(
            json.dumps(
                {
                    "status": "Success",
                    "filename": file.filename.replace(".nii", ""),
                }
            )
        )

    except Exception as e:
        logging.error(f"Processing error: {e}", exc_info=True)
        # Clean up any residual files if there was an error
        try:
            for file_path in FILES_DIRECTORY.glob("*"):
                file_path.unlink()
                logging.info(f"Deleted file: {file_path}")
        except Exception as cleanup_error:
            logging.error(f"Error during cleanup: {cleanup_error}")

        raise HTTPException(status_code=500, detail=str(e))


OBJ_FILES_DIRECTORY = script_dir / "files/obj_output"


@app.get("/obj/{filename}")
async def get_mesh(filename: str):
    file_path = OBJ_FILES_DIRECTORY / filename
    if file_path.exists() and file_path.is_file():
        return FileResponse(path=file_path)
    else:
        logging.info(f"File name: {filename}")
        return {"error": "File not found"}, 404


class Coordinates(BaseModel):
    x: int
    y: int
    z: int
    # file_path: str  # Path to the NIfTI file


class FilePath(BaseModel):
    file_path: str


def load_nifti_file(file_path):
    """Load a NIfTI file and return its data."""
    img = nib.load(file_path)
    return img.get_fdata()


def extract_slice(data, axis, index):
    """Extract a slice from the 3D data."""
    if axis == "x":
        return data[index, :, :]
    elif axis == "y":
        return data[:, index, :]
    elif axis == "z":
        return data[:, :, index]
    else:
        raise ValueError("Invalid axis. Axis must be 'x', 'y', or 'z'.")


# return the shape of data
@app.post("/get_dimensions/")
async def get_dimensions(file_info: FilePath):
    try:
        img = load_nifti_file(
            FILES_DIRECTORY / (file_info.file_path + "_normalized.nii.gz")
        )
        return JSONResponse(content={"dimensions": img.shape})
    except Exception as e:
        logging.error(f"Error while getting dimensions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# return slices
@app.post("/get_slices/")
async def get_slices(coordinates: Coordinates, file_info: FilePath):
    print(coordinates)  # For debugging
    try:
        data = load_nifti_file(
            FILES_DIRECTORY / (file_info.file_path + "_normalized.nii.gz")
        )
        slices = {
            "x_slice": rot90(
                extract_slice(data, "x", coordinates.x), k=1, axes=(0, 1)
            ).tolist(),
            "y_slice": rot90(
                extract_slice(data, "y", coordinates.y), k=1, axes=(0, 1)
            ).tolist(),
            "z_slice": rot90(
                extract_slice(data, "z", coordinates.z), k=1, axes=(0, 1)
            ).tolist(),
        }
        return JSONResponse(content=slices)
    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
