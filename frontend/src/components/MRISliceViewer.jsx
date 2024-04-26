import { useState, useEffect, useRef, useContext } from 'react';
import axios from 'axios';
import PropTypes from 'prop-types';
import TasksContext from '../contexts/TasksContext ';

const MRISliceViewer = ({ niftiDimensions }) => {
    const { currentTask } = useContext(TasksContext);
    console.log({ currentTask })


    const [slices, setSlices] = useState({ x_slice: [], y_slice: [], z_slice: [] });
    const [coordinates, setCoordinates] = useState({
        x: Math.floor(niftiDimensions?.x / 2),
        y: Math.floor(niftiDimensions?.y / 2),
        z: Math.floor(niftiDimensions?.z / 2)
    });
    const getSlicesApiUrl = 'http://localhost:8000/get_slices';

    const canvasRefX = useRef(null);
    const canvasRefY = useRef(null);
    const canvasRefZ = useRef(null);

    useEffect(() => {
        console.log({ coordinates });
        fetchSlices({
            "coordinates": coordinates,
            "file_info": {
                file_path: currentTask.name
            }
        });
    }, [coordinates]);

    const fetchSlices = async (coords) => {
        try {
            const response = await axios.post(getSlicesApiUrl, coords);
            console.log(response.data);
            setSlices(response.data);
        } catch (error) {
            console.error('Error fetching slices:', error);
        }
    };

    useEffect(() => {
        drawSlice('x', slices.x_slice);
        drawSlice('y', slices.y_slice);
        drawSlice('z', slices.z_slice);
    }, [slices]);

    const drawSlice = (axis, sliceData) => {
        const canvas = axis === 'x' ? canvasRefX.current : axis === 'y' ? canvasRefY.current : canvasRefZ.current;
        if (canvas && sliceData.length) {
            const ctx = canvas.getContext('2d');
            const imageData = ctx.createImageData(sliceData[0].length, sliceData.length);
            sliceData.flat().forEach((value, index) => {
                imageData.data[index * 4] = value;
                imageData.data[index * 4 + 1] = value;
                imageData.data[index * 4 + 2] = value;
                imageData.data[index * 4 + 3] = 255;
            });
            ctx.putImageData(imageData, 0, 0);
        }
    };

    const handleMouseMove = (axis, event) => {
        const canvas = axis === 'x' ? canvasRefX.current : axis === 'y' ? canvasRefY.current : canvasRefZ.current;
        if (canvas) {
            const rect = canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;

            // Calculate new coordinate values based on the clicked axis
            let newX, newY, newZ;
            switch (axis) {
                case 'x':
                    newY = Math.floor((y / rect.height) * niftiDimensions.y);
                    newZ = Math.floor((x / rect.width) * niftiDimensions.z);
                    setCoordinates(prev => ({ ...prev, y: newY, z: newZ }));
                    break;
                case 'y':
                    newX = Math.floor((x / rect.width) * niftiDimensions.x);
                    newZ = Math.floor((y / rect.height) * niftiDimensions.z);
                    setCoordinates(prev => ({ ...prev, x: newX, z: newZ }));
                    break;
                case 'z':
                    newX = Math.floor((x / rect.width) * niftiDimensions.x);
                    newY = Math.floor((y / rect.height) * niftiDimensions.y);
                    setCoordinates(prev => ({ ...prev, x: newX, y: newY }));
                    break;
                default:
                    break;
            }
        }
    };

    if (Object.keys(niftiDimensions).length > 0) {
        return (
            <>
                <div style={{ margin: "0.5rem 2rem" }}>
                    <div style={{ display: 'flex', justifyContent: 'center' }}>
                        <div>
                            <canvas ref={canvasRefX} width={250} height={250} onMouseDown={(e) => handleMouseMove('x', e)} />
                            <h3 className="text-left mt-2">X Dimension</h3>
                        </div>
                        <div>
                            <canvas ref={canvasRefY} width={250} height={250} onMouseDown={(e) => handleMouseMove('y', e)} />
                            <h3 className="text-left mt-2">Y Dimension</h3>
                        </div>
                        <div>
                            <canvas ref={canvasRefZ} width={250} height={250} onMouseDown={(e) => handleMouseMove('z', e)} />
                            <h3 className="text-left mt-2">Z Dimension</h3>
                        </div>
                    </div>
                </div>
                {/* <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full flex items-center justify-center">
                    <div className="bg-white p-5 rounded-lg flex items-center flex-col">
                        <div className="loader ease-linear rounded-full border-8 border-t-8 border-gray-200 h-12 w-12 mb-4"></div>
                        <h2 className="text-center text-gray-800 text-xl font-semibold">Loading...</h2>
                        <p className="w-1/3 text-center text-gray-500">Please wait while we process your request.</p>
                    </div>
                </div> */}
            </>
        );
    } else {
        return (
            <h1 className='text-4xl text-center m-2 text-white'>Loading your current segmentation!</h1>
        )
    }

};

MRISliceViewer.propTypes = {
    niftiDimensions: PropTypes.shape({
        x: PropTypes.number.isRequired,
        y: PropTypes.number.isRequired,
        z: PropTypes.number.isRequired
    }).isRequired
};

export default MRISliceViewer;