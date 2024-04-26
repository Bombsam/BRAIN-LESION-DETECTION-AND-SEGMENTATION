import { Canvas } from '@react-three/fiber';
import { Suspense, useEffect, useRef, useState } from 'react';
import { OrbitControls } from '@react-three/drei';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import * as THREE from 'three'; // Ensure THREE is imported to use in materials

const Model = ({ objPath, material }) => {
    const [mesh, setMesh] = useState();

    useEffect(() => {
        const loader = new OBJLoader();
        console.log({ objPath })
        fetch(`http://localhost:8000/obj/${objPath}`)
            .then(response => response.blob())
            .then(blob => blob.text())
            .then(objText => {
                const obj = loader.parse(objText);
                obj.traverse(child => {
                    if (child.isMesh) {
                        child.material = material;
                    }
                });
                obj.rotation.x = -Math.PI / 2; // Rotates the object 90 degrees on the X axis
                setMesh(obj);
            })
            .catch(error => console.log('Failed to load model', error));
    }, [objPath, material]);

    return mesh ? <primitive object={mesh} /> : null;
};

const ThreeDModelViewer = ({ filename }) => {
    const orbitControlsRef = useRef();

    useEffect(() => {
        // Whenever the filename changes, reset the controls
        if (orbitControlsRef.current) {
            orbitControlsRef.current.reset();
        }
    }, [filename]);

    const halfTransparentGray = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        opacity: 0.6,
        transparent: true
    });
    const halfTransparentYellow = new THREE.MeshStandardMaterial({
        color: 0xFFFF00,
        opacity: 0.7,
        transparent: true
    });

    return (
        <Canvas style={{ backgroundColor: '#000' }}>
            <ambientLight intensity={0.5} />
            <directionalLight position={[5, 5, 5]} intensity={1} />
            <Suspense fallback={null}>
                <Model objPath={`${filename}_brain.obj`} material={halfTransparentGray} />
                <Model objPath={`${filename}_lesion.obj`} material={halfTransparentYellow} />
                <Model objPath={`${filename}_lesion.obj`} material={halfTransparentYellow} />
            </Suspense>
            <OrbitControls
                ref={orbitControlsRef}
                autoRotate={false}
                rotateSpeed={1.5} // Default is 1, increase for faster rotation
                zoomSpeed={5} // Default is 1, increase for faster zoom
                panSpeed={4} // Default is 1, increase for faster panning
                maxZoom={null}
            />
        </Canvas>

    );
};

export default ThreeDModelViewer;