import { Canvas } from '@react-three/fiber';
import { Suspense, useEffect, useRef, useState } from 'react';
import { Environment, OrbitControls } from '@react-three/drei';
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
        color: 0x808080,
        opacity: 0.5,
        transparent: true
    });
    const halfTransparentYellow = new THREE.MeshStandardMaterial({
        color: 0xFFFF00,
        opacity: 0.5,
        transparent: true
    });

    return (
        <Canvas style={{ backgroundColor: '#e0e0e0' }}>
            <ambientLight intensity={0.5} />
            <directionalLight position={[5, 5, 5]} intensity={1} />
            <Suspense fallback={null}>
                <Model objPath={`${filename}_brain.obj`} material={halfTransparentGray} />
                <Model objPath={`${filename}_lesion.obj`} material={halfTransparentYellow} />
                <Model objPath={`${filename}_lesion.obj`} material={halfTransparentYellow} />
                <Environment preset="studio" background />
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


// import { useEffect } from 'react';
// import * as THREE from 'three';
// import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';

// const ThreeDModelViewer = ({ filename }) => {
//     useEffect(() => {
//         const scene = new THREE.Scene();
//         const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
//         camera.position.z = 10; // Adjust based on the scale of your models

//         const renderer = new THREE.WebGLRenderer({ antialias: true });
//         renderer.setSize(window.innerWidth, window.innerHeight);
//         document.body.appendChild(renderer.domElement);

//         const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
//         scene.add(ambientLight);
//         const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
//         directionalLight.position.set(0, 1, 0);
//         scene.add(directionalLight);

//         const grayMaterial = new THREE.MeshStandardMaterial({ color: 0x808080 });
//         const yellowMaterial = new THREE.MeshStandardMaterial({ color: 0xFFFF00 });

//         const loader = new OBJLoader();

//         function loadModel(objPath, material) {
//             fetch(`http://localhost:8000/obj/${objPath}`)
//                 .then(response => response.blob())
//                 .then(blob => {
//                     const url = URL.createObjectURL(blob);
//                     loader.load(
//                         url,
//                         object => {
//                             object.traverse(function (child) {
//                                 if (child instanceof THREE.Mesh) {
//                                     child.material = material;
//                                 }
//                             });
//                             scene.add(object);
//                         },
//                         xhr => console.log(`${objPath}: ${xhr.loaded / xhr.total * 100}% loaded`),
//                         error => console.log(`An error happened loading ${objPath}`)
//                     );
//                 });
//         }

//         loadModel(filename + '_brain.obj', grayMaterial);
//         loadModel(filename + '_lesion.obj', yellowMaterial);

//         const animate = () => {
//             requestAnimationFrame(animate);
//             renderer.render(scene, camera);
//         };

//         animate();

//         // Cleanup on component unmount
//         return () => {
//             document.body.removeChild(renderer.domElement);
//             scene.clear();
//             renderer.dispose();
//         };
//     }, []);

//     return <div id="3d-model-viewer" />;
// };

// export default ThreeDModelViewer;
