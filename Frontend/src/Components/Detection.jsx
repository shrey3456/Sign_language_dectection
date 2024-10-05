import React, { useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam';
import './Detection.css'; // Import the CSS file

const Detection = () => {
  const [hasPermission, setHasPermission] = useState(false);
  const [mode, setMode] = useState('live'); // Default mode is 'live'
  const [selectedImage, setSelectedImage] = useState(null); // State for uploaded image
  const webcamRef = useRef(null);

  useEffect(() => {
    const requestCameraPermission = async () => {
      try {
        await navigator.mediaDevices.getUserMedia({ video: true });
        setHasPermission(true);
      } catch (err) {
        console.error("Camera permission denied:", err);
        setHasPermission(false);
      }
    };

    if (mode === 'live') {
      requestCameraPermission();
    }
  }, [mode]); // Only request permission when switching to live mode

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(URL.createObjectURL(file)); // Create a URL for the image
      setMode('upload'); // Switch to upload mode
    }
  };

  const handleUpload = () => {
    // Here you would handle the image upload to your ML model
    console.log("Image uploaded:", selectedImage);
  };

  return (
    <div className="detection-container">
      <h2>Select Detection Mode</h2>
      <button onClick={() => setMode('live')}>Use Webcam</button>
      <button onClick={() => setMode('upload')}>Upload Image</button>

      {mode === 'live' && hasPermission ? (
        <Webcam
          audio={false}
          ref={webcamRef}
          style={{
            width: '100%',
            height: 'auto',
            transform: 'scaleX(-1)', // Mirror effect
            overflow: 'hidden'
          }}
        />
      ) : mode === 'live' && !hasPermission ? (
        <div className="permission-message">
          <p>Please allow camera access to use this feature.</p>
          <button onClick={() => window.location.reload()}>Retry</button>
        </div>
      ) : (
        <div>
          <input type="file" accept="image/*" onChange={handleImageChange} />
          {selectedImage && (
            <div>
              <h3>Selected Image:</h3>
              <img src={selectedImage} alt="Selected" />
              <button onClick={handleUpload}>Upload</button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Detection;
