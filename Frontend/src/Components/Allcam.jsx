// import React, { useEffect, useState } from 'react';
// import Webcam from 'react-webcam';

// const Allcam = () => {
//   const [hasPermission, setHasPermission] = useState(false);

//   useEffect(() => {
//     // Check webcam permission
//     navigator.mediaDevices.getUserMedia({ video: true })
//       .then(stream => {
//         console.log("Webcam permission granted");
//         setHasPermission(true);  // Set permission state
//       })
//       .catch(err => {
//         console.error("Webcam permission denied", err);
//         setHasPermission(false);  // Handle permission denial
//       });
//   }, []);

//   return (
//     <>
//       {hasPermission ? (
//         <Webcam
//           audio={false}
//           style={{ width: '100%', height: 'auto' }}
//         />
//       ) : (
//         <p>Waiting for webcam permissions or webcam not available...</p>
//       )}
//     </>
//   );
// };

// export default Allcam;
import React, { useEffect, useState } from 'react';
import Webcam from 'react-webcam';

const Allcam = () => {
  const [hasPermission, setHasPermission] = useState(false);

  useEffect(() => {
    // Check webcam permission
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        console.log("Webcam permission granted", stream);
        setHasPermission(true);  // Set permission state
      })
      .catch(err => {
        console.error("Webcam permission denied or error", err);
        setHasPermission(false);  // Handle permission denial
      });
  }, []);

  return (
    <>
      {hasPermission ? (
        <Webcam
          audio={false}
          style={{ 
            width: '100%', 
            height: 'auto', 
            transform: 'scaleX(-1)',  // Mirror effect
            overflow: 'hidden' 
          }}
        />
      ) : (
        <p>Waiting for webcam permissions or webcam not available...</p>
      )}
    </>
  );
};

export default Allcam;
