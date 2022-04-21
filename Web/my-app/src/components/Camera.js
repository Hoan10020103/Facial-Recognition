import React, { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";

const Camera = () => {
    const webRef = useRef(null);
    const [img, setImage] = useState(null);
    const Capture = () => {
        setImage(webRef.current.getScreenshot());
    }
    return (
        <div className="cameraForm">
            <Webcam ref={webRef} />
            <button onClick={Capture}>Capture</button>
            <img src={img} alt="" />
        </div>
    );
}

export default Camera;