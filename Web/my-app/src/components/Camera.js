import React, { useState, useRef } from "react";
import Webcam from "react-webcam";

const Camera = (img) => {
    const webRef = useRef(null);
    const [image, setImage] = useState(null);
    const Capture = () => {
        setImage(webRef.current.getScreenshot());
        img = image;
        console.log(img);
    }
    return (
        <div className="cameraForm">
            <Webcam ref={webRef} />
            <div className="welcome">
                <button onClick={Capture}>Capture</button>
            </div>
            <img src={img} alt="" />
        </div>
    );
}

export default Camera;