import React, { useState, useRef } from "react";
import Modal from 'react-bootstrap/Modal'
import Webcam from "react-webcam";

const LoginForm = ({ login, register, error }) => {
    const [userDetail, setUserDetail] = useState({ username: '', password: '', pictures: '' });
    const [registerState, setRegisterState] = useState(false);
    const [show, setShow] = useState(false);
    const handleClose = () => setShow(false);
    const handleShow = () => setShow(true);
    const webRef = useRef(null);
    const [image, setImage] = useState(null);
    const Capture = () => {
        setImage(webRef.current.getScreenshot());
        setUserDetail({...userDetail, pictures: image})
        console.log(userDetail);
    }

    const submitHandler = (e) => {
        e.preventDefault();
        if (!registerState) {
            login(userDetail);
        } else {
            register(userDetail);
            setRegisterState(false);
        }
    };

    const myStyle = {
        color: "white",
        backgroundColor: "red",
        padding: "10px",
        marginLeft: "10px",
    };

    const registerClick = () => {
        setRegisterState(true);
    }

    return (
        <div>
          <form id="myform" onSubmit={submitHandler}>
              <div className="form-inner">
                  <h2>Login</h2>
                  {/*Error */
                      (error !== '') ? (
                          <div className="error">{error}</div>
                      ) : null
                  }
                  <div className="form-group">
                      <label htmlFor="username">Name</label>
                      <input type="text" name="username" id="username" onChange={e => setUserDetail({...userDetail, username: e.target.value})} value={userDetail.username} />
                  </div>
                  <div className="form-group">
                      <label htmlFor="password">Password</label>
                      <input type="password" name="password" id="password" onChange={e => setUserDetail({...userDetail, password: e.target.value})} value={userDetail.password} />
                  </div>
                  <button onClick={handleShow}>Register</button>
              </div>
          </form>
          <Modal show={show} onHide={handleClose}>
            <Modal.Body style={{"text-align": "center"}}>
              <div className="cameraForm">
                  <Webcam ref={webRef} />
                  <div className="welcome">
                      <button onClick={Capture}>Capture</button>
                  </div>
              </div>
              <input type="submit" value="." />
              <input style={myStyle} type="submit" form="myform" value="login" />
              <input style={myStyle} form="myform" onClick={registerClick} type="submit" value="register" />
            </Modal.Body>
            <Modal.Footer>
              <div className="welcome">
                  <button variant="secondary" onClick={handleClose}>
                      Close
                  </button>
              </div>
            </Modal.Footer>
          </Modal>
        </div>
    )
}

export default LoginForm;