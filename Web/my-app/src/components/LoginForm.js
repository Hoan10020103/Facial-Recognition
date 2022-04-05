import React from "react";
import { useState } from "react";

const LoginForm = ({login, register, error, lst}) => {
    const [userDetail, setUserDetail] = useState({ username: '', password: '' });
    const [registerState, setRegisterState] = useState(false);
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
        <form onSubmit={submitHandler}>
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
                <input type="submit" value="login" />
                <input style={myStyle} onClick={registerClick} type="submit" value="register" />
            </div>
        </form>
    )
}

export default LoginForm;