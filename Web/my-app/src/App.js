import { useState, useEffect } from "react";
import LoginForm from "./components/LoginForm";
import Axios from 'axios';

function App() {
  const [user, setUser] = useState({ username: '', password: '' });
  const [error, setError] = useState('');
  const [userList, setUserList] = useState([]);

  const Login = (details) => {
    console.log(details);
    for (let i = 0; i < userList.length; i++) {
      console.log(userList[i]);
      if (details.username === userList[i].username && details.password === userList[i].passwords) {
        console.log('Login');
        setUser({username: userList[i].username, password: userList[i].password});
        return ;
      }
    }
    console.log('Not match');
    setError("Details do not match");
  }

  const Register = (details) => {
    console.log('register');
    Axios.post('http://localhost:3001/api/insert', {
        username: details.username,
        passwords: details.password,
    }).then(() => {
        alert('operation success !');
    });
    window.location.reload(false);
  }

  useEffect(() => {
    Axios.get('http://localhost:3001/api/get').then((res) => {
      console.log(res.data);
      setUserList(res.data);
    })
  }, [user]);

  const Logout = () => {
    console.log('Logout');
    setUser({ username: '', password: '' });
    setError('');
  }

  return (
    <div className="App">
      {(user.username !== '') ? (
        <div className="welcome">
          <h2>Welcome, <span>{user.username}</span></h2>
          <button onClick={Logout}>Logout</button>
          { userList.map((value) => {
            return <h1>username: {value.username} | password: {value.passwords}</h1> 
          }) }
        </div>
      ) : (
        <LoginForm login={Login} register={Register} error={error} lst={userList} />
      )}
    </div>
  );
}

export default App;
