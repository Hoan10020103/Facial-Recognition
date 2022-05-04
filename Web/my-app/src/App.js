import { useState, useEffect } from "react";
import LoginForm from "./components/LoginForm";
import UserTable from "./components/Table";
import Axios from 'axios';

function App() {
  const [user, setUser] = useState({ username: '', password: '' });
  const [error, setError] = useState('');
  const [userList, setUserList] = useState([]);

  function handlePostQuery(query){
    if (query !== "") {
      Axios.post('http://localhost:5000/post', query)
          .then(function(response){
              console.log(response);
              window.location.reload(false);
      //Perform action based on response
      })
      .catch(function(error){
          console.log(error);
      //Perform action based on error
      });
    } else {
      alert("The search query cannot be empty")
    }
  }

  const Login = (details) => {
    console.log(details);
    for (let i = 0; i < userList.users.length; i++) {
      if (details.username === userList.users[i][1] && details.password === userList.users[i][2]) {
        console.log('Login');
        setUser({username: details.username, password: details.password});
        return ;
      }
    }
    console.log('Not match');
    setError("Details do not match");
  }

  const Register = (details) => {
    console.log('register');
    console.log(details);
    handlePostQuery({ username: details.username, password: details.password })
  }

  useEffect(() => {
    Axios.get('http://localhost:5000/get').then((res) => {
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
          <UserTable users={userList.users} />
        </div>
      ) : (
        <LoginForm login={Login} register={Register} error={error} lst={userList} />
      )}
    </div>
  );
}


export default App;
