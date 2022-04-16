import React, { useEffect } from 'react';
import { useTable } from 'react-table';
import { useMemo } from 'react';
 
const UserTable = ({users}) => {

  const data = [{
      username: 'johnny',
      passwords: '123',
  }]

  useEffect(() => {
      console.log(users);
      users.forEach(element => {
        console.log(element);
      });
  });

  const columns = useMemo(
    () => [
      {
        Header: 'Username',
        accessor: 'username', // accessor is the "key" in the data
      },
      {
        Header: 'Passwords',
        accessor: 'passwords',
      },
    ],
    []
  );

  // Render the UI for your table
  return (
    <table>
      <tr>
        <th>Username</th>
        <th>Password</th>
        <th>Picture</th>
      </tr>
      {
          users.map((user) => (
              <tr>
                <td>{user.username}</td>
                <td>{user.passwords}</td>
                <td><img src="../../image/natsu.jpg" /></td>
              </tr>
          ))
      }
    </table>
  )
};

export default UserTable;