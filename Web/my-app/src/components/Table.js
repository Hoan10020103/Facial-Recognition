import React from 'react';
 
const UserTable = ({users}) => {
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
                <td>{user[1]}</td>
                <td>{user[2]}</td>
                <td><img src={user[3]} style={{"width":"100px", "height":"100px"}} alt="" /></td>
              </tr>
          ))
      }
    </table>
  )
};

export default UserTable;