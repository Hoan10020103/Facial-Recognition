from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="Chip1018!",
  database="users",
)

print(mydb)

app = Flask(__name__)
CORS(app)

@app.route("/")
def HelloWorld():
    return 'Hello World'

@app.route("/get")
def get_users():
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM info;")
    myresult = mycursor.fetchall()
    return {'users': myresult}


@app.route('/post', methods=['POST'])
def get_query_from_react():
    username = request.json['username']
    password = request.json['password']
    picture = request.json['pictures']
    data = request.get_json()
    print(data)
    print(username)
    print(password)
    mycursor = mydb.cursor()
    sql = "INSERT INTO info (username, passwords, pictures) VALUES (%s, %s, %s);"
    val = (username, password, picture)
    mycursor.execute(sql, val)
    mydb.commit()
    return data


if __name__ == "__main__":
    app.run(debug=True)
