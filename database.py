import sqlite3, bcrypt

def init_db():
    con = sqlite3.connect("users.db")
    con.execute("""CREATE TABLE IF NOT EXISTS users
        (id INTEGER PRIMARY KEY, username TEXT UNIQUE,
         password TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    con.commit(); con.close()

def register_user(username, password):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    try:
        con = sqlite3.connect("users.db")
        con.execute("INSERT INTO users (username,password) VALUES (?,?)",
                    (username, hashed))
        con.commit(); return True
    except: return False  # username taken

def verify_user(username, password):
    con = sqlite3.connect("users.db")
    row = con.execute("SELECT password FROM users WHERE username=?",
                      (username,)).fetchone()
    if row: return bcrypt.checkpw(password.encode(), row[0])
    return False