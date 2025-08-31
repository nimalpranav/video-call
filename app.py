from flask import Flask, render_template, request, session, redirect, url_for
from flask_socketio import SocketIO, join_room, leave_room, emit
import cv2
import numpy as np
import face_recognition
import os
import base64
import datetime
import secrets

# ----- Config -----
app = Flask(__name__)
app.config["SECRET_KEY"] = secrets.token_hex(16)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ----- Faces dir -----
FACES_DIR = "faces"
os.makedirs(FACES_DIR, exist_ok=True)

# ----- Load known faces -----
def load_faces():
    known_faces, known_names = [], []
    for filename in os.listdir(FACES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(FACES_DIR, filename)
            try:
                image = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(image)
                if encs:
                    known_faces.append(encs[0])
                    # store name from filename (without extension)
                    known_names.append(os.path.splitext(filename)[0])
            except Exception as e:
                print("Failed to load face", path, e)
    return known_faces, known_names

KNOWN_FACES, KNOWN_NAMES = load_faces()

# ----- Routes -----
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login", methods=["POST"])
def login():
    global KNOWN_FACES, KNOWN_NAMES
    data = request.json.get("image")
    if not data:
        return {"status": "fail"}

    # strip prefix if any and decode
    b64 = data.split(",", 1)[-1]
    try:
        img_bytes = base64.b64decode(b64)
    except Exception:
        return {"status": "fail"}

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"status": "fail"}

    encs = face_recognition.face_encodings(frame)
    if encs:
        unknown = encs[0]
        matches = face_recognition.compare_faces(KNOWN_FACES, unknown, tolerance=0.5) if KNOWN_FACES else []
        if True in matches:
            name = KNOWN_NAMES[matches.index(True)]
            session["user"] = name
            return {"status": "success", "name": name}
        else:
            # Save new face image with timestamp and reload DB
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"user_{ts}.jpg"
            path = os.path.join(FACES_DIR, filename)
            try:
                cv2.imwrite(path, frame)
            except Exception as e:
                print("Error saving face:", e)
                return {"status": "fail"}
            KNOWN_FACES, KNOWN_NAMES = load_faces()
            return {"status": "new_face", "name": filename}

    return {"status": "fail"}

@app.route("/call")
def call_room():
    user = session.get("user")
    if not user:
        return redirect(url_for("index"))
    room = request.args.get("room", "lobby")
    return render_template("call.html", user=user, room=room)

@app.route("/admin")
def admin():
    user = session.get("user")
    if not user or user != "admin":
        return "Access denied 🚫"

    users = []
    for f in os.listdir(FACES_DIR):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(FACES_DIR, f)
            ts = datetime.datetime.fromtimestamp(os.path.getctime(path))
            users.append({
                "name": os.path.splitext(f)[0],
                "added": ts.strftime("%Y-%m-%d %H:%M:%S")
            })

    return render_template("admin.html", user=user, users=users)

@app.route("/admin/logout_all")
def logout_all():
    if session.get("user") != "admin":
        return "Access denied 🚫"
    socketio.emit("force-logout", {})
    return redirect(url_for("admin"))


@app.route("/admin/delete/<username>")
def admin_delete(username):
    user = session.get("user")
    if user != "admin":
        return "Access denied 🚫"

    filepath = os.path.join(FACES_DIR, f"{username}.jpg")
    if os.path.exists(filepath):
        os.remove(filepath)

    # reload faces after deletion
    global KNOWN_FACES, KNOWN_NAMES
    KNOWN_FACES, KNOWN_NAMES = load_faces()

    return redirect(url_for("admin"))

@app.route("/admin/broadcast", methods=["POST"])
def admin_broadcast():
    user = session.get("user")
    if user != "admin":
        return "Access denied 🚫"

    message = request.form["message"]
    socketio.emit("broadcast", {"message": message})
    return redirect(url_for("admin"))


@app.route("/fail")
def fail():
    return render_template("fail.html")

# ----- Socket.IO signaling -----
@socketio.on("join")
def on_join(data):
    room = data.get("room")
    name = data.get("name")
    join_room(room)
    emit("user-joined", {"name": name}, to=room, include_self=False)

@socketio.on("leave")
def on_leave(data):
    room = data.get("room")
    name = data.get("name")
    leave_room(room)
    emit("user-left", {"name": name}, to=room)

@socketio.on("offer")
def on_offer(data):
    emit("offer", data, to=data.get("room"), include_self=False)

@socketio.on("answer")
def on_answer(data):
    emit("answer", data, to=data.get("room"), include_self=False)

@socketio.on("ice-candidate")
def on_ice_candidate(data):
    emit("ice-candidate", data, to=data.get("room"), include_self=False)

# ----- Run -----
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
