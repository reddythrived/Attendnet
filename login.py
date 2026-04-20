import os
import cv2
import numpy as np
import pandas as pd
import base64
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "secure_attendnet_key") # Read from env in production

# Constants
DATASET = "dataset"
ATT_FILE = "attendance/attendance.xlsx"
NUM_IMAGES = 6

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "student-dataset")

# Initialize Supabase client
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Connected to Supabase Successfully!")
    except Exception as e:
        print(f"Error connecting to Supabase: {e}")

# -----------------------
# Utility Functions
# -----------------------

def sync_excel_from_db():
    """Sync the Supabase Database to the local Excel file for backup/export"""
    if not supabase: return
    
    try:
        # Fetch all students and their attendance
        res = supabase.from_("students").select("name, usn, email, phone").execute()
        students = res.data
        
        if not students: return
        
        df = pd.DataFrame(students)
        df.columns = ["Name", "Reg_No", "Gmail", "Phone"]
        
        # Fetch all attendance logs
        att_res = supabase.from_("attendance").select("usn, marked_at, status").execute()
        logs = att_res.data
        
        for log in logs:
            date_str = log['marked_at'][:10] # Extract YYYY-MM-DD
            if date_str not in df.columns:
                df[date_str] = "Absent"
            df.loc[df["Reg_No"] == log["usn"], date_str] = log["status"]
            
        os.makedirs(os.path.dirname(ATT_FILE), exist_ok=True)
        df.to_excel(ATT_FILE, index=False)
        print("  ✓ Local Excel synced with database.")
    except Exception as e:
        print(f"Error syncing Excel: {e}")

# -----------------------
# Web Routes
# -----------------------

@app.route("/")
def index():
    """Landing Page: Role Selection"""
    return render_template("index.html")

@app.route("/recognition")
def recognition():
    """Page for marking attendance using AI - High Security Area"""
    # Force a fresh check for admin to avoid sticky sessions
    if not session.get("admin"):
        return redirect(url_for("admin_login", next=url_for("recognition")))
    return render_template("recognition.html")

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        password = request.form.get("password")
        if password == ADMIN_PASSWORD:
            session["admin"] = True
            # Handle redirection back to where they came from
            next_page = request.args.get("next")
            return redirect(next_page or url_for("admin_dashboard"))
        return render_template("error.html", error_message="Invalid Admin Password")
    return render_template("admin_login.html")

@app.route("/admin")
def admin_dashboard():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    return render_template("admin.html")

@app.route("/student/login", methods=["GET", "POST"])
def student_login():
    if request.method == "POST":
        usn = request.form.get("usn", "").strip().upper()
        password = request.form.get("password", "").strip()
        
        if not supabase: return render_template("error.html", error_message="Database error.")
        
        try:
            res = supabase.from_("students").select("id, name, usn, password").eq("usn", usn).execute()
            if not res.data:
                return render_template("error.html", error_message="USN not registered.")
            
            student = res.data[0]
            # Check password (plain text as requested for 'student1', or hashed)
            stored_pwd = student.get("password", "student1")
            
            if password == stored_pwd or (stored_pwd.startswith("pbkdf2:sha256") and check_password_hash(stored_pwd, password)):
                session["student_id"] = student["id"]
                session["student_name"] = student["name"]
                session["student_usn"] = student["usn"]
                return redirect(url_for("student_dashboard"))
            
            return render_template("error.html", error_message="Invalid password.")
        except Exception as e:
            return render_template("error.html", error_message=str(e))
            
    return render_template("student_login.html")

@app.route("/student-dashboard")
def student_dashboard():
    if not session.get("student_id"):
        return redirect(url_for("student_login"))
    
    usn = session.get("student_usn")
    
    # Fetch actual attendance for this student
    try:
        # Get all attendance records for this student
        res = supabase.from_("attendance").select("marked_at, status").eq("usn", usn).order("marked_at", desc=True).execute()
        attendance = res.data
        
        # Calculate Stats
        # For Total G-Days, we count UNIQUE dates present in the attendance table for ANY student (global school days)
        all_dates_res = supabase.from_("attendance").select("marked_at").execute()
        all_dates = set([d['marked_at'][:10] for d in (all_dates_res.data or [])])
        
        # We ensure a minimum total_days (baseline) so the dashboard feels established
        # If there are only a few dates, we assume at least a session count based on total table activity
        total_days = len(all_dates) if all_dates else 0
        
        # Specific Present count for THIS student
        present_days = len([a for a in attendance if a['status'] == 'Present'])
        
        # Accurate Percentage Calculation
        attendance_percentage = (present_days / total_days * 100) if total_days > 0 else 0
        
    except Exception as e:
        print(f"Stats Error: {e}")
        attendance = []
        total_days = 0
        present_days = 0
        attendance_percentage = 0
        
    return render_template("student_dashboard.html", 
                           name=session.get("student_name"), 
                           usn=session.get("student_usn"),
                           attendance=attendance,
                           total_days=total_days,
                           present_days=present_days,
                           percentage=round(attendance_percentage, 1))

# -----------------------
# API Endpoints
# -----------------------

@app.route("/api/students/descriptors")
def get_descriptors():
    """Fetch student names, IDs, and face descriptors for web recognition"""
    if not supabase: return jsonify([])
    try:
        res = supabase.from_("students").select("id, name, usn, face_descriptor").execute()
        return jsonify(res.data)
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'message'): error_msg = e.message
        return jsonify({"error": error_msg}), 500

@app.route("/api/students/photos/<usn>")
def get_student_photos(usn):
    """List local photos for a USN to assist in cloud sync"""
    user_path = os.path.join(DATASET, usn)
    if not os.path.exists(user_path):
        return jsonify([])
    
    photos = [f for f in os.listdir(user_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return jsonify(photos)

@app.route("/dataset/<usn>/<filename>")
def serve_dataset_photo(usn, filename):
    """Serve a specific student photo from the local dataset folder"""
    return send_file(os.path.join(DATASET, usn, filename))

@app.route("/api/students/update_descriptor", methods=["POST"])
def update_descriptor():
    """Update a specific student's face descriptor in the cloud"""
    data = request.json
    usn = data.get("usn")
    descriptor = data.get("descriptor")
    
    if not supabase: return jsonify({"success": False, "message": "No database connection"})
    
    try:
        supabase.from_("students").update({"face_descriptor": descriptor}).eq("usn", usn).execute()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route("/api/attendance/mark", methods=["POST"])
def mark_attendance():
    """Mark attendance from the web recognition page"""
    data = request.json
    student_id = data.get("student_id")
    usn = data.get("usn")
    status = data.get("status", "Present")
    
    if not supabase: return jsonify({"success": False, "message": "Database not connected."})

    try:
        # Check if already marked today
        today = datetime.now().strftime("%Y-%m-%d")
        # Match by USN and date
        # Note: In Supabase marked_at is timestamptz, so we check for the date part
        res = supabase.from_("attendance").select("id").eq("usn", usn).gte("marked_at", today).execute()
        
        if res.data:
            return jsonify({"success": False, "message": "Already marked for today."})

        # Insert attendance record
        supabase.from_("attendance").insert({
            "student_id": student_id,
            "usn": usn,
            "status": status
        }).execute()
        
        return jsonify({"success": True})
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'message'): error_msg = e.message
        return jsonify({"success": False, "message": error_msg})

@app.route("/register", methods=["GET", "POST"])
def register():
    """Handle both serving the registration form and processing it"""
    if request.method == "GET":
        return render_template("register.html")
    
    name = request.form.get("name", "").strip()
    reg_no = request.form.get("reg_no", "").strip().upper()
    email = request.form.get("email", "").strip()
    phone = request.form.get("phone", "").strip()
    password = request.form.get("password", "").strip() or "student1"
    face_descriptor = request.form.get("face_descriptor") # JSON string from client
    
    if not all([name, reg_no, email, phone, face_descriptor]):
        return render_template("error.html", error_message="All fields including face data are mandatory.")

    if not supabase: return render_template("error.html", error_message="Database connection error.")

    try:
        # 1. Register in Supabase Database
        descriptor_list = json.loads(face_descriptor)
        res = supabase.from_("students").insert({
            "name": name,
            "usn": reg_no,
            "email": email,
            "phone": phone,
            "password": password,
            "face_descriptor": descriptor_list
        }).execute()
        
        # 2. Upload photos to Supabase Storage
        camera_photos_json = request.form.get("camera_photos", "")
        
        if camera_photos_json:
            # Camera capture mode: decode base64 data URLs
            camera_photos = json.loads(camera_photos_json)
            for i, data_url in enumerate(camera_photos[:NUM_IMAGES]):
                # Strip the "data:image/jpeg;base64," prefix
                header, b64data = data_url.split(",", 1)
                img_bytes = base64.b64decode(b64data)
                path = f"dataset/{reg_no}/img{i+1}.jpg"
                supabase.storage.from_(SUPABASE_BUCKET).upload(
                    path=path,
                    file=img_bytes,
                    file_options={"content-type": "image/jpeg", "upsert": "true"}
                )
        else:
            # File upload mode
            uploaded_files = request.files.getlist("photos")
            for i, file in enumerate(uploaded_files[:NUM_IMAGES]):
                if file and file.filename:
                    file.seek(0)
                    path = f"dataset/{reg_no}/img{i+1}.jpg"
                    supabase.storage.from_(SUPABASE_BUCKET).upload(
                        path=path,
                        file=file.read(),
                        file_options={"content-type": "image/jpeg", "upsert": "true"}
                    )
                
        # Sync local Excel backup
        sync_excel_from_db()
        
        return render_template("success.html", name=name, reg_no=reg_no)
        
    except Exception as e:
        error_msg = str(e)
        try:
            # Try to parse Supabase error objects
            if hasattr(e, 'message'): 
                error_msg = e.message
            elif isinstance(e.args[0], dict):
                error_msg = e.args[0].get('message', str(e))
        except:
            pass
            
        if "duplicate" in error_msg.lower():
            return render_template("error.html", error_message=f"USN {reg_no} is already registered.")
        return render_template("error.html", error_message=error_msg)

@app.route("/api/admin/export")
def export_excel():
    """Export the latest database state to Excel and download"""
    if not session.get("admin"): return redirect(url_for("admin_login"))
    sync_excel_from_db()
    if os.path.exists(ATT_FILE):
        return send_file(ATT_FILE, as_attachment=True)
    return "Excel file not ready."

@app.route("/logout")
def logout():
    session.pop("admin", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    os.makedirs(DATASET, exist_ok=True)
    # Important: host='0.0.0.0' allows mobile devices on same Wi-Fi to connect
    app.run(host="0.0.0.0", port=5000, debug=True)
