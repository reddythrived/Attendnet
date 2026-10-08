import os
import cv2
import numpy as np
import pandas as pd
import base64
import json
import time
import threading
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
TEACHERS_LOCAL_FILE = "attendance/teachers.json"
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
# Supabase Keep-Alive Service
# Prevents Supabase project from going to sleep / pausing due to inactivity
# -----------------------
def supabase_keep_alive_worker():
    """Background worker that queries Supabase every 12 hours to prevent pause/sleep"""
    while True:
        try:
            if supabase:
                # Perform a lightweight ping query
                res = supabase.from_("students").select("id").limit(1).execute()
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✓ Supabase Keep-Alive Ping Executed Successfully.")
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠ Supabase Keep-Alive Warning: {e}")
        # Sleep for 12 hours (43200 seconds)
        time.sleep(43200)

# Start Keep-Alive daemon thread
keep_alive_thread = threading.Thread(target=supabase_keep_alive_worker, daemon=True)
keep_alive_thread.start()

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
        logs = att_res.data or []
        
        for log in logs:
            if not log.get('marked_at'): continue
            date_str = log['marked_at'][:10] # Extract YYYY-MM-DD
            if date_str not in df.columns:
                df[date_str] = "Absent"
            df.loc[df["Reg_No"] == log["usn"], date_str] = log.get("status", "Present")
            
        os.makedirs(os.path.dirname(ATT_FILE), exist_ok=True)
        df.to_excel(ATT_FILE, index=False)
        print("  ✓ Local Excel synced with database.")
    except Exception as e:
        print(f"Error syncing Excel: {e}")

def get_local_teachers():
    """Local fallback storage for teachers if Supabase table is not yet created"""
    if os.path.exists(TEACHERS_LOCAL_FILE):
        try:
            with open(TEACHERS_LOCAL_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_local_teacher(teacher_data):
    """Save teacher to local json backup"""
    os.makedirs(os.path.dirname(TEACHERS_LOCAL_FILE), exist_ok=True)
    teachers = get_local_teachers()
    # Check if duplicate email
    for t in teachers:
        if t['email'].lower() == teacher_data['email'].lower():
            return False
    teachers.append(teacher_data)
    with open(TEACHERS_LOCAL_FILE, 'w') as f:
        json.dump(teachers, f, indent=2)
    return True

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
    # Allow either Admin or Teacher session
    if not session.get("admin") and not session.get("teacher"):
        return redirect(url_for("admin_login", next=url_for("recognition")))
    return render_template("recognition.html")

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    next_page = request.form.get("next") or request.args.get("next")
    if request.method == "POST":
        password = request.form.get("password")
        if password == ADMIN_PASSWORD:
            session["admin"] = True
            # Direct redirect to the destination (e.g. /recognition or /admin)
            if next_page and next_page.strip():
                return redirect(next_page)
            return redirect(url_for("admin_dashboard"))
        return render_template("error.html", error_message="Invalid Admin Password")
    return render_template("admin_login.html", next=next_page)

@app.route("/admin")
def admin_dashboard():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    return render_template("admin.html")

# -----------------------
# Teacher Portal Routes
# -----------------------

@app.route("/teacher/register", methods=["GET", "POST"])
def teacher_register():
    """Enroll a new faculty member. Requires Admin Password for Authorization"""
    if request.method == "GET":
        return render_template("teacher_register.html")

    admin_password = request.form.get("admin_password", "").strip()
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    department = request.form.get("department", "").strip()
    phone = request.form.get("phone", "").strip()
    password = request.form.get("password", "").strip()

    # 1. Verify Admin Password
    if admin_password != ADMIN_PASSWORD:
        return render_template("error.html", error_message="Admin Authorization Failed. Incorrect Admin Password provided for teacher enrollment.")

    if not all([name, email, password]):
        return render_template("error.html", error_message="Name, Email/ID, and Password are required.")

    # 2. Attempt Save to Supabase (with fallback to local storage)
    teacher_record = {
        "name": name,
        "email": email,
        "department": department,
        "phone": phone,
        "password": password
    }

    saved_in_cloud = False
    if supabase:
        try:
            supabase.from_("teachers").insert(teacher_record).execute()
            saved_in_cloud = True
        except Exception as e:
            print(f"Supabase teachers insert warning (using local fallback if needed): {e}")

    # Also save to local fallback
    save_local_teacher(teacher_record)

    return render_template("success.html", name=name, reg_no=f"Teacher: {email}")

@app.route("/teacher/login", methods=["GET", "POST"])
def teacher_login():
    """Teacher authentication portal"""
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        authenticated = False
        teacher_info = None

        # Check Supabase first
        if supabase:
            try:
                res = supabase.from_("teachers").select("id, name, email, department, password").eq("email", email).execute()
                if res.data and len(res.data) > 0:
                    t = res.data[0]
                    if t["password"] == password:
                        authenticated = True
                        teacher_info = t
            except Exception as e:
                print(f"Teacher login Supabase check error: {e}")

        # Check Local fallback if not found in cloud
        if not authenticated:
            local_teachers = get_local_teachers()
            for t in local_teachers:
                if t["email"].lower() == email.lower() and t["password"] == password:
                    authenticated = True
                    teacher_info = t
                    break

        if authenticated and teacher_info:
            session["teacher"] = True
            session["teacher_name"] = teacher_info.get("name", "Faculty")
            session["teacher_email"] = teacher_info.get("email", email)
            session["teacher_dept"] = teacher_info.get("department", "Academics")
            return redirect(url_for("teacher_dashboard"))

        return render_template("error.html", error_message="Invalid Teacher Email/ID or Password.")

    return render_template("teacher_login.html")

@app.route("/teacher/dashboard")
def teacher_dashboard():
    """Teacher Attendance Management & Override Dashboard"""
    if not session.get("teacher") and not session.get("admin"):
        return redirect(url_for("teacher_login"))

    teacher_name = session.get("teacher_name", "Faculty Member")
    teacher_dept = session.get("teacher_dept", "Department")
    return render_template("teacher_dashboard.html", teacher_name=teacher_name, teacher_dept=teacher_dept)

# -----------------------
# Student Portal Routes
# -----------------------

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
    
    try:
        res = supabase.from_("attendance").select("marked_at, status").eq("usn", usn).order("marked_at", desc=True).execute()
        attendance = res.data or []
        
        all_dates_res = supabase.from_("attendance").select("marked_at").execute()
        all_dates = set([d['marked_at'][:10] for d in (all_dates_res.data or []) if d.get('marked_at')])
        
        total_days = len(all_dates) if all_dates else 0
        present_days = len([a for a in attendance if a.get('status') == 'Present'])
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

@app.route("/api/keepalive")
def api_keepalive():
    """Manual or external ping endpoint to ensure Supabase stays active"""
    status = "healthy"
    db_status = "connected"
    if supabase:
        try:
            supabase.from_("students").select("id").limit(1).execute()
        except Exception as e:
            db_status = f"error: {e}"
            status = "degraded"
    else:
        db_status = "unconfigured"
    
    return jsonify({
        "status": status,
        "database": db_status,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/admin/stats")
def get_admin_stats():
    """Get real-time statistics for the Admin Dashboard"""
    if not session.get("admin"):
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    if not supabase:
        return jsonify({"success": False, "message": "Database not connected"}), 500

    try:
        today = datetime.now().strftime("%Y-%m-%d")
        # 1. Fetch all students
        st_res = supabase.from_("students").select("id, name, usn, email").order("usn").execute()
        students = st_res.data or []

        # 2. Fetch today's attendance logs
        att_res = supabase.from_("attendance").select("student_id, usn, status, marked_at") \
            .gte("marked_at", f"{today}T00:00:00") \
            .lte("marked_at", f"{today}T23:59:59") \
            .execute()
        
        logs_map = {log["usn"]: log for log in (att_res.data or [])}

        present_count = 0
        formatted_students = []
        for s in students:
            usn = s["usn"]
            log = logs_map.get(usn)
            is_present = bool(log and log.get("status") == "Present")
            if is_present:
                present_count += 1
            
            formatted_students.append({
                "id": s["id"],
                "name": s["name"],
                "usn": usn,
                "status": log.get("status") if log else "Absent",
                "marked_at": log.get("marked_at") if log else None
            })

        return jsonify({
            "success": True,
            "registered_count": len(students),
            "today_checkins": present_count,
            "date": today,
            "students": formatted_students
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/students/descriptors")
def get_descriptors():
    """Fetch student names, IDs, and face descriptors for web recognition"""
    if not supabase: return jsonify([])
    try:
        res = supabase.from_("students").select("id, name, usn, face_descriptor").execute()
        return jsonify(res.data or [])
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'message'): error_msg = e.message
        return jsonify({"error": error_msg}), 500


@app.route("/api/teacher/attendance")
def get_teacher_attendance():
    """Get all students and their attendance status for a specific date"""
    if not session.get("teacher") and not session.get("admin"):
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    target_date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))

    if not supabase:
        return jsonify({"success": False, "message": "Database not connected"}), 500

    try:
        # 1. Fetch all registered students
        st_res = supabase.from_("students").select("id, name, usn, email").order("usn").execute()
        students = st_res.data or []

        # 2. Fetch attendance logs for the target date
        att_res = supabase.from_("attendance").select("student_id, usn, status, marked_at") \
            .gte("marked_at", f"{target_date}T00:00:00") \
            .lte("marked_at", f"{target_date}T23:59:59") \
            .execute()
        
        logs_map = {log["usn"]: log for log in (att_res.data or [])}

        roster = []
        for s in students:
            usn = s["usn"]
            log = logs_map.get(usn)
            roster.append({
                "id": s["id"],
                "name": s["name"],
                "usn": usn,
                "email": s["email"],
                "status": log["status"] if log else "Absent",
                "marked_at": log["marked_at"] if log else None
            })

        return jsonify({"success": True, "date": target_date, "students": roster})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/teacher/attendance/update", methods=["POST"])
def update_teacher_attendance():
    """Teacher override endpoint: Mark Present or Absent for any student and date"""
    if not session.get("teacher") and not session.get("admin"):
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    data = request.json or {}
    usn = data.get("usn", "").strip().upper()
    date_str = data.get("date", datetime.now().strftime("%Y-%m-%d"))
    status = data.get("status", "Present")

    if not usn or not date_str:
        return jsonify({"success": False, "message": "USN and Date are required"}), 400

    today = datetime.now().strftime("%Y-%m-%d")
    if date_str > today:
        return jsonify({"success": False, "message": "Cannot mark or modify attendance for future dates"}), 400

    if not supabase:
        return jsonify({"success": False, "message": "Database not connected"}), 500

    try:
        # Find student
        st_res = supabase.from_("students").select("id").eq("usn", usn).execute()
        if not st_res.data:
            return jsonify({"success": False, "message": f"Student {usn} not found"}), 404
        
        student_id = st_res.data[0]["id"]

        # Check existing attendance for this date
        check = supabase.from_("attendance").select("id") \
            .eq("usn", usn) \
            .gte("marked_at", f"{date_str}T00:00:00") \
            .lte("marked_at", f"{date_str}T23:59:59") \
            .execute()

        if check.data and len(check.data) > 0:
            # Update existing log
            log_id = check.data[0]["id"]
            supabase.from_("attendance").update({
                "status": status,
                "marked_at": f"{date_str}T{datetime.now().strftime('%H:%M:%S')}Z"
            }).eq("id", log_id).execute()
        else:
            # Insert new log
            supabase.from_("attendance").insert({
                "student_id": student_id,
                "usn": usn,
                "status": status,
                "marked_at": f"{date_str}T{datetime.now().strftime('%H:%M:%S')}Z"
            }).execute()

        # Update local Excel file
        sync_excel_from_db()

        return jsonify({"success": True, "usn": usn, "status": status, "date": date_str})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/attendance/mark", methods=["POST"])
def mark_attendance():
    """Mark attendance from the web recognition page"""
    data = request.json or {}
    student_id = data.get("student_id")
    usn = data.get("usn")
    status = data.get("status", "Present")
    
    if not supabase: return jsonify({"success": False, "message": "Database not connected."})

    try:
        today = datetime.now().strftime("%Y-%m-%d")
        res = supabase.from_("attendance").select("id").eq("usn", usn).gte("marked_at", f"{today}T00:00:00").execute()
        
        if res.data:
            return jsonify({"success": False, "message": "Already marked for today."})

        # Insert attendance record
        supabase.from_("attendance").insert({
            "student_id": student_id,
            "usn": usn,
            "status": status
        }).execute()
        
        # Keep local Excel in sync
        sync_excel_from_db()
        
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
            camera_photos = json.loads(camera_photos_json)
            for i, data_url in enumerate(camera_photos[:NUM_IMAGES]):
                header, b64data = data_url.split(",", 1)
                img_bytes = base64.b64decode(b64data)
                path = f"dataset/{reg_no}/img{i+1}.jpg"
                try:
                    supabase.storage.from_(SUPABASE_BUCKET).upload(
                        path=path,
                        file=img_bytes,
                        file_options={"content-type": "image/jpeg", "upsert": "true"}
                    )
                except Exception as upload_err:
                    print(f"Storage upload error for {path}: {upload_err}")
        else:
            uploaded_files = request.files.getlist("photos")
            for i, file in enumerate(uploaded_files[:NUM_IMAGES]):
                if file and file.filename:
                    file.seek(0)
                    path = f"dataset/{reg_no}/img{i+1}.jpg"
                    try:
                        supabase.storage.from_(SUPABASE_BUCKET).upload(
                            path=path,
                            file=file.read(),
                            file_options={"content-type": "image/jpeg", "upsert": "true"}
                        )
                    except Exception as upload_err:
                        print(f"Storage upload error for {path}: {upload_err}")
                
        # Sync local Excel backup
        sync_excel_from_db()
        
        return render_template("success.html", name=name, reg_no=reg_no)
        
    except Exception as e:
        error_msg = str(e)
        try:
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
    if not session.get("admin") and not session.get("teacher"): 
        return redirect(url_for("admin_login"))
    sync_excel_from_db()
    if os.path.exists(ATT_FILE):
        return send_file(ATT_FILE, as_attachment=True)
    return "Excel file not ready."

@app.route("/logout")
def logout():
    session.pop("admin", None)
    session.pop("teacher", None)
    session.pop("teacher_name", None)
    session.pop("teacher_email", None)
    session.pop("teacher_dept", None)
    session.pop("student_id", None)
    return redirect(url_for("index"))

if __name__ == "__main__":
    os.makedirs(DATASET, exist_ok=True)
    os.makedirs("attendance", exist_ok=True)
    print("Starting AttendNet Server on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
