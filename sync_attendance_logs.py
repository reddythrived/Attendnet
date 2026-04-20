import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime

# Load credentials
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def sync_logs():
    if not (SUPABASE_URL and SUPABASE_KEY):
        print("Supabase credentials missing.")
        return

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Target the historical file found in the legacy project directory
    excel_path = r"C:\Users\Turushotharedy\Desktop\Projects\PCLL\attendance\attendance.xlsx"
    if not os.path.exists(excel_path):
        # Fallback to local if not found
        excel_path = "attendance/attendance.xlsx"
    
    if not os.path.exists(excel_path):
        print(f"Excel file not found at {excel_path}")
        return

    try:
        # Read Excel
        df = pd.read_excel(excel_path)
        
        # 1. Identify Attendance Columns (Dates)
        # We look for columns that look like dates or are NOT standard ID fields
        id_fields = ['name', 'reg_no', 'gmail', 'phone', 'usn', 'email', 'student_id']
        date_cols = [c for c in df.columns if str(c).lower() not in id_fields]
        
        if not date_cols:
            print("No date columns found. Available columns:", df.columns.tolist())
            return

        print(f"Deep Sync initialized. Processing columns: {date_cols}")

        # 2. Extract and Sync
        new_logs_count = 0
        for index, row in df.iterrows():
            usn = str(row['Reg_No']).strip()
            
            # Explicitly Skip 017 as requested
            if usn.upper() == '23BTRCL017':
                continue
            
            # Find student ID
            res = supabase.from_("students").select("id").eq("usn", usn).execute()
            if not res.data:
                continue
            student_id = res.data[0]['id']

            for date_col in date_cols:
                status = str(row[date_col]).strip().lower()
                
                # Check for "Present", "P", "1", "Yes"
                is_present = status in ['present', 'p', '1', 'yes']
                
                final_status = "Present" if is_present else ("Absent" if status != 'nan' else None)
                
                if final_status:
                    # Parse date_col to a standard string
                    try:
                        # Try to parse if it's a date object or a string that looks like a date
                        date_obj = pd.to_datetime(str(date_col))
                        date_str = date_obj.strftime("%Y-%m-%d")
                    except:
                        # Fallback to string if parsing fails
                        date_str = str(date_col)

                    # Check if log already exists for this USN and Date
                    # We check for attendance starting on that date
                    check = supabase.from_("attendance").select("id, status") \
                        .eq("usn", usn) \
                        .gte("marked_at", date_str) \
                        .lte("marked_at", f"{date_str} 23:59:59") \
                        .execute()
                    
                    if not check.data:
                        # Sync to DB
                        supabase.from_("attendance").insert({
                            "student_id": student_id,
                            "usn": usn,
                            "status": final_status,
                            "marked_at": f"{date_str} 09:00:00" # Default morning time for historical logs
                        }).execute()
                        new_logs_count += 1
                    elif check.data[0]['status'] != final_status and final_status == 'Present':
                        # Update if previously marked absent but now present (though rare from excel)
                        pass

        print(f"Success: {new_logs_count} new historical logs added to the dashboard.")
        
    except Exception as e:
        print(f"Sync Error: {e}")

if __name__ == "__main__":
    sync_logs()
