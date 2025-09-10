from flask import Flask, render_template, jsonify, request
import json
from datetime import datetime
import threading
import time

app = Flask(__name__)

# Global variables to store system data
system_data = {
    "teacher_present": False,
    "last_teacher_arrival": None,
    "last_teacher_departure": None,
    "system_status": "Online",
    "detection_count": 0
}

def update_system_data(teacher_present):
    """Update system data from main application"""
    global system_data
    
    system_data["teacher_present"] = teacher_present
    system_data["detection_count"] += 1
    
    if teacher_present:
        system_data["last_teacher_arrival"] = datetime.now().isoformat()
    else:
        system_data["last_teacher_departure"] = datetime.now().isoformat()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get current system status"""
    return jsonify(system_data)

@app.route('/api/teacher-status')
def get_teacher_status():
    """Get teacher presence status"""
    return jsonify({
        "teacher_present": system_data["teacher_present"],
        "last_arrival": system_data["last_teacher_arrival"],
        "last_departure": system_data["last_teacher_departure"]
    })

def run_dashboard(host='0.0.0.0', port=5000):
    """Run the web dashboard"""
    print(f"🌐 Student Dashboard starting at http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == '__main__':
    run_dashboard()
