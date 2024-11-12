import pyrebase

# Firebase configuration from your provided values
config = {
    "apiKey": "AIzaSyAmNUWqtHhIrWGPPZbttYV_8FRpG2yAO1g",
    "authDomain": "mastertech-ltu.firebaseapp.com",
    "databaseURL": "https://mastertech-ltu-default-rtdb.asia-southeast1.firebasedatabase.app",
    "projectId": "mastertech-ltu",
    "storageBucket": "mastertech-ltu.firebasestorage.app",
    "messagingSenderId": "180248828825",
    "appId": "1:180248828825:web:228f5bf1b9cafede655108",
    "measurementId": "G-3BE84WN713"
}

# Initialize Pyrebase app
firebase = pyrebase.initialize_app(config)
db = firebase.database()

# Function to read the log file and send data to Firebase
def upload_held_log_to_firebase(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        data = line.strip().split(', ')
        if len(data) == 4:
            record = {
                'ID': data[0],
                'StartTime': data[1],
                'EndTime': data[2],
                'Duration': data[3]
            }
            # Push data to Firebase under 'held_logs' node
            db.child("held_logs").push(record)

# Call the function with the path to your held_log.txt file
upload_held_log_to_firebase('held_log.txt')
