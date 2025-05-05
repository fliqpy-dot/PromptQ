from flask import Flask, jsonify, request, render_template
import os
from queue_system import SimpleQueue  # Your custom FIFO queue class

app = Flask(__name__)

# Ensure templates folder exists
os.makedirs('templates', exist_ok=True)

# In-memory queue and current number being served
fifo_queue = SimpleQueue()
current_number = {"number": "—"}  # Default display for user view

# Admin Page
@app.route('/admin', methods=['GET'])
def admin():
    return render_template('admin.html')

# User Page
@app.route('/user', methods=['GET'])
def user():
    return render_template('user.html')

# Enqueue a new number (admin action)
@app.route("/enqueue", methods=["POST"])
def enqueue():
    data = request.json
    item = data.get("item")
    if not item:
        return jsonify({"error": "Missing item parameter"}), 400
    fifo_queue.enqueue(item)
    return jsonify({"message": f"Enqueued: {item}"}), 200

# Dequeue next number (admin serves someone)
@app.route("/dequeue", methods=["POST"])
def dequeue():
    item = fifo_queue.dequeue()
    if item is not None:
        current_number["number"] = item  # ✅ This updates the live view
        return jsonify({"message": f"Dequeued and now serving: {item}"}), 200
    else:
        current_number["number"] = "—"  # Optional reset
        return jsonify({"message": "Queue is empty!"}), 400

# Clear the queue (admin reset)
@app.route("/clear", methods=["POST"])
def clear():
    fifo_queue.clear()
    current_number["number"] = "—"  # Reset displayed number
    return jsonify({"message": "Cleared FIFO Queue!"}), 200

# View queue status (size only for now)
@app.route("/status", methods=["GET"])
def status():
    size = fifo_queue.size()
    return jsonify({
        "fifo_queue": {
            "size": size,
        }
    })

# Manually update current number (admin override)
@app.route("/api/update-number", methods=["POST"])
def update_number():
    data = request.json
    number = data.get("number")
    if number is None:
        return jsonify({"error": "Missing number"}), 400
    current_number["number"] = number
    return jsonify({"message": "Number updated successfully"}), 200

# User-facing route to fetch currently served number
@app.route("/api/current-number", methods=["GET"])
def get_current_number():
    return jsonify(current_number)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 for local development, Railway will provide its port
    app.run(host="0.0.0.0", port=port, debug=True)
