import os
import subprocess
import requests
import logging
import time
import json
import threading
import shutil
from flask import Flask, request, jsonify, send_from_directory, abort

# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# === Flask App Initialization ===
app = Flask(__name__)

# === Configuration ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'temp_processing')
OUTPUT_HLS_DIR = os.path.join(BASE_DIR, 'output_hls') # Stores final HLS for pickup
QUALITY_NAME = "360p" # Specific to this converter

# === Status Tracking (In-Memory) ===
# Structure: { "video_id": {"status": "...", "error": None, "files": []} }
conversion_status = {}
status_lock = threading.Lock()

# === Helper Functions ===

def ensure_dir(directory):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")
        except OSError as e:
            logging.error(f"Failed to create directory {directory}: {e}")
            raise

def update_status(video_id, status=None, error=None, files=None):
    """Safely updates the status dictionary."""
    with status_lock:
        if video_id not in conversion_status:
            conversion_status[video_id] = {"status": "pending", "error": None, "files": []}

        if status is not None:
            conversion_status[video_id]["status"] = status
        if error is not None:
            # Append errors
            current_error = conversion_status[video_id].get("error")
            if current_error:
                 conversion_status[video_id]["error"] = f"{current_error}\n{error}"
            else:
                 conversion_status[video_id]["error"] = error
        if files is not None:
             conversion_status[video_id]["files"] = files
        logging.info(f"Status updated for {video_id} ({QUALITY_NAME}): {conversion_status[video_id]}")


def get_status(video_id):
     """Safely retrieves the status."""
     with status_lock:
         return conversion_status.get(video_id, {"status": "not_found"}).copy() # Return a copy

def cleanup_temp_files(directory):
    """Removes a temporary directory."""
    if os.path.exists(directory):
        try:
            shutil.rmtree(directory)
            logging.info(f"Cleaned up temporary directory: {directory}")
        except OSError as e:
            logging.error(f"Error cleaning up temporary directory {directory}: {e}")


def run_ffmpeg_conversion(video_id, source_path, output_dir, target_height, v_bitrate, a_bitrate, timeout):
    """Runs the ffmpeg HLS conversion."""
    thread_name = threading.current_thread().name
    playlist_name = f"playlist_{QUALITY_NAME}.m3u8"
    segment_filename = f"{QUALITY_NAME}_%05d.ts" # e.g., 360p_00001.ts
    output_playlist_path = os.path.join(output_dir, playlist_name)
    output_segment_path = os.path.join(output_dir, segment_filename)

    ensure_dir(output_dir)

    ffmpeg_cmd = [
        'ffmpeg',
        '-i', source_path,
        '-vf', f'scale=-2:{target_height}',  # Scale height, maintain aspect ratio
        '-c:v', 'libx264', '-preset', 'medium', '-b:v', v_bitrate,
        '-c:a', 'aac', '-b:a', a_bitrate,
        '-hls_time', '6',             # Segment duration (seconds)
        '-hls_playlist_type', 'vod', # Video on Demand playlist
        '-hls_segment_filename', output_segment_path,
        '-f', 'hls',
        output_playlist_path
    ]

    logging.info(f"[{thread_name}] Starting ffmpeg for {video_id} ({QUALITY_NAME}). Command: {' '.join(ffmpeg_cmd)}")
    update_status(video_id, status="processing")

    try:
        process = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, timeout=timeout)
        logging.info(f"[{thread_name}] ffmpeg completed successfully for {video_id} ({QUALITY_NAME}).")
        logging.debug(f"[{thread_name}] ffmpeg stdout:\n{process.stdout}")
        logging.debug(f"[{thread_name}] ffmpeg stderr:\n{process.stderr}")

        # List generated files
        generated_files = [f for f in os.listdir(output_dir) if f.endswith('.ts') or f.endswith('.m3u8')]
        update_status(video_id, status="completed", files=generated_files)
        return True

    except subprocess.CalledProcessError as e:
        error_msg = f"ffmpeg failed for {video_id} ({QUALITY_NAME}) with code {e.returncode}. Error: {e.stderr}"
        logging.error(f"[{thread_name}] {error_msg}")
        update_status(video_id, status="error", error=error_msg)
        return False
    except subprocess.TimeoutExpired:
        error_msg = f"ffmpeg timed out after {timeout}s for {video_id} ({QUALITY_NAME})."
        logging.error(f"[{thread_name}] {error_msg}")
        update_status(video_id, status="error", error=error_msg)
        return False
    except Exception as e:
        error_msg = f"An unexpected error occurred during ffmpeg conversion for {video_id} ({QUALITY_NAME}): {e}"
        logging.error(f"[{thread_name}] {error_msg}", exc_info=True)
        update_status(video_id, status="error", error=error_msg)
        return False


def process_video_task(video_id, source_url, target_height, v_bitrate, a_bitrate, timeout):
    """Background task to download and convert."""
    thread_name = threading.current_thread().name
    logging.info(f"[{thread_name}] Task started for {video_id} ({QUALITY_NAME}) from {source_url}")

    temp_video_dir = os.path.join(TEMP_DIR, video_id)
    final_hls_dir = os.path.join(OUTPUT_HLS_DIR, video_id, QUALITY_NAME)
    source_filename = os.path.basename(source_url).split('?')[0] # Basic filename extraction
    if not source_filename: source_filename = f"{video_id}_source_video" # Fallback name
    local_source_path = os.path.join(temp_video_dir, source_filename)

    ensure_dir(temp_video_dir)
    ensure_dir(final_hls_dir) # Ensure final dir exists early

    # 1. Download source video
    update_status(video_id, status="downloading")
    download_success = False
    try:
        logging.info(f"[{thread_name}] Downloading source for {video_id} to {local_source_path}...")
        with requests.get(source_url, stream=True, timeout=(15, 300)) as r: # (connect, read)
            r.raise_for_status()
            with open(local_source_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192*4):
                    if chunk: f.write(chunk)
        logging.info(f"[{thread_name}] Download complete for {video_id}.")
        if not os.path.exists(local_source_path) or os.path.getsize(local_source_path) == 0:
            raise ValueError("Downloaded file is missing or empty.")
        download_success = True
    except requests.exceptions.RequestException as e:
        error_msg = f"Failed to download source {source_url} for {video_id}: {e}"
        logging.error(f"[{thread_name}] {error_msg}")
        update_status(video_id, status="error", error=error_msg)
    except (IOError, ValueError) as e:
        error_msg = f"File error during/after download for {video_id}: {e}"
        logging.error(f"[{thread_name}] {error_msg}")
        update_status(video_id, status="error", error=error_msg)
    except Exception as e:
         error_msg = f"Unexpected download error for {video_id}: {e}"
         logging.error(f"[{thread_name}] {error_msg}", exc_info=True)
         update_status(video_id, status="error", error=error_msg)

    if not download_success:
        cleanup_temp_files(temp_video_dir)
        return # Stop processing if download failed

    # 2. Run FFmpeg Conversion
    conversion_success = run_ffmpeg_conversion(
        video_id, local_source_path, final_hls_dir, target_height, v_bitrate, a_bitrate, timeout
    )

    # 3. Cleanup temporary source file
    cleanup_temp_files(temp_video_dir) # Remove the whole temp dir for this video_id

    if conversion_success:
         logging.info(f"[{thread_name}] Conversion task finished successfully for {video_id} ({QUALITY_NAME}).")
    else:
         logging.error(f"[{thread_name}] Conversion task failed for {video_id} ({QUALITY_NAME}).")
         # Ensure status is 'error'
         if get_status(video_id)['status'] != 'error':
             update_status(video_id, status='error', error='Conversion failed (check logs)')


# === Flask Routes ===

@app.route('/convert', methods=['POST'])
def handle_conversion_request():
    """Receives request from main server to start conversion."""
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "error": "Invalid JSON payload"}), 400

    video_id = data.get('video_id')
    source_url = data.get('source_url')
    target_height = data.get('target_height')
    v_bitrate = data.get('video_bitrate')
    a_bitrate = data.get('audio_bitrate')
    timeout = data.get('timeout', 3600) # Default timeout

    if not all([video_id, source_url, target_height, v_bitrate, a_bitrate]):
        return jsonify({"status": "error", "error": "Missing required parameters"}), 400

    # Check if already processing this ID
    current_stat = get_status(video_id)
    if current_stat['status'] not in ['pending', 'not_found', 'error']: # Allow re-processing on error
         logging.warning(f"Received duplicate request for already processing/completed video_id: {video_id}")
         # return jsonify({"status": "error", "error": f"Video {video_id} is already {current_stat['status']}"}), 409 # Conflict
         # OR let it proceed to potentially overwrite/retry? Let's allow retry for now.
         logging.info(f"Allowing re-processing request for video_id: {video_id} (current status: {current_stat['status']})")


    logging.info(f"Received conversion request for {video_id} ({QUALITY_NAME}). Starting background task.")
    update_status(video_id, status="pending") # Mark as pending before starting thread

    # Start background processing
    processing_thread = threading.Thread(
        target=process_video_task,
        args=(video_id, source_url, target_height, v_bitrate, a_bitrate, timeout),
        name=f"Converter-{QUALITY_NAME}-{video_id[:8]}"
    )
    processing_thread.daemon = True # Allow main thread to exit even if tasks are running (consider implications)
    processing_thread.start()

    return jsonify({"status": "processing_started", "video_id": video_id}), 202 # Accepted

@app.route('/status/<video_id>', methods=['GET'])
def get_conversion_status(video_id):
    """Returns the current status of a conversion job."""
    status_info = get_status(video_id)
    if status_info['status'] == 'not_found':
         return jsonify({"status": "not_found", "error": "Video ID not processed by this server or invalid."}), 404
    else:
         # Only return status and error to the main server polling request
         return jsonify({
             "status": status_info.get("status"),
             "error": status_info.get("error")
         })


@app.route('/files/<video_id>', methods=['GET'])
def list_generated_files(video_id):
    """Lists the generated HLS files for collection by the main server."""
    status_info = get_status(video_id)
    if status_info.get("status") != "completed":
        error_msg = f"Files not available. Status is {status_info.get('status', 'unknown')}."
        if status_info.get('error'):
             error_msg += f" Error: {status_info.get('error')}"
        return jsonify({"error": error_msg, "files": []}), 404 # Or 409 Conflict if still processing

    files_list = status_info.get("files", [])
    return jsonify({"files": files_list})


@app.route('/files/<video_id>/<path:filename>', methods=['GET'])
def download_generated_file(video_id, filename):
    """Allows the main server to download a specific generated HLS file."""
    status_info = get_status(video_id)
    if status_info.get("status") != "completed":
        abort(404, description=f"Conversion for {video_id} not completed or failed.")

    # Check if the requested file is in the list of generated files for security
    generated_files = status_info.get("files", [])
    if filename not in generated_files:
        logging.warning(f"Attempt to download non-listed file '{filename}' for video_id {video_id}")
        abort(404, description="File not found or not part of the generated set.")

    file_directory = os.path.join(OUTPUT_HLS_DIR, video_id, QUALITY_NAME)

    if not os.path.exists(os.path.join(file_directory, filename)):
         logging.error(f"File {filename} listed but not found in {file_directory} for {video_id}")
         abort(404, description="File not found on server.")

    logging.debug(f"Serving file {filename} from {file_directory}")
    # Set appropriate MIME types for HLS
    if filename.endswith('.m3u8'):
        mime_type = 'application/vnd.apple.mpegurl'
    elif filename.endswith('.ts'):
        mime_type = 'video/mp2t'
    else:
        mime_type = None # Let Flask handle it

    return send_from_directory(file_directory, filename, mimetype=mime_type)


# === Initialization ===
if __name__ == '__main__':
    ensure_dir(TEMP_DIR)
    ensure_dir(OUTPUT_HLS_DIR)
    # Run on port 5001 for 360p converter
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
