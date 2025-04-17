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
QUALITY_NAME = "720p" # !!! এই সার্ভারের জন্য নির্দিষ্ট কোয়ালিটি (480p, 720p পরিবর্তন করুন) !!!

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
         # Return a copy to prevent modification outside the lock
         return conversion_status.get(video_id).copy() if video_id in conversion_status else None

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
    # Ensure segment filenames are unique within the directory
    segment_filename = f"{QUALITY_NAME}_%05d.ts"
    output_playlist_path = os.path.join(output_dir, playlist_name)
    output_segment_path_pattern = os.path.join(output_dir, segment_filename) # Pass pattern to ffmpeg

    ensure_dir(output_dir)

    ffmpeg_cmd = [
        'ffmpeg',
        '-i', source_path,
        '-vf', f'scale=-2:{target_height}',  # Scale height, maintain aspect ratio
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', # Use CRF for better quality/size balance? Or stick to bitrate?
        '-b:v', v_bitrate, # Target video bitrate
        '-maxrate', str(int(v_bitrate.replace('k', '000')) * 1.2) + 'k', # Limit bitrate peaks slightly
        '-bufsize', str(int(v_bitrate.replace('k', '000')) * 2) + 'k',   # VBV buffer size
        '-c:a', 'aac', '-b:a', a_bitrate,
        '-hls_time', '6',             # Segment duration (seconds)
        '-hls_playlist_type', 'vod', # Video on Demand playlist
        '-hls_segment_filename', output_segment_path_pattern, # Use pattern
        '-f', 'hls',
        output_playlist_path
    ]

    logging.info(f"[{thread_name}] Starting ffmpeg for {video_id} ({QUALITY_NAME}). Command: {' '.join(ffmpeg_cmd)}")
    update_status(video_id, status="processing")

    try:
        # Use subprocess.PIPE for stderr to capture logs if needed
        process = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, timeout=timeout, encoding='utf-8', errors='ignore')
        logging.info(f"[{thread_name}] ffmpeg completed successfully for {video_id} ({QUALITY_NAME}).")
        logging.debug(f"[{thread_name}] ffmpeg stdout:\n{process.stdout}")
        logging.debug(f"[{thread_name}] ffmpeg stderr:\n{process.stderr}") # Stderr often contains progress info

        # List generated files *after* successful completion
        generated_files = [f for f in os.listdir(output_dir) if f.endswith('.ts') or f.endswith('.m3u8')]
        if not any(f.endswith('.m3u8') for f in generated_files):
             raise RuntimeError("ffmpeg completed but no playlist file was generated.")
        if not any(f.endswith('.ts') for f in generated_files):
              raise RuntimeError("ffmpeg completed but no segment files were generated.")

        update_status(video_id, status="completed", files=generated_files)
        return True

    except subprocess.CalledProcessError as e:
        error_msg = f"ffmpeg failed for {video_id} ({QUALITY_NAME}) with code {e.returncode}.\nStderr: {e.stderr}"
        logging.error(f"[{thread_name}] {error_msg}")
        update_status(video_id, status="error", error=error_msg)
        return False
    except subprocess.TimeoutExpired as e:
        error_msg = f"ffmpeg timed out after {timeout}s for {video_id} ({QUALITY_NAME}).\nStderr: {e.stderr if e.stderr else 'N/A'}"
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
    # Store HLS files directly in the final output structure
    final_hls_dir = os.path.join(OUTPUT_HLS_DIR, video_id, QUALITY_NAME)

    # Basic filename extraction - improve robustness if needed
    source_filename_base = source_url.split('/')[-1].split('?')[0]
    if not source_filename_base or '.' not in source_filename_base:
         source_filename = f"{video_id}_source_video.mp4" # Default name with extension
         logging.warning(f"Could not determine source filename from URL, using default: {source_filename}")
    else:
         source_filename = source_filename_base # Keep original name if possible
    local_source_path = os.path.join(temp_video_dir, source_filename)

    # Ensure directories exist
    try:
        ensure_dir(temp_video_dir)
        ensure_dir(final_hls_dir) # Create final dir early
    except Exception as e:
         logging.error(f"[{thread_name}] Failed to create directories for {video_id}: {e}")
         update_status(video_id, status="error", error=f"Failed to create processing directories: {e}")
         return # Cannot proceed

    # 1. Download source video
    update_status(video_id, status="downloading")
    download_success = False
    try:
        logging.info(f"[{thread_name}] Downloading source for {video_id} to {local_source_path}...")
        start_dl_time = time.time()
        # Add headers to mimic a browser? Sometimes helps with CDN restrictions
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        with requests.get(source_url, stream=True, timeout=(20, 600), headers=headers) as r: # Increased timeouts (connect, read)
            r.raise_for_status() # Check for HTTP errors
            # Check content length if available
            total_size = int(r.headers.get('content-length', 0))
            if total_size > 0: logging.info(f"[{thread_name}] Download size: {total_size / (1024*1024):.2f} MB")

            bytes_downloaded = 0
            with open(local_source_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192*4): # 32KB chunks
                    if chunk:
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
            end_dl_time = time.time()
            logging.info(f"[{thread_name}] Download complete for {video_id} in {end_dl_time - start_dl_time:.2f}s. Total bytes: {bytes_downloaded}")

        # Verify download
        if not os.path.exists(local_source_path) or os.path.getsize(local_source_path) == 0:
            raise ValueError("Downloaded file is missing or empty after streaming finished.")
        if total_size > 0 and bytes_downloaded < total_size * 0.95: # Check if we got most of the expected size
             logging.warning(f"[{thread_name}] Downloaded size {bytes_downloaded} is significantly less than expected {total_size}. Potential truncation.")
             # Decide whether to proceed or fail based on tolerance

        download_success = True

    except requests.exceptions.Timeout:
        error_msg = f"Timeout downloading source {source_url} for {video_id}"
        logging.error(f"[{thread_name}] {error_msg}")
        update_status(video_id, status="error", error=error_msg)
    except requests.exceptions.HTTPError as e:
         error_msg = f"HTTP Error {e.response.status_code} downloading source {source_url} for {video_id}: {e.response.reason}"
         logging.error(f"[{thread_name}] {error_msg}")
         update_status(video_id, status="error", error=error_msg)
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error downloading source {source_url} for {video_id}: {e}"
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
        cleanup_temp_files(temp_video_dir) # Clean up failed download attempt
        return # Stop processing

    # 2. Run FFmpeg Conversion (Outputs directly to final_hls_dir)
    conversion_success = run_ffmpeg_conversion(
        video_id, local_source_path, final_hls_dir, target_height, v_bitrate, a_bitrate, timeout
    )

    # 3. Cleanup temporary source file/directory *after* conversion attempt
    cleanup_temp_files(temp_video_dir)

    if conversion_success:
         logging.info(f"[{thread_name}] Conversion task finished successfully for {video_id} ({QUALITY_NAME}). Final files in {final_hls_dir}")
    else:
         logging.error(f"[{thread_name}] Conversion task failed for {video_id} ({QUALITY_NAME}).")
         # Ensure status is 'error', error message should be set by run_ffmpeg_conversion
         if get_status(video_id)['status'] != 'error':
             update_status(video_id, status='error', error='Conversion failed (check logs)')
         # Clean up potentially failed HLS output dir?
         cleanup_temp_files(final_hls_dir)


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
    timeout = data.get('timeout', FFMPEG_TIMEOUT) # Use default from main server if provided

    # Basic validation
    if not all([video_id, source_url, target_height, v_bitrate, a_bitrate]):
        logging.error(f"Missing parameters in /convert request: {data}")
        return jsonify({"status": "error", "error": "Missing required parameters"}), 400
    if not isinstance(target_height, int):
         logging.error(f"Invalid target_height type in /convert request: {target_height}")
         return jsonify({"status": "error", "error": "target_height must be an integer"}), 400
    if not isinstance(timeout, int) or timeout <= 0:
         logging.error(f"Invalid timeout value in /convert request: {timeout}")
         return jsonify({"status": "error", "error": "Invalid timeout value"}), 400

    # Check if already processing this ID
    current_stat = get_status(video_id)
    if current_stat and current_stat['status'] not in ['error', 'pending']: # Allow re-processing on error or if never started
         logging.warning(f"Received duplicate request for video_id: {video_id} which is already {current_stat['status']}. Allowing retry.")
         # Optionally return 409 Conflict if you don't want retries:
         # return jsonify({"status": "error", "error": f"Video {video_id} is already {current_stat['status']}"}), 409

    logging.info(f"Received conversion request for {video_id} ({QUALITY_NAME}). Starting background task.")
    update_status(video_id, status="pending") # Mark as pending before starting thread

    # Start background processing task
    processing_thread = threading.Thread(
        target=process_video_task,
        args=(video_id, source_url, target_height, v_bitrate, a_bitrate, timeout),
        name=f"Converter-{QUALITY_NAME}-{video_id[:8]}"
    )
    processing_thread.daemon = True # Allows flask to exit even if thread is running
    processing_thread.start()

    # Return 'Accepted' - main server will poll for status
    return jsonify({"status": "processing_started", "video_id": video_id}), 202


@app.route('/status/<video_id>', methods=['GET'])
def get_conversion_status(video_id):
    """Returns the current status of a conversion job."""
    status_info = get_status(video_id)

    if status_info is None: # If get_status returns None (ID not found)
         # Main server polling before /convert request processed or ID unknown
         logging.warning(f"Status requested for unknown or pending video_id: {video_id}")
         # Return 'pending' to indicate it might start later
         return jsonify({"status": "pending", "error": None}), 200
    else:
         # Status found, return it (could be pending, downloading, processing, completed, error)
         return jsonify({
             "status": status_info.get("status"),
             "error": status_info.get("error") # Include error message if status is 'error'
         }), 200


@app.route('/files/<video_id>', methods=['GET'])
def list_generated_files(video_id):
    """Lists the generated HLS files for collection by the main server."""
    status_info = get_status(video_id)

    if not status_info or status_info.get("status") != "completed":
        error_msg = f"Files not available. Status is {status_info.get('status', 'unknown') if status_info else 'unknown'}."
        if status_info and status_info.get('error'):
             error_msg += f" Error: {status_info.get('error')}"
        # Use 404 Not Found if the process hasn't completed successfully
        return jsonify({"error": error_msg, "files": []}), 404

    # Return the list of files stored in the status dictionary
    files_list = status_info.get("files", [])
    return jsonify({"files": files_list}), 200


@app.route('/files/<video_id>/<path:filename>', methods=['GET'])
def download_generated_file(video_id, filename):
    """Allows the main server to download a specific generated HLS file."""
    status_info = get_status(video_id)

    # Check if conversion completed successfully
    if not status_info or status_info.get("status") != "completed":
        abort(404, description=f"Conversion for {video_id} not completed or failed.")

    # Security: Basic check against path traversal in filename
    safe_filename = filename.replace('../', '')
    if safe_filename != filename:
        logging.error(f"Potential path traversal detected in file download request: {filename}")
        abort(400, description="Invalid filename.")

    # Check if the requested file is in the list of generated files stored in status
    generated_files = status_info.get("files", [])
    if safe_filename not in generated_files:
        logging.warning(f"Attempt to download non-listed or invalid file '{safe_filename}' for video_id {video_id}")
        abort(404, description="File not found or not part of the generated set.")

    # Construct the path to the file within the output HLS directory
    file_directory = os.path.join(OUTPUT_HLS_DIR, video_id, QUALITY_NAME)

    # Check if the actual file exists on disk before sending
    if not os.path.isfile(os.path.join(file_directory, safe_filename)):
         logging.error(f"File {safe_filename} listed in status but not found on disk in {file_directory} for {video_id}")
         abort(404, description="File not found on server.")

    logging.debug(f"Serving file {safe_filename} from {file_directory}")
    # Set appropriate MIME types for HLS
    if safe_filename.endswith('.m3u8'):
        mime_type = 'application/vnd.apple.mpegurl'
    elif safe_filename.endswith('.ts'):
        mime_type = 'video/mp2t'
    else:
        mime_type = None # Let Flask handle it

    try:
        response = send_from_directory(file_directory, safe_filename, mimetype=mime_type)
        # Add CORS headers if needed (main server might be on different domain)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        logging.error(f"Error sending file {safe_filename} from {file_directory}: {e}", exc_info=True)
        abort(500)


# === Initialization ===
if __name__ == '__main__':
    ensure_dir(TEMP_DIR)
    ensure_dir(OUTPUT_HLS_DIR)
    # Use port provided by environment or default
    # !!! প্রতিটি কনভার্টার সার্ভারের জন্য পোর্ট নম্বর পরিবর্তন করুন !!!
    port = int(os.environ.get('PORT', 5001)) # 360p এর জন্য 5001, 480p এর জন্য 5002, 720p এর জন্য 5003
    # Debug=False for production
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True) # Threaded is useful for background tasks
