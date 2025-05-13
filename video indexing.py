# Install primary dependencies
!pip install whisper-openai opencv-python face_recognition gradio spacy faiss-cpu numpy sentence-transformers

# Install YOLOv8 for object detection
!pip install ultralytics

# Download the spaCy language model
!python -m spacy download en_core_web_sm

# For Google Colab, ensure ffmpeg is installed
!apt-get update && apt-get install -y ffmpeg

import whisper
# Explains: import whisper
import cv2
# Explains: import cv2
import face_recognition
# Explains: import face_recognition
import gradio as gr
# Explains: import gradio as gr
import os
# Explains: import os
import json
# Explains: import json
import spacy
# Explains: import spacy
import faiss
# Explains: import faiss
import numpy as np
# Explains: import numpy as np
from datetime import timedelta
# Explains: from datetime import timedelta
from sentence_transformers import SentenceTransformer
# Explains: from sentence_transformers import SentenceTransformer

# Initialize models
try:
# Explains: try:
    nlp = spacy.load("en_core_web_sm")
    # Explains: nlp = spacy.load("en_core_web_sm")
except OSError:
# Explains: except OSError:
    # Download the model if not already installed
    spacy.cli.download("en_core_web_sm")
    # Explains: spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    # Explains: nlp = spacy.load("en_core_web_sm")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
# Explains: embedder = SentenceTransformer("all-MiniLM-L6-v2")

def transcribe_audio(video_path):
# Explains: def transcribe_audio(video_path):
    """Transcribe audio from video using Whisper with word-level timestamps"""
    # Explains: """Transcribe audio from video using Whisper with word-level timestamps"""
    model = whisper.load_model("base")
    # Explains: model = whisper.load_model("base")

    # Corrected line: Remove the unsupported 'word_timestamps' argument
    result = model.transcribe(video_path)
    # Explains: result = model.transcribe(video_path)

    # Extract word-level segments
    word_segments = []
    # Explains: word_segments = []

    # Process word timestamps if available
    if "words" in result:

        for word_data in result["words"]:
        # Explains: for word_data in result["words"]:
            word_segments.append({
            # Explains: word_segments.append({
                "start": str(timedelta(seconds=int(word_data["start"]))),
                # Explains: "start": str(timedelta(seconds=int(word_data["start"]))),
                "end": str(timedelta(seconds=int(word_data["end"]))),
                # Explains: "end": str(timedelta(seconds=int(word_data["end"]))),
                "start_seconds": word_data["start"],
                # Explains: "start_seconds": word_data["start"],
                "end_seconds": word_data["end"],
                # Explains: "end_seconds": word_data["end"],
                "text": word_data["word"]
                # Explains: "text": word_data["word"]
            })
            # [EXPLANATION NEEDED]
    else:
    # Explains: else:
        # Fallback to sentence-level if word timestamps not available
        # (Some Whisper versions might not support word timestamps)
        for seg in result["segments"]:
        # Explains: for seg in result["segments"]:
            text = seg["text"]
            # Explains: text = seg["text"]
            start = str(timedelta(seconds=int(seg["start"])))
            # Explains: start = str(timedelta(seconds=int(seg["start"])))
            end = str(timedelta(seconds=int(seg["end"])))
            # Explains: end = str(timedelta(seconds=int(seg["end"])))

            # Split the text into words (simple split)
            words = text.split()
            # Explains: words = text.split()

            # Estimate duration per word and create word segments
            if words:
            # Explains: if words:
                duration_per_word = (seg["end"] - seg["start"]) / len(words)
                # Explains: duration_per_word = (seg["end"] - seg["start"]) / len(words)

                for i, word in enumerate(words):
                # Explains: for i, word in enumerate(words):
                    word_start = seg["start"] + (i * duration_per_word)
                    # Explains: word_start = seg["start"] + (i * duration_per_word)
                    word_end = word_start + duration_per_word
                    # Explains: word_end = word_start + duration_per_word

                    word_segments.append({
                    # Explains: word_segments.append({
                        "start": str(timedelta(seconds=int(word_start))),
                        # Explains: "start": str(timedelta(seconds=int(word_start))),
                        "end": str(timedelta(seconds=int(word_end))),
                        # Explains: "end": str(timedelta(seconds=int(word_end))),
                        "start_seconds": word_start,
                        # Explains: "start_seconds": word_start,
                        "end_seconds": word_end,
                        # Explains: "end_seconds": word_end,
                        "text": word
                        # Explains: "text": word
                    })
                    # [EXPLANATION NEEDED]

    return word_segments
    # Explains: return word_segments

def extract_named_entities(texts):
# Explains: def extract_named_entities(texts):
    """Extract named entities from transcript segments"""
    # Explains: """Extract named entities from transcript segments"""
    all_entities = []
    # Explains: all_entities = []

    # Group words into chunks for better entity recognition
    chunk_size = 10
    # Explains: chunk_size = 10
    chunks = []
    # Explains: chunks = []
    current_chunk = {"text": "", "start_seconds": None, "end_seconds": None}
    # Explains: current_chunk = {"text": "", "start_seconds": None, "end_seconds": None}

    for i, t in enumerate(texts):
    # Explains: for i, t in enumerate(texts):
        if i % chunk_size == 0 and i > 0:
        # Explains: if i % chunk_size == 0 and i > 0:
            # Save the current chunk and start a new one
            chunks.append(current_chunk)
            # Explains: chunks.append(current_chunk)
            current_chunk = {"text": t["text"], "start_seconds": t["start_seconds"], "end_seconds": t["end_seconds"]}
            # Explains: current_chunk = {"text": t["text"], "start_seconds": t["start_seconds"], "end_seconds": t["end_seconds"]}
        else:
        # Explains: else:
            # Add to current chunk
            if current_chunk["start_seconds"] is None:
            # Explains: if current_chunk["start_seconds"] is None:
                current_chunk["start_seconds"] = t["start_seconds"]
                # Explains: current_chunk["start_seconds"] = t["start_seconds"]

            current_chunk["text"] += " " + t["text"]
            # Explains: current_chunk["text"] += " " + t["text"]
            current_chunk["end_seconds"] = t["end_seconds"]
            # Explains: current_chunk["end_seconds"] = t["end_seconds"]

    # Add the last chunk if it exists
    if current_chunk["text"]:
    # Explains: if current_chunk["text"]:
        chunks.append(current_chunk)
        # Explains: chunks.append(current_chunk)

    # Process chunks for entities
    for chunk in chunks:
    # Explains: for chunk in chunks:
        doc = nlp(chunk["text"])
        # Explains: doc = nlp(chunk["text"])
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        # Explains: entities = [(ent.text, ent.label_) for ent in doc.ents]

        if entities:  # Only add if entities exist
        # Explains: if entities:  # Only add if entities exist
            all_entities.append({
            # Explains: all_entities.append({
                "start": str(timedelta(seconds=int(chunk["start_seconds"]))),
                # Explains: "start": str(timedelta(seconds=int(chunk["start_seconds"]))),
                "end": str(timedelta(seconds=int(chunk["end_seconds"]))),
                # Explains: "end": str(timedelta(seconds=int(chunk["end_seconds"]))),
                "start_seconds": chunk["start_seconds"],
                # Explains: "start_seconds": chunk["start_seconds"],
                "end_seconds": chunk["end_seconds"],
                # Explains: "end_seconds": chunk["end_seconds"],
                "entities": entities
                # Explains: "entities": entities
            })
            # [EXPLANATION NEEDED]

    return all_entities
    # Explains: return all_entities

def detect_objects(video_path):
# Explains: def detect_objects(video_path):
    """Detect objects in video using YOLOv8"""
    # Explains: """Detect objects in video using YOLOv8"""
    try:
    # Explains: try:
        from ultralytics import YOLO
        # Explains: from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        # Explains: model = YOLO("yolov8n.pt")
    except ImportError:
    # Explains: except ImportError:
        raise ImportError("ultralytics package not installed. Please install with 'pip install ultralytics'")
        # Explains: raise ImportError("ultralytics package not installed. Please install with 'pip install ultralytics'")

    cap = cv2.VideoCapture(video_path)
    # Explains: cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
    # Explains: if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        # Explains: raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    # Explains: fps = cap.get(cv2.CAP_PROP_FPS)
    frame_id = 0
    # Explains: frame_id = 0
    results = []
    # Explains: results = []

    while cap.isOpened():
    # Explains: while cap.isOpened():
        ret, frame = cap.read()
        # Explains: ret, frame = cap.read()
        if not ret:
        # Explains: if not ret:
            break
            # Explains: break

        # Process every 5 seconds
        if frame_id % int(fps * 5) == 0:
        # Explains: if frame_id % int(fps * 5) == 0:
            seconds = frame_id // fps
            # Explains: seconds = frame_id // fps
            timestamp = str(timedelta(seconds=int(seconds)))
            # Explains: timestamp = str(timedelta(seconds=int(seconds)))

            # YOLOv8 detection
            r = model(frame, verbose=False)
            # Explains: r = model(frame, verbose=False)

            # Extract object names
            labels = []
            # Explains: labels = []
            if r and len(r) > 0:
            # Explains: if r and len(r) > 0:
                boxes = r[0].boxes
                # Explains: boxes = r[0].boxes
                if boxes is not None and len(boxes) > 0:
                # Explains: if boxes is not None and len(boxes) > 0:
                    cls_tensor = boxes.cls
                    # Explains: cls_tensor = boxes.cls
                    if cls_tensor is not None and len(cls_tensor) > 0:
                    # Explains: if cls_tensor is not None and len(cls_tensor) > 0:
                        cls_values = cls_tensor.cpu().numpy()
                        # Explains: cls_values = cls_tensor.cpu().numpy()
                        labels = list(set([r[0].names[int(c)] for c in cls_values]))
                        # Explains: labels = list(set([r[0].names[int(c)] for c in cls_values]))

            if labels:
            # Explains: if labels:
                results.append({
                # Explains: results.append({
                    "time": timestamp,
                    # Explains: "time": timestamp,
                    "seconds": seconds,
                    # Explains: "seconds": seconds,
                    "objects": labels
                    # Explains: "objects": labels
                })
                # [EXPLANATION NEEDED]

        frame_id += 1
        # Explains: frame_id += 1

    cap.release()
    # Explains: cap.release()
    return results
    # Explains: return results

def detect_faces(video_path):
# Explains: def detect_faces(video_path):
    """Detect faces in video frames"""
    # Explains: """Detect faces in video frames"""
    cap = cv2.VideoCapture(video_path)
    # Explains: cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
    # Explains: if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
        # Explains: raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    # Explains: fps = cap.get(cv2.CAP_PROP_FPS)
    frame_id = 0
    # Explains: frame_id = 0
    results = []
    # Explains: results = []

    while cap.isOpened():
    # Explains: while cap.isOpened():
        ret, frame = cap.read()
        # Explains: ret, frame = cap.read()
        if not ret:
        # Explains: if not ret:
            break
            # Explains: break

        # Process every 5 seconds
        if frame_id % int(fps * 5) == 0:
        # Explains: if frame_id % int(fps * 5) == 0:
            seconds = frame_id // fps
            # Explains: seconds = frame_id // fps
            timestamp = str(timedelta(seconds=int(seconds)))
            # Explains: timestamp = str(timedelta(seconds=int(seconds)))

            # Convert BGR to RGB (face_recognition requires RGB)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Explains: rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Use face_recognition library
            face_locations = face_recognition.face_locations(rgb, model="hog")
            # Explains: face_locations = face_recognition.face_locations(rgb, model="hog")

            if face_locations:
            # Explains: if face_locations:
                # Save the frame with faces for reference
                frame_filename = f"face_frame_{seconds}.jpg"
                # Explains: frame_filename = f"face_frame_{seconds}.jpg"
                frame_path = os.path.join("video_analysis_results", frame_filename)
                # Explains: frame_path = os.path.join("video_analysis_results", frame_filename)
                os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                # Explains: os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                cv2.imwrite(frame_path, frame)
                # Explains: cv2.imwrite(frame_path, frame)

                results.append({
                # Explains: results.append({
                    "time": timestamp,
                    # Explains: "time": timestamp,
                    "seconds": seconds,
                    # Explains: "seconds": seconds,
                    "faces_detected": len(face_locations),
                    # Explains: "faces_detected": len(face_locations),
                    "frame_file": frame_filename
                    # Explains: "frame_file": frame_filename
                })
                # [EXPLANATION NEEDED]

        frame_id += 1
        # Explains: frame_id += 1

    cap.release()
    # Explains: cap.release()
    return results
    # Explains: return results

def semantic_indexing(segments):
# Explains: def semantic_indexing(segments):
    """Create semantic index of transcript segments for search"""
    # Explains: """Create semantic index of transcript segments for search"""
    # Since we have word-level segments, we'll group them into meaningful chunks for indexing
    chunk_size = 10
    # Explains: chunk_size = 10
    chunks = []
    # Explains: chunks = []
    current_chunk = {"text": "", "start_seconds": None, "end_seconds": None}
    # Explains: current_chunk = {"text": "", "start_seconds": None, "end_seconds": None}

    for i, segment in enumerate(segments):
    # Explains: for i, segment in enumerate(segments):
        if i % chunk_size == 0 and i > 0:
        # Explains: if i % chunk_size == 0 and i > 0:
            chunks.append(current_chunk)
            # Explains: chunks.append(current_chunk)
            current_chunk = {"text": segment["text"], "start_seconds": segment["start_seconds"], "end_seconds": segment["end_seconds"]}
            # Explains: current_chunk = {"text": segment["text"], "start_seconds": segment["start_seconds"], "end_seconds": segment["end_seconds"]}
        else:
        # Explains: else:
            if current_chunk["start_seconds"] is None:
            # Explains: if current_chunk["start_seconds"] is None:
                current_chunk["start_seconds"] = segment["start_seconds"]
                # Explains: current_chunk["start_seconds"] = segment["start_seconds"]
            current_chunk["text"] += " " + segment["text"]
            # Explains: current_chunk["text"] += " " + segment["text"]
            current_chunk["end_seconds"] = segment["end_seconds"]
            # Explains: current_chunk["end_seconds"] = segment["end_seconds"]

    # Add the last chunk
    if current_chunk["text"]:
    # Explains: if current_chunk["text"]:
        chunks.append(current_chunk)
        # Explains: chunks.append(current_chunk)

    # Create embedding for each chunk
    texts = [chunk["text"] for chunk in chunks]
    # Explains: texts = [chunk["text"] for chunk in chunks]

    # If we have no texts, return empty index
    if not texts:
    # Explains: if not texts:
        dimension = embedder.get_sentence_embedding_dimension()
        # Explains: dimension = embedder.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dimension)
        # Explains: index = faiss.IndexFlatL2(dimension)
        return index, np.array([]).reshape(0, dimension).astype('float32'), [], chunks
        # Explains: return index, np.array([]).reshape(0, dimension).astype('float32'), [], chunks

    embeddings = embedder.encode(texts)
    # Explains: embeddings = embedder.encode(texts)
    embeddings_np = np.array(embeddings).astype('float32')
    # Explains: embeddings_np = np.array(embeddings).astype('float32')

    dimension = embeddings_np.shape[1]
    # Explains: dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    # Explains: index = faiss.IndexFlatL2(dimension)

    # Add embeddings to the index
    index.add(embeddings_np)
    # Explains: index.add(embeddings_np)

    return index, embeddings_np, texts, chunks
    # Explains: return index, embeddings_np, texts, chunks

def search_in_video(query, index, embeddings_np, chunks, entities, objects, faces, k=5):
# Explains: def search_in_video(query, index, embeddings_np, chunks, entities, objects, faces, k=5):
    """Search for query in video and return relevant segments with timestamps"""
    # Explains: """Search for query in video and return relevant segments with timestamps"""
    # Encode the query
    query_embedding = embedder.encode([query])
    # Explains: query_embedding = embedder.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    # Explains: query_embedding = np.array(query_embedding).astype('float32')

    results = []
    # Explains: results = []

    # Search in the index if it has data
    if embeddings_np.size > 0:
    # Explains: if embeddings_np.size > 0:
        D, I = index.search(query_embedding, min(k, embeddings_np.shape[0]))
        # Explains: D, I = index.search(query_embedding, min(k, embeddings_np.shape[0]))

        for idx in I[0]:
        # Explains: for idx in I[0]:
            if idx < len(chunks):
            # Explains: if idx < len(chunks):
                chunk = chunks[idx]
                # Explains: chunk = chunks[idx]
                results.append({
                # Explains: results.append({
                    "type": "transcript",
                    # Explains: "type": "transcript",
                    "start": str(timedelta(seconds=int(chunk["start_seconds"]))),
                    # Explains: "start": str(timedelta(seconds=int(chunk["start_seconds"]))),
                    "end": str(timedelta(seconds=int(chunk["end_seconds"]))),
                    # Explains: "end": str(timedelta(seconds=int(chunk["end_seconds"]))),
                    "text": chunk["text"],
                    # Explains: "text": chunk["text"],
                    "seconds": chunk["start_seconds"]
                    # Explains: "seconds": chunk["start_seconds"]
                })
                # [EXPLANATION NEEDED]

    # Search in named entities
    entity_matches = []
    # Explains: entity_matches = []
    for entity_group in entities:
    # Explains: for entity_group in entities:
        for entity_text, entity_type in entity_group["entities"]:
        # Explains: for entity_text, entity_type in entity_group["entities"]:
            if query.lower() in entity_text.lower():
            # Explains: if query.lower() in entity_text.lower():
                entity_matches.append({
                # Explains: entity_matches.append({
                    "type": "named_entity",
                    # Explains: "type": "named_entity",
                    "start": entity_group["start"],
                    # Explains: "start": entity_group["start"],
                    "end": entity_group["end"],
                    # Explains: "end": entity_group["end"],
                    "entity": entity_text,
                    # Explains: "entity": entity_text,
                    "entity_type": entity_type,
                    # Explains: "entity_type": entity_type,
                    "seconds": entity_group["start_seconds"]
                    # Explains: "seconds": entity_group["start_seconds"]
                })
                # [EXPLANATION NEEDED]

    # Search in object detections
    object_matches = []
    # Explains: object_matches = []
    for obj in objects:
    # Explains: for obj in objects:
        for detected_obj in obj["objects"]:
        # Explains: for detected_obj in obj["objects"]:
            if query.lower() in detected_obj.lower():
            # Explains: if query.lower() in detected_obj.lower():
                object_matches.append({
                # Explains: object_matches.append({
                    "type": "object",
                    # Explains: "type": "object",
                    "time": obj["time"],
                    # Explains: "time": obj["time"],
                    "object": detected_obj,
                    # Explains: "object": detected_obj,
                    "seconds": obj["seconds"]
                    # Explains: "seconds": obj["seconds"]
                })
                # [EXPLANATION NEEDED]

    # Combine all results and sort by timestamp
    all_results = results + entity_matches + object_matches
    # Explains: all_results = results + entity_matches + object_matches
    all_results.sort(key=lambda x: x["seconds"])
    # Explains: all_results.sort(key=lambda x: x["seconds"])

    return all_results
    # Explains: return all_results

def save_output(transcript, entities, objects, faces, index, output_base):
# Explains: def save_output(transcript, entities, objects, faces, index, output_base):
    """Save analysis results to files"""
    # Explains: """Save analysis results to files"""
    txt_lines = []
    # Explains: txt_lines = []
    json_output = {
    # Explains: json_output = {
        "transcript": transcript,
        # Explains: "transcript": transcript,
        "named_entities": entities,
        # Explains: "named_entities": entities,
        "objects": objects,
        # Explains: "objects": objects,
        "faces": faces,
        # Explains: "faces": faces,
        "metadata": {
        # Explains: "metadata": {
            "timestamp": str(timedelta()),
            # Explains: "timestamp": str(timedelta()),
            "version": "1.2",
            # Explains: "version": "1.2",
            "statistics": {
            # Explains: "statistics": {
                "transcript_words": len(transcript),
                # Explains: "transcript_words": len(transcript),
                "named_entities": sum(len(e["entities"]) for e in entities),
                # Explains: "named_entities": sum(len(e["entities"]) for e in entities),
                "object_detection_points": len(objects),
                # Explains: "object_detection_points": len(objects),
                "face_detection_points": len(faces)
                # Explains: "face_detection_points": len(faces)
            }
            # [EXPLANATION NEEDED]
        }
        # [EXPLANATION NEEDED]
    }
    # [EXPLANATION NEEDED]

    # Generate a formatted report
    txt_lines.append("========================================")
    # Explains: txt_lines.append("========================================")
    txt_lines.append("           VIDEO ANALYSIS REPORT        ")
    # Explains: txt_lines.append("           VIDEO ANALYSIS REPORT        ")
    txt_lines.append("========================================")
    # Explains: txt_lines.append("========================================")
    txt_lines.append(f"Generated on: {timedelta()}")
    # Explains: txt_lines.append(f"Generated on: {timedelta()}")
    txt_lines.append("")
    # Explains: txt_lines.append("")

    txt_lines.append("===== TRANSCRIPT =====")
    # Explains: txt_lines.append("===== TRANSCRIPT =====")
    # Group words into sentences for readability in the report
    current_sentence = ""
    # Explains: current_sentence = ""
    current_start = None
    # Explains: current_start = None
    current_end = None
    # Explains: current_end = None

    for word in transcript:
    # Explains: for word in transcript:
        if not current_start:
        # Explains: if not current_start:
            current_start = word["start"]
            # Explains: current_start = word["start"]

        current_sentence += word["text"] + " "
        # Explains: current_sentence += word["text"] + " "
        current_end = word["end"]
        # Explains: current_end = word["end"]

        # Simple heuristic: if word ends with punctuation, consider it end of sentence
        if word["text"].endswith(('.', '?', '!')):
        # Explains: if word["text"].endswith(('.', '?', '!')):
            txt_lines.append(f"[{current_start} - {current_end}] {current_sentence.strip()}")
            # Explains: txt_lines.append(f"[{current_start} - {current_end}] {current_sentence.strip()}")
            current_sentence = ""
            # Explains: current_sentence = ""
            current_start = None
            # Explains: current_start = None

    # Add the last sentence if any
    if current_sentence:
    # Explains: if current_sentence:
        txt_lines.append(f"[{current_start} - {current_end}] {current_sentence.strip()}")
        # Explains: txt_lines.append(f"[{current_start} - {current_end}] {current_sentence.strip()}")

    txt_lines.append("\n===== NAMED ENTITIES =====")
    # Explains: txt_lines.append("\n===== NAMED ENTITIES =====")
    for e in entities:
    # Explains: for e in entities:
        if e["entities"]:  # Only add if entities exist
        # Explains: if e["entities"]:  # Only add if entities exist
            ent_str = ", ".join([f"{t} ({l})" for t, l in e["entities"]])
            # Explains: ent_str = ", ".join([f"{t} ({l})" for t, l in e["entities"]])
            txt_lines.append(f"[{e['start']} - {e['end']}] {ent_str}")
            # Explains: txt_lines.append(f"[{e['start']} - {e['end']}] {ent_str}")

    txt_lines.append("\n===== OBJECT DETECTION =====")
    # Explains: txt_lines.append("\n===== OBJECT DETECTION =====")
    for o in objects:
    # Explains: for o in objects:
        txt_lines.append(f"[{o['time']}] {', '.join(o['objects'])}")
        # Explains: txt_lines.append(f"[{o['time']}] {', '.join(o['objects'])}")

    txt_lines.append("\n===== FACE DETECTION =====")
    # Explains: txt_lines.append("\n===== FACE DETECTION =====")
    for f in faces:
    # Explains: for f in faces:
        txt_lines.append(f"[{f['time']}] Faces Detected: {f['faces_detected']}")
        # Explains: txt_lines.append(f"[{f['time']}] Faces Detected: {f['faces_detected']}")

    # Add summaries
    txt_lines.append("\n===== SUMMARY =====")
    # Explains: txt_lines.append("\n===== SUMMARY =====")
    txt_lines.append(f"Total transcript words: {len(transcript)}")
    # Explains: txt_lines.append(f"Total transcript words: {len(transcript)}")
    txt_lines.append(f"Total named entities detected: {sum(len(e['entities']) for e in entities)}")
    # Explains: txt_lines.append(f"Total named entities detected: {sum(len(e['entities']) for e in entities)}")

    # Get most frequent entities
    all_entities = []
    # Explains: all_entities = []
    for e in entities:
    # Explains: for e in entities:
        all_entities.extend([t for t, _ in e["entities"]])
        # Explains: all_entities.extend([t for t, _ in e["entities"]])

    if all_entities:
    # Explains: if all_entities:
        entity_freq = {}
        # Explains: entity_freq = {}
        for entity in all_entities:
        # Explains: for entity in all_entities:
            entity_freq[entity] = entity_freq.get(entity, 0) + 1
            # Explains: entity_freq[entity] = entity_freq.get(entity, 0) + 1

        # Get top 5 most frequent entities
        top_entities = sorted(entity_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        # Explains: top_entities = sorted(entity_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_entities:
        # Explains: if top_entities:
            txt_lines.append("Most frequent entities:")
            # Explains: txt_lines.append("Most frequent entities:")
            for entity, count in top_entities:
            # Explains: for entity, count in top_entities:
                txt_lines.append(f"  - {entity}: {count} occurrences")
                # Explains: txt_lines.append(f"  - {entity}: {count} occurrences")

    # Get most frequent objects
    all_objects = []
    # Explains: all_objects = []
    for o in objects:
    # Explains: for o in objects:
        all_objects.extend(o["objects"])
        # Explains: all_objects.extend(o["objects"])

    if all_objects:
    # Explains: if all_objects:
        object_freq = {}
        # Explains: object_freq = {}
        for obj in all_objects:
        # Explains: for obj in all_objects:
            object_freq[obj] = object_freq.get(obj, 0) + 1
            # Explains: object_freq[obj] = object_freq.get(obj, 0) + 1

        # Get top 5 most frequent objects
        top_objects = sorted(object_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        # Explains: top_objects = sorted(object_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_objects:
        # Explains: if top_objects:
            txt_lines.append("Most frequent objects:")
            # Explains: txt_lines.append("Most frequent objects:")
            for obj, count in top_objects:
            # Explains: for obj, count in top_objects:
                txt_lines.append(f"  - {obj}: {count} occurrences")
                # Explains: txt_lines.append(f"  - {obj}: {count} occurrences")

    # Search instructions
    txt_lines.append("\n===== SEARCH INSTRUCTIONS =====")
    # Explains: txt_lines.append("\n===== SEARCH INSTRUCTIONS =====")
    txt_lines.append("To search for content in this video:")
    # Explains: txt_lines.append("To search for content in this video:")
    txt_lines.append("1. Use the 'Search Video' tab in the interface")
    # Explains: txt_lines.append("1. Use the 'Search Video' tab in the interface")
    txt_lines.append("2. Enter keywords related to speech, objects, or entities")
    # Explains: txt_lines.append("2. Enter keywords related to speech, objects, or entities")
    txt_lines.append("3. Results will show relevant timestamps where your query appears")
    # Explains: txt_lines.append("3. Results will show relevant timestamps where your query appears")
    txt_lines.append("\n===== REAL-TIME DETECTION =====")
    # Explains: txt_lines.append("\n===== REAL-TIME DETECTION =====")
    txt_lines.append("To see real-time object detection:")
    # Explains: txt_lines.append("To see real-time object detection:")
    txt_lines.append("1. Use the 'Real-time Detection' tab")
    # Explains: txt_lines.append("1. Use the 'Real-time Detection' tab")
    txt_lines.append("2. Upload a video or use your webcam")
    # Explains: txt_lines.append("2. Upload a video or use your webcam")
    txt_lines.append("3. The system will detect objects in real-time")
    # Explains: txt_lines.append("3. The system will detect objects in real-time")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    # Explains: os.makedirs(os.path.dirname(output_base), exist_ok=True)

    txt_path = output_base + ".txt"
    # Explains: txt_path = output_base + ".txt"
    json_path = output_base + ".json"
    # Explains: json_path = output_base + ".json"
    index_path = output_base + "_index.faiss"
    # Explains: index_path = output_base + "_index.faiss"

    try:
    # Explains: try:
        with open(txt_path, "w", encoding="utf-8") as f:
        # Explains: with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines))
            # Explains: f.write("\n".join(txt_lines))
        with open(json_path, "w", encoding="utf-8") as f:
        # Explains: with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
            # Explains: json.dump(json_output, f, indent=2, ensure_ascii=False)

        # Save the FAISS index for later use
        faiss.write_index(index, index_path)
        # Explains: faiss.write_index(index, index_path)
    except Exception as e:
    # Explains: except Exception as e:
        print(f"Error saving output files: {str(e)}")
        # Explains: print(f"Error saving output files: {str(e)}")
        raise
        # Explains: raise

    return txt_path, json_path, index_path
    # Explains: return txt_path, json_path, index_path

# Detect Colab environment
def is_colab():
# Explains: def is_colab():
    try:
    # Explains: try:
        import google.colab
        # Explains: import google.colab
        return True
        # Explains: return True
    except ImportError:
    # Explains: except ImportError:
        return False
        # Explains: return False

# Process video function
def process_video(file):
# Explains: def process_video(file):
    try:
    # Explains: try:
        # For Gradio file upload or direct path string
        if hasattr(file, 'name'):
        # Explains: if hasattr(file, 'name'):
            video_path = file.name
            # Explains: video_path = file.name
        else:
        # Explains: else:
            video_path = file
            # Explains: video_path = file

        # Validate file exists
        if not os.path.exists(video_path):
        # Explains: if not os.path.exists(video_path):
            return f"Error: File not found: {video_path}", None, None, None, None
            # Explains: return f"Error: File not found: {video_path}", None, None, None, None

        # Step 1: Transcribe with word-level timestamps
        transcript = transcribe_audio(video_path)
        # Explains: transcript = transcribe_audio(video_path)

        # Step 2: Named Entities
        entities = extract_named_entities(transcript)
        # Explains: entities = extract_named_entities(transcript)

        # Step 3: Object Detection
        objects = detect_objects(video_path)
        # Explains: objects = detect_objects(video_path)

        # Step 4: Face Detection
        faces = detect_faces(video_path)
        # Explains: faces = detect_faces(video_path)

        # Step 5: Index the data
        index, embeddings, texts, chunks = semantic_indexing(transcript)
        # Explains: index, embeddings, texts, chunks = semantic_indexing(transcript)

        # Step 6: Save results with appropriate path
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        # Explains: base_name = os.path.splitext(os.path.basename(video_path))[0]

        # Use a directory that's accessible in Colab or normal environment
        if is_colab():
        # Explains: if is_colab():
            output_dir = "/content/video_analysis_results"
            # Explains: output_dir = "/content/video_analysis_results"
        else:
        # Explains: else:
            output_dir = os.path.join(os.getcwd(), "video_analysis_results")
            # Explains: output_dir = os.path.join(os.getcwd(), "video_analysis_results")

        os.makedirs(output_dir, exist_ok=True)
        # Explains: os.makedirs(output_dir, exist_ok=True)
        output_base = os.path.join(output_dir, base_name + "_indexed")
        # Explains: output_base = os.path.join(output_dir, base_name + "_indexed")

        # Save content
        txt_path, json_path, index_path = save_output(transcript, entities, objects, faces, index, output_base)
        # Explains: txt_path, json_path, index_path = save_output(transcript, entities, objects, faces, index, output_base)

        # Store metadata for search functionality
        metadata = {
        # Explains: metadata = {
            "transcript": transcript,
            # Explains: "transcript": transcript,
            "entities": entities,
            # Explains: "entities": entities,
            "objects": objects,
            # Explains: "objects": objects,
            "faces": faces,
            # Explains: "faces": faces,
            "index": index,
            # Explains: "index": index,
            "embeddings": embeddings,
            # Explains: "embeddings": embeddings,
            "texts": texts,
            # Explains: "texts": texts,
            "chunks": chunks,
            # Explains: "chunks": chunks,
            "video_path": video_path
            # Explains: "video_path": video_path
        }
        # [EXPLANATION NEEDED]

        # Create a summary for display
        summary = f"Analysis complete for {base_name}:\n"
        # Explains: summary = f"Analysis complete for {base_name}:\n"
        summary += f"- Found {len(transcript)} transcript words\n"
        # Explains: summary += f"- Found {len(transcript)} transcript words\n"
        summary += f"- Found {sum(len(e['entities']) for e in entities)} named entities\n"
        # Explains: summary += f"- Found {sum(len(e['entities']) for e in entities)} named entities\n"
        summary += f"- Detected objects at {len(objects)} timestamps\n"
        # Explains: summary += f"- Detected objects at {len(objects)} timestamps\n"
        summary += f"- Detected faces at {len(faces)} timestamps\n"
        # Explains: summary += f"- Detected faces at {len(faces)} timestamps\n"
        summary += f"\nFiles saved to: {output_dir}\n"
        # Explains: summary += f"\nFiles saved to: {output_dir}\n"
        summary += f"\nSearch functionality is now available. Use the 'Search Video' tab to find content in the video."
        # Explains: summary += f"\nSearch functionality is now available. Use the 'Search Video' tab to find content in the video."

        return summary, txt_path, json_path, index_path, metadata
        # Explains: return summary, txt_path, json_path, index_path, metadata

    except Exception as e:
    # Explains: except Exception as e:
        import traceback
        # Explains: import traceback
        error_msg = f"Error during processing: {str(e)}\n{traceback.format_exc()}"
        # Explains: error_msg = f"Error during processing: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        # Explains: print(error_msg)
        return error_msg, None, None, None, None
        # Explains: return error_msg, None, None, None, None

def search_video(query, metadata):
# Explains: def search_video(query, metadata):
    """Search for query in processed video"""
    # Explains: """Search for query in processed video"""
    if not query or not metadata:
    # Explains: if not query or not metadata:
        return "Please enter a search query and process a video first.", None
        # Explains: return "Please enter a search query and process a video first.", None

    results = search_in_video(
    # Explains: results = search_in_video(
        query,
        # Explains: query,
        metadata["index"],
        # Explains: metadata["index"],
        metadata["embeddings"],
        # Explains: metadata["embeddings"],
        metadata["chunks"],
        # Explains: metadata["chunks"],
        metadata["entities"],
        # Explains: metadata["entities"],
        metadata["objects"],
        # Explains: metadata["objects"],
        metadata["faces"]
        # Explains: metadata["faces"]
    )
    # [EXPLANATION NEEDED]

    if not results:
    # Explains: if not results:
        return f"No results found for '{query}'.", None
        # Explains: return f"No results found for '{query}'.", None

    result_text = f"Found {len(results)} results for '{query}':\n\n"
    # Explains: result_text = f"Found {len(results)} results for '{query}':\n\n"

    # Keep track of the first result timestamp for video playback
    first_result_seconds = None
    # Explains: first_result_seconds = None

    for i, res in enumerate(results):
    # Explains: for i, res in enumerate(results):
        if first_result_seconds is None:
        # Explains: if first_result_seconds is None:
            first_result_seconds = res["seconds"]
            # Explains: first_result_seconds = res["seconds"]

        if res["type"] == "transcript":
        # Explains: if res["type"] == "transcript":
            result_text += f"{i+1}. [TRANSCRIPT] at {res['start']} - {res['end']}\n"
            # Explains: result_text += f"{i+1}. [TRANSCRIPT] at {res['start']} - {res['end']}\n"
            result_text += f"   \"{res['text']}\"\n\n"
            # Explains: result_text += f"   \"{res['text']}\"\n\n"
        elif res["type"] == "named_entity":
        # Explains: elif res["type"] == "named_entity":
            result_text += f"{i+1}. [ENTITY: {res['entity_type']}] at {res['start']}\n"
            # Explains: result_text += f"{i+1}. [ENTITY: {res['entity_type']}] at {res['start']}\n"
            result_text += f"   Found entity \"{res['entity']}\"\n\n"
            # Explains: result_text += f"   Found entity \"{res['entity']}\"\n\n"
        elif res["type"] == "object":
        # Explains: elif res["type"] == "object":
            result_text += f"{i+1}. [OBJECT] at {res['time']}\n"
            # Explains: result_text += f"{i+1}. [OBJECT] at {res['time']}\n"
            result_text += f"   Detected object: {res['object']}\n\n"
            # Explains: result_text += f"   Detected object: {res['object']}\n\n"

    return result_text, first_result_seconds
    # Explains: return result_text, first_result_seconds

# Helper function for Colab direct download
def direct_colab_download(file_path):
# Explains: def direct_colab_download(file_path):
    """Force a direct download in Colab bypassing Gradio interface"""
    # Explains: """Force a direct download in Colab bypassing Gradio interface"""
    try:
    # Explains: try:
        from google.colab import files
        # Explains: from google.colab import files
        if file_path and os.path.exists(file_path):
        # Explains: if file_path and os.path.exists(file_path):
            files.download(file_path)
            # Explains: files.download(file_path)
            return f"Downloading {os.path.basename(file_path)}..."
            # Explains: return f"Downloading {os.path.basename(file_path)}..."
        else:
        # Explains: else:
            return f"Error: File not found at {file_path}"
            # Explains: return f"Error: File not found at {file_path}"
    except ImportError:
    # Explains: except ImportError:
        return "This function only works in Google Colab"
        # Explains: return "This function only works in Google Colab"

# Real-time object detection
def realtime_object_detection(source):
# Explains: def realtime_object_detection(source):
    """Perform real-time object detection with annotated video output"""
    # Explains: """Perform real-time object detection with annotated video output"""
    try:
    # Explains: try:
        from ultralytics import YOLO
        # Explains: from ultralytics import YOLO
        import cv2
        # Explains: import cv2
        import tempfile
        # Explains: import tempfile
        import os
        # Explains: import os
        from PIL import Image
        # Explains: from PIL import Image

        # Load the YOLOv8 model
        model = YOLO("yolov8n.pt")
        # Explains: model = YOLO("yolov8n.pt")

        # Create temporary output file
        temp_dir = tempfile.mkdtemp()
        # Explains: temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "annotated_video.mp4")
        # Explains: output_path = os.path.join(temp_dir, "annotated_video.mp4")

        # Initialize video writer
        cap = cv2.VideoCapture(source)
        # Explains: cap = cv2.VideoCapture(source)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # Explains: frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Explains: frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Explains: fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Explains: fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        # Explains: out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
        # Explains: while cap.isOpened():
            ret, frame = cap.read()
            # Explains: ret, frame = cap.read()
            if not ret:
            # Explains: if not ret:
                break
                # Explains: break

            # Perform detection
            results = model.predict(frame, verbose=False)
            # Explains: results = model.predict(frame, verbose=False)

            # Draw annotations on the frame
            for result in results:
            # Explains: for result in results:
                boxes = result.boxes
                # Explains: boxes = result.boxes
                if boxes is not None:
                # Explains: if boxes is not None:
                    for box in boxes:
                    # Explains: for box in boxes:
                        # Get box coordinates and info
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        # Explains: x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        confidence = box.conf[0].item()
                        # Explains: confidence = box.conf[0].item()
                        class_id = int(box.cls[0].item())
                        # Explains: class_id = int(box.cls[0].item())
                        label = model.names[class_id]
                        # Explains: label = model.names[class_id]

                        # Draw green bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        # Explains: cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Create label text
                        text = f"{label} {confidence:.2f}"
                        # Explains: text = f"{label} {confidence:.2f}"

                        # Calculate text position
                        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        # Explains: (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

                        # Draw text background
                        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1 - 10), (0, 255, 0), -1)
                        # Explains: cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1 - 10), (0, 255, 0), -1)

                        # Draw text
                        cv2.putText(frame, text, (x1, y1 - 10),
                        # Explains: cv2.putText(frame, text, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                                   # Explains: cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Write frame to output video
            out.write(frame)
            # Explains: out.write(frame)

        cap.release()
        # Explains: cap.release()
        out.release()
        # Explains: out.release()

        return output_path
        # Explains: return output_path

    except Exception as e:
    # Explains: except Exception as e:
        import traceback
        # Explains: import traceback
        error_msg = f"Error during real-time detection: {str(e)}\n{traceback.format_exc()}"
        # Explains: error_msg = f"Error during real-time detection: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        # Explains: print(error_msg)
        return error_msg
        # Explains: return error_msg

    except Exception as e:
    # Explains: except Exception as e:
        import traceback
        # Explains: import traceback
        error_msg = f"Error during real-time detection: {str(e)}\n{traceback.format_exc()}"
        # Explains: error_msg = f"Error during real-time detection: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        # Explains: print(error_msg)
        return error_msg
        # Explains: return error_msg

# Define the Gradio interface with search functionality and real-time detection
def create_interface():
# Explains: def create_interface():
    with gr.Blocks(title="🎥 Advanced Video Indexer") as demo:
    # Explains: with gr.Blocks(title="🎥 Advanced Video Indexer") as demo:
        # Store metadata between tabs
        metadata_store = gr.State(None)
        # Explains: metadata_store = gr.State(None)
        processed_video_path = gr.State(None)
        # Explains: processed_video_path = gr.State(None)

        gr.Markdown("# 🎥 Advanced Video Indexer with Search and Real-time Detection")
        # Explains: gr.Markdown("# 🎥 Advanced Video Indexer with Search and Real-time Detection")
        gr.Markdown("Upload a video to extract word-level captions, objects, named entities, and faces. Then search for specific content in the video or run real-time object detection.")
        # Explains: gr.Markdown("Upload a video to extract word-level captions, objects, named entities, and faces. Then search for specific content in the video or run real-time object detection.")

        with gr.Tabs():
        # Explains: with gr.Tabs():
            with gr.TabItem("Process Video"):
            # Explains: with gr.TabItem("Process Video"):
                with gr.Row():
                # Explains: with gr.Row():
                    video_input = gr.Video(label="Upload your MP4 video")
                    # Explains: video_input = gr.Video(label="Upload your MP4 video")

                with gr.Row():
                # Explains: with gr.Row():
                    analyze_btn = gr.Button("Analyze Video", variant="primary")
                    # Explains: analyze_btn = gr.Button("Analyze Video", variant="primary")

                with gr.Row():
                # Explains: with gr.Row():
                    output_text = gr.Textbox(label="Analysis Summary", lines=10)
                    # Explains: output_text = gr.Textbox(label="Analysis Summary", lines=10)

                # Download section
                with gr.Row():
                # Explains: with gr.Row():
                    with gr.Column():
                    # Explains: with gr.Column():
                        txt_file = gr.File(label="TXT Report")
                        # Explains: txt_file = gr.File(label="TXT Report")
                    with gr.Column():
                    # Explains: with gr.Column():
                        json_file = gr.File(label="JSON Data")
                        # Explains: json_file = gr.File(label="JSON Data")
                    with gr.Column():
                    # Explains: with gr.Column():
                        index_file = gr.File(label="FAISS Index")
                        # Explains: index_file = gr.File(label="FAISS Index")

            with gr.TabItem("Search Video"):
            # Explains: with gr.TabItem("Search Video"):
                with gr.Row():
                # Explains: with gr.Row():
                    search_query = gr.Textbox(label="Search Query", placeholder="Enter keywords to search in the video...")
                    # Explains: search_query = gr.Textbox(label="Search Query", placeholder="Enter keywords to search in the video...")
                    search_btn = gr.Button("Search", variant="primary")
                    # Explains: search_btn = gr.Button("Search", variant="primary")

                with gr.Row():
                # Explains: with gr.Row():
                    search_results = gr.Textbox(label="Search Results", lines=15)
                    # Explains: search_results = gr.Textbox(label="Search Results", lines=15)

                with gr.Row():
                # Explains: with gr.Row():
                    video_player = gr.Video(label="Video Player", interactive=False)
                    # Explains: video_player = gr.Video(label="Video Player", interactive=False)

                with gr.Row():
                # Explains: with gr.Row():
                    timestamp_input = gr.Number(label="Jump to Time (seconds)", value=0, precision=0)
                    # Explains: timestamp_input = gr.Number(label="Jump to Time (seconds)", value=0, precision=0)
                    go_to_timestamp_btn = gr.Button("Jump to Time")
                    # Explains: go_to_timestamp_btn = gr.Button("Jump to Time")

            with gr.TabItem("Real-time Detection"):
            # Explains: with gr.TabItem("Real-time Detection"):
                gr.Markdown("## Real-time Object Detection")
                # Explains: gr.Markdown("## Real-time Object Detection")
                gr.Markdown("Upload a video or use your webcam to detect objects in real-time with visual annotations.")
                # Explains: gr.Markdown("Upload a video or use your webcam to detect objects in real-time with visual annotations.")

                with gr.Row():
                # Explains: with gr.Row():
                    realtime_video_input = gr.Video(label="Video Source", interactive=True)
                    # Explains: realtime_video_input = gr.Video(label="Video Source", interactive=True)

                with gr.Row():
                # Explains: with gr.Row():
                    realtime_btn = gr.Button("Start Real-time Detection", variant="primary")
                    # Explains: realtime_btn = gr.Button("Start Real-time Detection", variant="primary")

                with gr.Row():
                # Explains: with gr.Row():
                    # Changed from Textbox to Video output
                    realtime_output = gr.Video(label="Annotated Output", autoplay=True)
                    # Explains: realtime_output = gr.Video(label="Annotated Output", autoplay=True)

        # Check if we're in Colab to add direct download buttons
        in_colab = is_colab()
        # Explains: in_colab = is_colab()
        if in_colab:
        # Explains: if in_colab:
            # Store file paths as hidden variables
            txt_path_var = gr.Textbox(visible=False)
            # Explains: txt_path_var = gr.Textbox(visible=False)
            json_path_var = gr.Textbox(visible=False)
            # Explains: json_path_var = gr.Textbox(visible=False)
            index_path_var = gr.Textbox(visible=False)
            # Explains: index_path_var = gr.Textbox(visible=False)

            with gr.Row() as colab_buttons:
            # Explains: with gr.Row() as colab_buttons:
                with gr.Column():
                # Explains: with gr.Column():
                    txt_colab_btn = gr.Button("Force Download TXT Report (Colab)", variant="secondary")
                    # Explains: txt_colab_btn = gr.Button("Force Download TXT Report (Colab)", variant="secondary")
                with gr.Column():
                # Explains: with gr.Column():
                    json_colab_btn = gr.Button("Force Download JSON Data (Colab)", variant="secondary")
                    # Explains: json_colab_btn = gr.Button("Force Download JSON Data (Colab)", variant="secondary")
                with gr.Column():
                # Explains: with gr.Column():
                    index_colab_btn = gr.Button("Force Download FAISS Index (Colab)", variant="secondary")
                    # Explains: index_colab_btn = gr.Button("Force Download FAISS Index (Colab)", variant="secondary")

            # Add direct download functionality for Colab
            txt_colab_btn.click(
            # Explains: txt_colab_btn.click(
                fn=direct_colab_download,
                # Explains: fn=direct_colab_download,
                inputs=txt_path_var,
                # Explains: inputs=txt_path_var,
                outputs=gr.Textbox(label="Download Status")
                # Explains: outputs=gr.Textbox(label="Download Status")
            )
            # [EXPLANATION NEEDED]

            json_colab_btn.click(
            # Explains: json_colab_btn.click(
                fn=direct_colab_download,
                # Explains: fn=direct_colab_download,
                inputs=json_path_var,
                # Explains: inputs=json_path_var,
                outputs=gr.Textbox(label="Download Status")
                # Explains: outputs=gr.Textbox(label="Download Status")
            )
            # [EXPLANATION NEEDED]

            index_colab_btn.click(
            # Explains: index_colab_btn.click(
                fn=direct_colab_download,
                # Explains: fn=direct_colab_download,
                inputs=index_path_var,
                # Explains: inputs=index_path_var,
                outputs=gr.Textbox(label="Download Status")
                # Explains: outputs=gr.Textbox(label="Download Status")
            )
            # [EXPLANATION NEEDED]

            # Define process function with Colab outputs
            def process_and_show_all(file):
            # Explains: def process_and_show_all(file):
                summary, txt_path, json_path, index_path, meta = process_video(file)
                # Explains: summary, txt_path, json_path, index_path, meta = process_video(file)
                return summary, txt_path, json_path, index_path, txt_path, json_path, index_path, meta, file.name if hasattr(file, 'name') else file
                # Explains: return summary, txt_path, json_path, index_path, txt_path, json_path, index_path, meta, file.name if hasattr(file, 'name') else file

            # Connect analyze button to process function
            analyze_btn.click(
            # Explains: analyze_btn.click(
                fn=process_and_show_all,
                # Explains: fn=process_and_show_all,
                inputs=video_input,
                # Explains: inputs=video_input,
                outputs=[output_text, txt_file, json_file, index_file,
                # Explains: outputs=[output_text, txt_file, json_file, index_file,
                         txt_path_var, json_path_var, index_path_var,
                         # Explains: txt_path_var, json_path_var, index_path_var,
                         metadata_store, processed_video_path]
                         # Explains: metadata_store, processed_video_path]
            )
            # [EXPLANATION NEEDED]

        else:
        # Explains: else:
            # Standard processing for non-Colab environments
            def process_and_show(file):
            # Explains: def process_and_show(file):
                summary, txt_path, json_path, index_path, meta = process_video(file)
                # Explains: summary, txt_path, json_path, index_path, meta = process_video(file)
                return summary, txt_path, json_path, index_path, meta, file.name if hasattr(file, 'name') else file
                # Explains: return summary, txt_path, json_path, index_path, meta, file.name if hasattr(file, 'name') else file

            # Connect analyze button to process function
            analyze_btn.click(
            # Explains: analyze_btn.click(
                fn=process_and_show,
                # Explains: fn=process_and_show,
                inputs=video_input,
                # Explains: inputs=video_input,
                outputs=[output_text, txt_file, json_file, index_file,
                # Explains: outputs=[output_text, txt_file, json_file, index_file,
                         metadata_store, processed_video_path]
                         # Explains: metadata_store, processed_video_path]
            )
            # [EXPLANATION NEEDED]

        # Connect search functionality
        search_btn.click(
        # Explains: search_btn.click(
            fn=search_video,
            # Explains: fn=search_video,
            inputs=[search_query, metadata_store],
            # Explains: inputs=[search_query, metadata_store],
            outputs=[search_results, timestamp_input]
            # Explains: outputs=[search_results, timestamp_input]
        )
        # [EXPLANATION NEEDED]

        # Function to jump to timestamp
        def jump_to_timestamp(video_path, timestamp):
        # Explains: def jump_to_timestamp(video_path, timestamp):
          if not video_path:
          # Explains: if not video_path:
            return None
            # Explains: return None

          try:
          # Explains: try:
            timestamp = float(timestamp) if timestamp is not None else 0
            # Explains: timestamp = float(timestamp) if timestamp is not None else 0
            # Return just the video path (Gradio will handle seeking)
            return video_path
            # Explains: return video_path
          except:
          # Explains: except:
            return video_path
            # Explains: return video_path

        # Connect timestamp button and input
        go_to_timestamp_btn.click(
        # Explains: go_to_timestamp_btn.click(
    fn=lambda video_path, timestamp: video_path,  # Just return the video path
    # Explains: fn=lambda video_path, timestamp: video_path,  # Just return the video path
    inputs=[processed_video_path, timestamp_input],
    # Explains: inputs=[processed_video_path, timestamp_input],
    outputs=video_player
    # Explains: outputs=video_player
)
# [EXPLANATION NEEDED]

        # Connect real-time detection
        realtime_btn.click(
        # Explains: realtime_btn.click(
            fn=realtime_object_detection,
            # Explains: fn=realtime_object_detection,
            inputs=realtime_video_input,
            # Explains: inputs=realtime_video_input,
            outputs=realtime_output
            # Explains: outputs=realtime_output
        )
        # [EXPLANATION NEEDED]

        # Display special Colab instructions if needed
        if in_colab:
            gr.Markdown("""
            ## Special Instructions for Colab

            If the standard download links don't work:
            1. Use the "Force Download" buttons above
            2. Or access files directly in `/content/video_analysis_results/` folder

            For real-time detection in Colab:
            1. You may need to grant camera permissions
            2. For video files, upload them first before running detection
            """)

    return demo

    return demo
    # Explains: return demo

# Launch the interface
# Launch the interface
if __name__ == "__main__":
# Explains: if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    # Explains: demo = create_interface()

    # Add special settings for Colab
    if is_colab():
    # Explains: if is_colab():
        demo.launch(debug=True, share=True)
        # Explains: demo.launch(debug=True, share=True)
    else:
    # Explains: else:
        demo.launch()
        # Explains: demo.launch()
