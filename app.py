# ==========================================================
# Lecture Video -> ASR -> Vision -> Align -> PDF notes
# Run the web app :  python app.py            (add --share for a public link)
# Run on one video:  python app.py --video lecture.mp4
# ==========================================================
import os, re, glob, json, time, subprocess, argparse
import cv2, torch, whisper, clip
import numpy as np
import pytesseract
import gradio as gr

from xml.sax.saxutils import escape
from PIL import Image as PILImage
from sklearn.feature_extraction.text import CountVectorizer

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

from transformers import BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

# ---------------- PATHS (override with environment variables) ----------------
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "model_final.pth")   # fine-tuned Detectron2 weights
WORK_DIR = os.environ.get("WORK_DIR", "outputs")                    # PDFs, JSON and crops go here

CROP_DIR = os.path.join(WORK_DIR, "crops")
OUTPUT_PDF = os.path.join(WORK_DIR, "final_notes.pdf")
os.makedirs(CROP_DIR, exist_ok=True)

# ---------------- PARAMETERS (as in the thesis) ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["text", "title", "list", "table", "figure"]
COCO_CFG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

CHUNK_SIZE = 800          # characters per transcript chunk
MIN_CHUNK_WORDS = 30      # skip very short chunks
LAYOUT_CONF = 0.70        # figure / table detection confidence
HUMAN_CONF = 0.70         # person detection confidence
HUMAN_IOU = 0.10          # drop visuals overlapping a person above this IoU
DEDUP_THRESHOLD = 0.90    # CLIP cosine similarity for duplicate removal
TIME_WINDOW = 30          # seconds, alignment window
ALPHA, BETA = 0.7, 0.3    # temporal weight, semantic weight

print("Using device:", DEVICE)

# ==========================================================
# LOAD MODELS
# ==========================================================
print("Loading models...")

asr_model = whisper.load_model("base", device=DEVICE)

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer = BartForConditionalGeneration.from_pretrained(
    "facebook/bart-large-cnn").to(DEVICE)

text_embedder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)

# layout detector: Faster R-CNN fine-tuned on PubLayNet (5 classes)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(COCO_CFG))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.MODEL.WEIGHTS = WEIGHTS_PATH
cfg.MODEL.DEVICE = DEVICE
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = LAYOUT_CONF
layout_predictor = DefaultPredictor(cfg)

# human detector: COCO-pretrained Faster R-CNN (class 0 = person)
hcfg = get_cfg()
hcfg.merge_from_file(model_zoo.get_config_file(COCO_CFG))
hcfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(COCO_CFG)
hcfg.MODEL.DEVICE = DEVICE
hcfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = HUMAN_CONF
human_predictor = DefaultPredictor(hcfg)

print("All models loaded")

# ==========================================================
# MODULE 1: AUDIO -> SUMMARY + TIMESTAMPS
# ==========================================================
def extract_audio(video):
    audio = os.path.join(WORK_DIR, "audio.wav")
    subprocess.run(["ffmpeg", "-y", "-i", video, "-ac", "1", "-ar", "16000", audio],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio


def clean_summary(text):
    bad = ["in this lecture", "we will discuss",
           "today we are going to", "now we will see"]
    for p in bad:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
    return text.strip()


def module1(video):
    print("Module 1 (Audio)...")
    audio = extract_audio(video)
    segments = asr_model.transcribe(audio, fp16=(DEVICE == "cuda"))["segments"]
    if not segments:
        return []

    transcript = " ".join(s["text"] for s in segments)
    chunks = [transcript[i:i + CHUNK_SIZE] for i in range(0, len(transcript), CHUNK_SIZE)]

    summaries = []
    for chunk in chunks:
        if len(chunk.split()) < MIN_CHUNK_WORDS:
            continue
        inputs = tokenizer(chunk, return_tensors="pt",
                           truncation=True, max_length=1024).to(DEVICE)
        ids = summarizer.generate(inputs["input_ids"],
                                  max_length=120, min_length=40, num_beams=4)
        text = tokenizer.decode(ids[0], skip_special_tokens=True)
        summaries.append(clean_summary(text))

    seg_emb = text_embedder.encode([s["text"] for s in segments], convert_to_tensor=True)

    aligned = []
    for sent in summaries:
        emb = text_embedder.encode(sent, convert_to_tensor=True)
        idx = util.cos_sim(emb, seg_emb)[0].argmax().item()
        aligned.append({
            "start": float(segments[idx]["start"]),
            "end": float(segments[idx]["end"]),
            "summary": sent,
            "transcript": segments[idx]["text"].strip()
        })

    with open(os.path.join(WORK_DIR, "module1_output.json"), "w") as f:
        json.dump(aligned, f, indent=4)
    return aligned


# ==========================================================
# MODULE 2: VISION -> FIGURES / TABLES
# ==========================================================
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def get_clip_embedding(img_bgr):
    img = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    img = clip_preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = clip_model.encode_image(img)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]


def module2(video):
    print("Module 2 (Vision)...")
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(fps))                       # 1 frame per second

    visuals, seen, frame_id = [], [], 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % step == 0:
            t = frame_id / fps
            H, W = frame.shape[:2]

            # 1) people in the frame
            h_inst = human_predictor(frame)["instances"].to("cpu")
            human_boxes = [b for b, c in zip(h_inst.pred_boxes.tensor.numpy(),
                                             h_inst.pred_classes.numpy()) if int(c) == 0]

            # 2) figures and tables
            inst = layout_predictor(frame)["instances"].to("cpu")
            for box, cls in zip(inst.pred_boxes.tensor.numpy(), inst.pred_classes):
                label = CLASS_NAMES[int(cls)]
                if label not in ("table", "figure"):
                    continue

                # drop visuals that overlap the instructor
                if any(iou(box, hb) > HUMAN_IOU for hb in human_boxes):
                    continue

                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # 3) remove duplicates with CLIP
                emb = get_clip_embedding(crop)
                if any(float(np.dot(emb, e)) > DEDUP_THRESHOLD for e in seen):
                    continue
                seen.append(emb)

                path = os.path.join(CROP_DIR, f"{label}_{int(t)}.jpg")
                cv2.imwrite(path, crop)

                # 4) OCR + text embedding (None when no text)
                text = pytesseract.image_to_string(
                    cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)).strip()
                text_emb = (text_embedder.encode(text, convert_to_tensor=True)
                            if text else None)

                visuals.append({"timestamp": float(t), "image": path,
                                "text": text, "text_emb": text_emb})
        frame_id += 1

    cap.release()

    with open(os.path.join(WORK_DIR, "module2_output.json"), "w") as f:
        json.dump([{k: v for k, v in x.items() if k != "text_emb"} for x in visuals],
                  f, indent=4)
    return visuals


# ==========================================================
# MODULE 3: ALIGN TEXT AND VISUALS
# ==========================================================
def align(m1, m2):
    print("Module 3 (Align)...")
    aligned = []

    for seg in m1:
        mid = (seg["start"] + seg["end"]) / 2
        s_emb = text_embedder.encode(seg["summary"], convert_to_tensor=True)
        best_img, best_score = None, -1

        for v in m2:
            d = abs(mid - v["timestamp"])
            if d > TIME_WINDOW:
                continue
            sim = 0.0
            if v["text_emb"] is not None:
                sim = max(0.0, util.cos_sim(s_emb, v["text_emb"])[0][0].item())
            score = ALPHA * (1 - d / TIME_WINDOW) + BETA * sim
            if score > best_score:
                best_score, best_img = score, v["image"]

        aligned.append({"summary": seg["summary"], "image": best_img})

    with open(os.path.join(WORK_DIR, "module3_output.json"), "w") as f:
        json.dump(aligned, f, indent=4)
    return aligned


# ==========================================================
# TITLE GENERATION (keyword extraction + MMR)
# ==========================================================
def generate_title(m1, top_n=3, diversity=0.4):
    try:
        summaries = [x["summary"] for x in m1]
        text = " ".join(summaries)

        vec = CountVectorizer(
            ngram_range=(1, 2), stop_words="english",
            token_pattern=r"(?u)\b[A-Za-z][A-Za-z\-]{2,}\b").fit([text])
        cands = list(vec.get_feature_names_out())
        if len(cands) < top_n:
            raise ValueError("not enough candidates")

        doc_emb = text_embedder.encode(summaries, convert_to_tensor=True).mean(dim=0, keepdim=True)
        c_emb = text_embedder.encode(cands, convert_to_tensor=True)
        rel = util.cos_sim(doc_emb, c_emb)[0]
        cc = util.cos_sim(c_emb, c_emb)

        selected = [int(rel.argmax())]
        while len(selected) < top_n:
            best, best_score = None, -1e9
            for i in range(len(cands)):
                if i in selected:
                    continue
                words = set(cands[i].split())
                if any(words & set(cands[j].split()) for j in selected):
                    continue
                red = max(float(cc[i][j]) for j in selected)
                score = (1 - diversity) * float(rel[i]) - diversity * red
                if score > best_score:
                    best, best_score = i, score
            if best is None:
                break
            selected.append(best)

        kws = [cands[i].title() for i in selected]
        if len(kws) == 1:
            return f"Notes on {kws[0]}"
        return "Notes on " + ", ".join(kws[:-1]) + " and " + kws[-1]

    except Exception:
        # fallback: BART title
        text = " ".join(x["summary"] for x in m1[:5])
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=512).to(DEVICE)
        ids = summarizer.generate(inputs["input_ids"], max_length=20, min_length=5)
        return tokenizer.decode(ids[0], skip_special_tokens=True).title()


# ==========================================================
# PDF GENERATION
# ==========================================================
def generate_pdf(notes, title):
    print("Generating PDF...")
    doc = SimpleDocTemplate(OUTPUT_PDF, pagesize=A4)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Heading1"],
                                 alignment=1, spaceAfter=20)

    elements = [Paragraph(escape(title), title_style), Spacer(1, 0.3 * inch)]

    for n in notes:
        for s in n["summary"].split("."):
            if s.strip():
                elements.append(Paragraph(escape(s.strip()), styles["Normal"]))
                elements.append(Spacer(1, 0.12 * inch))
        elements.append(Spacer(1, 0.2 * inch))

        if n["image"] and os.path.exists(n["image"]):
            iw, ih = PILImage.open(n["image"]).size
            scale = min(4.5 * inch / iw, 3 * inch / ih)     # fit in 4.5 x 3 in, keep aspect
            elements.append(RLImage(n["image"], width=iw * scale, height=ih * scale))
            elements.append(Spacer(1, 0.3 * inch))

    doc.build(elements)
    return OUTPUT_PDF


# ==========================================================
# FULL PIPELINE
# ==========================================================
def run_pipeline(video_path, progress=None):
    def tick(p, msg):
        print(msg)
        if progress:
            progress(p, desc=msg)

    for f in glob.glob(os.path.join(CROP_DIR, "*")):      # clear crops of the previous run
        os.remove(f)

    tick(0.10, "Step 1/4: Transcribing and summarizing audio")
    m1 = module1(video_path)
    if not m1:
        raise ValueError("No usable speech found in the video.")

    tick(0.40, "Step 2/4: Detecting figures and tables")
    m2 = module2(video_path)

    tick(0.80, "Step 3/4: Aligning text with visuals")
    m3 = align(m1, m2)

    tick(0.90, "Step 4/4: Building PDF")
    title = generate_title(m1)
    return generate_pdf(m3, title), title

# ==========================================================
# GRADIO FRONTEND
# ==========================================================
def make_notes(video_path, progress=gr.Progress()):
    if video_path is None:
        raise gr.Error("Please upload a video.")
    try:
        pdf, _ = run_pipeline(video_path, progress)
    except ValueError as e:
        raise gr.Error(str(e))
    return pdf


def build_demo():
    return gr.Interface(
        fn=make_notes,
        inputs=gr.Video(label="Upload lecture video"),
        outputs=gr.File(label="Download PDF notes"),
        title="Lecture Video to PDF Notes",
        description="Upload a lecture video. The system transcribes the audio, summarizes it, "
                    "detects figures and tables, and builds a PDF. Please wait a few minutes.",
    )


# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lecture video -> PDF notes")
    parser.add_argument("--video", help="process a video from the command line (no web UI)")
    parser.add_argument("--share", action="store_true", help="create a temporary public Gradio link")
    args = parser.parse_args()

    if args.video:
        start = time.time()
        pdf, title = run_pipeline(args.video)
        print("\nPDF:", pdf)
        print("Title:", title)
        print("Time:", round(time.time() - start, 1), "sec")
    else:
        build_demo().queue().launch(share=args.share)
