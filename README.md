# Lecture Video to PDF Notes

A multimodal pipeline that takes a lecture video and produces a structured PDF with summarized text and the relevant figures and tables placed next to it. It also includes a web frontend built with Gradio.

## How it works

```
Video
 ├── Audio stream ──► Whisper (ASR) ──► 800-char chunks ──► BART summaries ──► timestamps (MiniLM matching)
 │
 └── Frames (1 fps) ─► Detectron2 (figure / table) ─► remove instructor overlap (person detector, IoU)
                      ─► CLIP duplicate removal ─► Tesseract OCR + MiniLM embedding
                                   │
 Summaries + Visuals ──► Alignment (time proximity + text similarity) ──► PDF
```

| Stage | Model / method |
|---|---|
| Speech recognition | Whisper `base` |
| Summarization | `facebook/bart-large-cnn` (chunk size 800 chars, 40 to 120 tokens) |
| Layout detection | Faster R-CNN R50-FPN (Detectron2) fine-tuned on PubLayNet (5 classes) |
| Instructor filtering | COCO-pretrained Faster R-CNN, person class, IoU > 0.10 removed |
| Duplicate removal | CLIP ViT-B/32 embeddings, cosine > 0.90 |
| OCR | Tesseract |
| Alignment | score = 0.7 x (1 - d/30) + 0.3 x text similarity, 30 s window |
| Title | Keyword extraction + MMR using MiniLM embeddings |
| PDF | ReportLab |
| Frontend | Gradio |

## Project structure

```
├── app.py            # full pipeline + Gradio web app (also usable from the command line)
├── notebook.ipynb    # Kaggle notebook version
├── requirements.txt
└── README.md
```

## Model weights

The fine-tuned Detectron2 weights (`model_final.pth`) are too large for GitHub.
Download them from: **ADD_YOUR_HUGGING_FACE_OR_DRIVE_LINK_HERE**

## Run on Kaggle (easiest)

1. Upload `notebook.ipynb` to Kaggle.
2. Settings: **Accelerator = GPU**, **Internet = ON**.
3. Add your video and the weights as Kaggle inputs, and update `VIDEO_PATH` and `WEIGHTS_PATH`.
4. Run the cells from top to bottom. The last cell prints a temporary `gradio.live` link.

## Run locally

Requirements: Python 3.10+, a CUDA GPU (recommended), `ffmpeg` and `tesseract-ocr` installed.

```bash
pip install torch torchvision          # choose the build for your system
pip install --no-build-isolation -r requirements.txt

export WEIGHTS_PATH=/path/to/model_final.pth
python app.py                          # opens the web app
python app.py --share                  # also creates a temporary public link
python app.py --video lecture.mp4      # process one video without the UI
```

Outputs (PDF, JSON files, cropped figures) are saved in `outputs/` (change with `WORK_DIR`).

## Output files

| File | Content |
|---|---|
| `final_notes.pdf` | The generated notes |
| `module1_output.json` | Summaries with start/end timestamps |
| `module2_output.json` | Detected figures/tables with timestamp and OCR text |
| `module3_output.json` | Final summary-to-image alignment |

## Results

Add screenshots here, for example:

- Gradio frontend: `screenshots/frontend.png`
- Sample PDF page: `screenshots/pdf_result.png`

## Limitations

- Processing is slow without a GPU, because two Detectron2 models run on every sampled frame.
- Only figures and tables are extracted (text blocks, titles and lists are ignored).
- Summaries depend on the Whisper transcript quality, so noisy audio reduces accuracy.
- Each summary is matched to one transcript segment, so its timestamp covers a short range.

## Author

Shanmugaharini, B.E. Computer Science, Madras Institute of Technology (2022 to 2026)
