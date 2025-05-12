
import os
import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

class SmartCCTVApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart CCTV App")

        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.frame_skip = tk.IntVar(value=5)
        self.diff_threshold = tk.IntVar(value=30)
        self.warmup_frames = tk.IntVar(value=30)
        self.screenshots_folder = "screenshots"
        self.model = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=50, detectShadows=False)
        self.detected_classes = set()

        self.canvas = tk.Canvas(root, width=640, height=480, bg="black")
        self.canvas.grid(row=0, column=0, columnspan=7, padx=10, pady=10)
        self.canvas_image = None
        self.canvas_image_tk = None

        tk.Button(root, text="Upload Video", command=self.load_video).grid(row=1, column=0, padx=5)
        tk.Label(root, text="Frame Skip:").grid(row=1, column=1, sticky="e")
        tk.Spinbox(root, from_=1, to=60, width=5, textvariable=self.frame_skip).grid(row=1, column=2, sticky="w")
        tk.Label(root, text="Diff Thresh:").grid(row=1, column=3, sticky="e")
        tk.Spinbox(root, from_=1, to=255, width=5, textvariable=self.diff_threshold).grid(row=1, column=4, sticky="w")
        tk.Label(root, text="Warmup:").grid(row=1, column=5, sticky="e")
        tk.Spinbox(root, from_=0, to=300, width=5, textvariable=self.warmup_frames).grid(row=1, column=6, sticky="w")

        self.start_btn = tk.Button(root, text="Start Processing", state="disabled", command=self.start_processing)
        self.start_btn.grid(row=2, column=0, columnspan=2, pady=10)

        self.progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.progress.grid(row=2, column=2, columnspan=5, padx=5)

        self.status_lbl = tk.Label(root, text="Status: idle")
        self.status_lbl.grid(row=3, column=0, columnspan=7, pady=5)

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files","*.mp4 *.avi *.mov")])
        if not path:
            return
        self.video_path = path
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        os.makedirs(self.screenshots_folder, exist_ok=True)
        self.start_btn.config(state="normal")
        self.status_lbl.config(text=f"Loaded: {os.path.basename(path)} ({self.total_frames} frames)")
        try:
            self.status_lbl.config(text="Loading YOLOv8 model…")
            self.root.update()
            self.model = YOLO("yolov8n.pt")
            self.status_lbl.config(text="Model loaded — ready to start")
        except Exception as e:
            messagebox.showerror("Model error", f"Failed to load model:\n{e}")
            self.start_btn.config(state="disabled")

    def _update_canvas(self, frame):
        disp = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self.canvas_image_tk = ImageTk.PhotoImage(img)
        if self.canvas_image is None:
            self.canvas_image = self.canvas.create_image(0, 0, anchor="nw", image=self.canvas_image_tk)
        else:
            self.canvas.itemconfig(self.canvas_image, image=self.canvas_image_tk)

    def start_processing(self):
        if not self.cap or not self.model:
            return
        self.start_btn.config(state="disabled")
        self.progress["value"] = 0
        self.status_lbl.config(text="Processing…")
        threading.Thread(target=self.process_video, daemon=True).start()

    def process_video(self):
        frame_idx = 0
        saved_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_idx += 1

            mask = self.bg_subtractor.apply(frame)
            if frame_idx <= self.warmup_frames.get():
                self.root.after(0, self._update_canvas, frame)
                continue

            if frame_idx % self.frame_skip.get() != 0:
                self.root.after(0, self._update_canvas, frame)
                continue

            kernel = np.ones((5,5), np.uint8)
            mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask_clean = cv2.dilate(mask_clean, kernel, iterations=2)
            cnts, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not any(cv2.contourArea(c) > 1000 for c in cnts):
                self.root.after(0, self._update_canvas, frame)
                continue

            results = self.model(frame)[0]
            annotated = frame.copy()
            current_detected = set()
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = mask_clean[y1:y2, x1:x2]
                if crop.size == 0 or np.count_nonzero(crop)/crop.size < 0.1:
                    continue
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                current_detected.add(cls_name)
                if cls_name in self.detected_classes:
                    conf = float(box.conf[0])
                    label = f"{cls_name} {conf:.2f}"
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(annotated, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            if current_detected:
                self.detected_classes.update(current_detected)
                ts = time.strftime("%Y%m%d_%H%M%S")
                fname = f"frame{frame_idx}_{ts}.jpg"
                path = os.path.join(self.screenshots_folder, fname)
                cv2.imwrite(path, annotated)
                saved_count += 1

            display = annotated if any(self.model.names[int(b.cls[0])] in self.detected_classes for b in results.boxes) else frame
            self.root.after(0, self._update_canvas, display)

            progress_pct = frame_idx / self.total_frames * 100
            self.root.after(0, lambda p=progress_pct: self.progress.config(value=p))
            self.root.after(0, lambda s=f"Processed {frame_idx}/{self.total_frames}, saved {saved_count}":
                                   self.status_lbl.config(text=s))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.root.after(0, lambda: self.start_btn.config(state="normal"))
        self.root.after(0, lambda s=f"Finished: {saved_count} screenshots":
                                   self.status_lbl.config(text=s))

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartCCTVApp(root)
    root.mainloop()
