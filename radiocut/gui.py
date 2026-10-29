"""Beginner-friendly Tk GUI: drop a file, pick models, Process, review, Export."""
import json, os, threading, queue, tkinter as tk
from tkinter import ttk, filedialog, messagebox
from . import __version__
from .pipeline import analyze, save_project, DEFAULTS
from .export import render, write_cutlists
from .transcribe import MODELS
from .llm import PRESETS, list_models
from .util import fmt_ts, app_dir

SETTINGS = os.path.join(app_dir(), "settings.json")


class App:
    def __init__(self, root):
        self.root = root; self.res = None; self.q = queue.Queue()
        root.title(f"RadioCut {__version__} - talk radio to YouTube, offline")
        root.geometry("1000x680")
        s = dict(DEFAULTS); s["llm_engine"] = "None (heuristics only)"
        try:
            s.update(json.load(open(SETTINGS)))
        except Exception:
            pass
        self.path = tk.StringVar()
        self.wm = tk.StringVar(value=s["whisper_model"])
        self.engine = tk.StringVar(value=s["llm_engine"])
        self.url = tk.StringVar(value=s["llm_base"]); self.lm = tk.StringVar(value=s["llm_model"])
        self.th = tk.DoubleVar(value=s["threshold"]); self.split = tk.BooleanVar(value=False)
        self.still = tk.StringVar()
        f = ttk.Frame(root, padding=8); f.pack(fill="x")
        self.drop = tk.Label(f, text="Drop a recording here (mp4, mp3, wav, m4a...) or click to browse",
                             relief="groove", height=3, bg="#eef6fb", cursor="hand2")
        self.drop.pack(fill="x"); self.drop.bind("<Button-1>", lambda e: self.browse())
        try:
            from tkinterdnd2 import DND_FILES
            self.drop.drop_target_register(DND_FILES)
            self.drop.dnd_bind("<<Drop>>", lambda e: self.set_file(root.tk.splitlist(e.data)[0]))
        except Exception:
            pass
        g = ttk.Frame(root, padding=(8, 0)); g.pack(fill="x")
        ttk.Label(g, text="Speech model").grid(row=0, column=0, sticky="w")
        ttk.Combobox(g, textvariable=self.wm, values=MODELS, width=10, state="readonly").grid(row=0, column=1, sticky="w")
        ttk.Label(g, text="  Local AI").grid(row=0, column=2)
        cb = ttk.Combobox(g, textvariable=self.engine, width=22, state="readonly",
                          values=["None (heuristics only)", "LM Studio", "Ollama", "Custom"])
        cb.grid(row=0, column=3); cb.bind("<<ComboboxSelected>>", self.engine_changed)
        ttk.Entry(g, textvariable=self.url, width=28).grid(row=0, column=4, padx=4)
        self.lmbox = ttk.Combobox(g, textvariable=self.lm, width=30); self.lmbox.grid(row=0, column=5)
        ttk.Button(g, text="Refresh models", command=self.refresh).grid(row=0, column=6, padx=4)
        ttk.Label(g, text="Sensitivity (lower = cut more)").grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Scale(g, variable=self.th, from_=0.3, to=0.8).grid(row=1, column=2, columnspan=2, sticky="we")
        ttk.Checkbutton(g, text="Also export one clip per kept segment", variable=self.split).grid(row=1, column=4, columnspan=2, sticky="w")
        ttk.Button(g, text="Still image (audio-only)...", command=self.pick_still).grid(row=1, column=6)
        b = ttk.Frame(root, padding=8); b.pack(fill="x")
        self.pbtn = ttk.Button(b, text="1. Process", command=self.process); self.pbtn.pack(side="left")
        self.ebtn = ttk.Button(b, text="2. Export video + cut list", command=self.export, state="disabled"); self.ebtn.pack(side="left", padx=6)
        ttk.Button(b, text="Toggle keep/remove", command=self.toggle).pack(side="left")
        self.pb = ttk.Progressbar(b, length=250); self.pb.pack(side="left", padx=10)
        self.status = ttk.Label(b, text="Ready"); self.status.pack(side="left")
        cols = ("action", "start", "end", "dur", "label", "why")
        self.tv = ttk.Treeview(root, columns=cols, show="headings")
        for c, w in zip(cols, (70, 80, 80, 70, 80, 560)):
            self.tv.heading(c, text=c.title()); self.tv.column(c, width=w, anchor="w")
        self.tv.pack(fill="both", expand=True, padx=8)
        self.tv.bind("<Double-1>", lambda e: self.toggle())
        self.tv.tag_configure("rm", background="#fde2e2"); self.tv.tag_configure("keep", background="#e6f6e6")
        self.summary = ttk.Label(root, padding=8, text="Double-click a row to keep/remove it.")
        self.summary.pack(fill="x")
        self.engine_changed(); root.after(150, self.poll)

    def engine_changed(self, *_):
        e = self.engine.get()
        if e in PRESETS:
            self.url.set(PRESETS[e]); self.refresh(quiet=True)

    def refresh(self, quiet=False):
        try:
            ms = list_models(self.url.get()); self.lmbox["values"] = ms
            if ms and self.lm.get() not in ms:
                self.lm.set(ms[0])
        except Exception as e:
            if not quiet:
                messagebox.showwarning("Local AI", f"Could not reach {self.url.get()}\nStart LM Studio's server (Developer tab) or Ollama.\n\n{e}")

    def browse(self):
        p = filedialog.askopenfilename(filetypes=[("Media", "*.mp4 *.mkv *.mov *.mp3 *.wav *.m4a *.aac *.flac *.ogg *.webm"), ("All", "*.*")])
        if p: self.set_file(p)

    def set_file(self, p):
        self.path.set(p); self.drop.config(text=os.path.basename(p))

    def pick_still(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if p: self.still.set(p)

    def cfg(self):
        c = {"whisper_model": self.wm.get(), "threshold": round(self.th.get(), 2),
             "use_llm": not self.engine.get().startswith("None"), "llm_base": self.url.get(), "llm_model": self.lm.get(),
             "llm_engine": self.engine.get()}
        json.dump(c, open(SETTINGS, "w")); return c

    def process(self):
        if not self.path.get():
            return self.browse()
        cfg = self.cfg(); self.pbtn.config(state="disabled")
        def work():
            try:
                r = analyze(self.path.get(), cfg, progress=lambda p, m: self.q.put(("p", p, m)),
                            log=lambda m: self.q.put(("p", None, m)))
                self.q.put(("done", r))
            except Exception as e:
                self.q.put(("err", str(e)))
        threading.Thread(target=work, daemon=True).start()

    def export(self):
        out = filedialog.asksaveasfilename(defaultextension=".mp4", initialfile=os.path.splitext(os.path.basename(self.path.get()))[0] + "-radiocut.mp4")
        if not out: return
        split, still = self.split.get(), self.still.get() or None
        self.ebtn.config(state="disabled")
        def work():
            try:
                stem = os.path.splitext(out)[0]
                save_project(self.res, stem + ".project.json"); write_cutlists(self.res, stem)
                self.q.put(("p", 0.1, "Rendering video..."))
                render(self.res["source"], self.res["segments"], out, still=still)
                if split:
                    ks = [s for s in self.res["segments"] if s["keep"]]
                    for i, s in enumerate(ks, 1):
                        self.q.put(("p", i / len(ks), f"Clip {i}/{len(ks)}"))
                        render(self.res["source"], [s], f"{stem}.part{i:02d}.mp4", still=still)
                self.q.put(("exported", out))
            except Exception as e:
                self.q.put(("err", str(e)))
        threading.Thread(target=work, daemon=True).start()

    def toggle(self):
        if not self.res: return
        for iid in self.tv.selection():
            s = self.res["segments"][int(iid)]; s["keep"] = not s["keep"]
        self.fill()

    def fill(self):
        self.tv.delete(*self.tv.get_children()); kept = 0
        for i, s in enumerate(self.res["segments"]):
            d = s["end"] - s["start"]; kept += d if s["keep"] else 0
            self.tv.insert("", "end", iid=str(i), tags=("keep" if s["keep"] else "rm",),
                           values=("KEEP" if s["keep"] else "REMOVE", fmt_ts(s["start"]), fmt_ts(s["end"]), fmt_ts(d),
                                   s["label"], s["reason"] or s["preview"]))
        self.summary.config(text=f"Keeping {fmt_ts(kept)} of {fmt_ts(self.res['duration'])}. Double-click a row to keep/remove it.")

    def poll(self):
        try:
            while True:
                m = self.q.get_nowait()
                if m[0] == "p":
                    if m[1] is not None: self.pb["value"] = m[1] * 100
                    self.status.config(text=m[2])
                elif m[0] == "done":
                    self.res = m[1]; self.fill(); self.pbtn.config(state="normal"); self.ebtn.config(state="normal")
                    self.status.config(text="Review the list, then Export")
                elif m[0] == "exported":
                    self.ebtn.config(state="normal"); self.status.config(text="Exported")
                    messagebox.showinfo("RadioCut", f"Saved {m[1]}\nplus .cutlist.csv/.json, .edl and .youtube-chapters.txt")
                elif m[0] == "err":
                    self.pbtn.config(state="normal"); self.ebtn.config(state="normal" if self.res else "disabled")
                    messagebox.showerror("RadioCut", m[1])
        except queue.Empty:
            pass
        self.root.after(150, self.poll)


def main():
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except Exception:
        root = tk.Tk()
    App(root); root.mainloop()
