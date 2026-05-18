import tkinter as tk
from tkinter import ttk, messagebox
import json
import traceback

# Import the pipeline classifier
from pipeline_model import pipeline

class JA4ClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("JA4+ Prototype Pipeline Classifier")
        self.root.geometry("900x900")
        self.root.minsize(800, 800)
        
        # Apply a simple theme
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground="#2c3e50")
        style.configure("ResultLabel.TLabel", font=("Segoe UI", 11))
        style.configure("ResultValue.TLabel", font=("Segoe UI", 11, "bold"), foreground="#27ae60")
        style.configure("Card.TFrame", background="#f8f9fa", relief="solid", borderwidth=1)
        
        self.create_widgets()
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20 20 20 20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ── Header ──────────────────────────────────────────────
        ttk.Label(main_frame, text="JA4+ Fingerprint Classification", style="Title.TLabel").pack(pady=(0, 20))
        
        # ── Inputs Frame ────────────────────────────────────────
        input_frame = ttk.LabelFrame(main_frame, text="Input Fingerprints (Paste JSON Snippet)", padding="15 15 15 15")
        input_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Grid settings
        input_frame.columnconfigure(0, weight=1)
        
        self.txt_input = tk.Text(input_frame, height=8, width=80, font=("Consolas", 10), bg="#fdfdfd")
        self.txt_input.grid(row=0, column=0, sticky=tk.EW, pady=5)
        
        # Actions
        btn_frame = ttk.Frame(input_frame)
        btn_frame.grid(row=1, column=0, pady=10)
        
        ttk.Button(btn_frame, text="Classify", command=self.on_classify, width=20).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Clear", command=self.on_clear, width=10).pack(side=tk.LEFT, padx=10)
        
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(main_frame, textvariable=self.status_var, foreground="#7f8c8d").pack(anchor=tk.W, pady=(0, 10))

        # ── Results Frame ───────────────────────────────────────
        self.result_frame = ttk.LabelFrame(main_frame, text="Classification Results", padding="15 15 15 15")
        self.result_frame.pack(fill=tk.BOTH, expand=True)
        
        res_left = ttk.Frame(self.result_frame)
        res_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        res_left.columnconfigure(1, weight=1)
        
        res_right = ttk.Frame(self.result_frame)
        res_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # UI Elements for results
        self.res_app_var = tk.StringVar(value="-")
        self.res_cat_var = tk.StringVar(value="-")
        self.res_conf_var = tk.StringVar(value="-")
        self.res_src_var = tk.StringVar(value="-")
        
        ttk.Label(res_left, text="Application:", style="ResultLabel.TLabel").grid(row=0, column=0, sticky=tk.W, pady=5, padx=5)
        ttk.Label(res_left, textvariable=self.res_app_var, style="ResultValue.TLabel").grid(row=0, column=1, sticky=tk.W, pady=5)

        ttk.Label(res_left, text="Category:", style="ResultLabel.TLabel").grid(row=1, column=0, sticky=tk.W, pady=5, padx=5)
        ttk.Label(res_left, textvariable=self.res_cat_var, style="ResultValue.TLabel").grid(row=1, column=1, sticky=tk.W, pady=5)

        ttk.Label(res_left, text="Confidence:", style="ResultLabel.TLabel").grid(row=2, column=0, sticky=tk.W, pady=5, padx=5)
        ttk.Label(res_left, textvariable=self.res_conf_var, style="ResultValue.TLabel").grid(row=2, column=1, sticky=tk.W, pady=5)

        ttk.Label(res_left, text="Decision Source:", style="ResultLabel.TLabel").grid(row=3, column=0, sticky=tk.W, pady=5, padx=5)
        ttk.Label(res_left, textvariable=self.res_src_var, style="ResultValue.TLabel").grid(row=3, column=1, sticky=tk.W, pady=5)

        # Reasoning Text Box
        ttk.Label(res_left, text="Reasoning:", style="ResultLabel.TLabel").grid(row=4, column=0, sticky=tk.NW, pady=10, padx=5)
        self.txt_reasoning = tk.Text(res_left, height=6, width=40, wrap=tk.WORD, font=("Segoe UI", 10), bg="#f8f9fa")
        self.txt_reasoning.grid(row=4, column=1, sticky=tk.NSEW, pady=10)
        res_left.rowconfigure(4, weight=1)
        self.txt_reasoning.config(state=tk.DISABLED)
        
        # Top-K Treeview
        ttk.Label(res_right, text="Top Candidates:", style="ResultLabel.TLabel").pack(anchor=tk.NW, pady=5, padx=5)
        
        columns = ("rank", "application")
        self.tree_topk = ttk.Treeview(res_right, columns=columns, show="headings")
        self.tree_topk.heading("rank", text="Rank")
        self.tree_topk.heading("application", text="Application / Prediction")
        self.tree_topk.column("rank", width=50, anchor=tk.CENTER)
        self.tree_topk.column("application", width=300)
        
        self.tree_topk.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Initialize empty
        self.on_clear()

    def on_clear(self):
        self.txt_input.delete(1.0, tk.END)
        
        self.res_app_var.set("-")
        self.res_cat_var.set("-")
        self.res_conf_var.set("-")
        self.res_src_var.set("-")
        
        self.txt_reasoning.config(state=tk.NORMAL)
        self.txt_reasoning.delete(1.0, tk.END)
        self.txt_reasoning.config(state=tk.DISABLED)
        
        for item in self.tree_topk.get_children():
            self.tree_topk.delete(item)
            
        self.status_var.set("Ready.")

    def on_classify(self):
        import re
        raw_text = self.txt_input.get(1.0, tk.END).strip()
        
        if not raw_text:
            messagebox.showwarning("Input Error", "Please paste a JSON snippet to classify.")
            return

        parsed = {}
        # Try brute-force regex first since it handles partial/broken JSON robustly
        for match in re.finditer(r'"([^"]+)"\s*:\s*"([^"]+)"', raw_text):
            parsed[match.group(1)] = match.group(2)

        def _get(keys):
            for k in keys:
                if k in parsed:
                    return parsed[k]
            return None

        ja4 = _get(["ja4", "ja4_fingerprint"])
        ja4s = _get(["ja4s", "ja4s_fingerprint"])
        ja4ts = _get(["ja4ts"])
        ja4_string = _get(["ja4_string", "ja4_fingerprint_string"])
        ja4s_string = _get(["ja4s_string", "ja4s_fingerprint_string"])
        
        if not any([ja4, ja4s, ja4ts, ja4_string, ja4s_string]):
            messagebox.showwarning("Input Error", "Could not find valid JA4 fields in the input text. Ensure it is formatted like '\"ja4_fingerprint\": \"...\"'.")
            return
            
        self.status_var.set("Classifying...")
        self.root.update()
        
        try:
            # Run the pipeline module
            result = pipeline.classify(
                ja4=ja4,
                ja4s=ja4s,
                ja4ts=ja4ts,
                ja4_string=ja4_string,
                ja4s_string=ja4s_string,
                observation_id="gui_query"
            )
            
            # Display main results
            self.res_app_var.set(result.predicted_application or "Unknown")
            self.res_cat_var.set(result.predicted_category or "Unknown")
            
            conf_str = str(result.confidence).capitalize()
            self.res_conf_var.set(conf_str)
            self.res_src_var.set(result.decision_source)
            
            # Display reasoning
            self.txt_reasoning.config(state=tk.NORMAL)
            self.txt_reasoning.delete(1.0, tk.END)
            self.txt_reasoning.insert(tk.END, result.reasoning)
            self.txt_reasoning.config(state=tk.DISABLED)
            
            # Build Top-K List
            for item in self.tree_topk.get_children():
                self.tree_topk.delete(item)
                
            top_k = []
            if result.predicted_application:
                top_k.append(result.predicted_application)
                
            rf_top_k = result.model_details.get("rf", {}).get("top_k", [])
            for app in rf_top_k:
                if app not in top_k:
                    top_k.append(app)
                    
            eg_top_k = result.model_details.get("local", {}).get("candidates", [])
            for app in eg_top_k:
                if app not in top_k:
                    top_k.append(app)
                    
            for i, app in enumerate(top_k[:5]):
                self.tree_topk.insert("", tk.END, values=(f"#{i+1}", app))
                
            if not top_k:
                self.tree_topk.insert("", tk.END, values=("-", "No specific application candidates found."))
                
            self.status_var.set("Classification completed.")
            
        except Exception as e:
            err_msg = traceback.format_exc()
            messagebox.showerror("Pipeline Error", f"An error occurred during classification:\n\n{str(e)}")
            self.status_var.set("Error during classification.")

def main():
    root = tk.Tk()
    app = JA4ClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
