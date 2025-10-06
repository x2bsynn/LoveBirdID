import tkinter as tk
import threading
import time
import random

# ----------------------------
# UI CONFIG SECTION (EDIT HERE)
# ----------------------------

UI_CONFIG = {
    "window": {
        "title": "LovebirdID Displayz     ,  mn   ",
        "bg": "#ffffff",
        "size": "800x480"
    },
    "padding": 20,
    "audio_box": {
        "x": 20,
        "y": 20,
        "width": 760,
        "height": 210,
        "bg": "#ffffff",
        "border": "#000000",
        "title_font": ("Inter-Bold", 16,),
        "text_font": ("Arial", 10),
        "text_color": "#333333"
    },
    "restart_btn": {
        "x": 20,
        "y": 250,
        "width": 240,
        "height": 95,
        "bg": "#444455",
        "fg": "#ffffff",
        "font": ("Inter-Bold", 16, "bold")
    },
    "start_btn": {
        "x": 20,
        "y": 250 + 95 + 20,
        "width": 240,
        "height": 95,
        "bg": "#444455",
        "fg": "#ffffff",
        "font": ("Inter-Bold", 16, "bold")
    },
    "weight_box": {
        "x": 280,
        "y": 250,
        "width": 240,
        "height": 210,
        "bg": "#cc2222",
        "border": "#000000",
        "title_font": ("Arial", 16, "bold"),
        "text_font": ("Arial", 28, "bold"),
        "text_color": "#ffffff"
    },
    "gender_box": {
        "x": 540,
        "y": 250,
        "width": 240,
        "height": 210,
        "bg": "#cc2222",
        "border": "#000000",
        "title_font": ("Arial", 16, "bold"),
        
        #---SYMBOL & TEXT CONFIF ---
        "symbol_font": ("Arial", 40, "bold"),
        "male_symbol_color": "#00ccff",
        "female_symbol_color": "#ff66cc",
        
        "text_font": ("Arial", 20, "bold"),
        "text_color": "#ffffff"
    },
    "countdown": {
        "font": ("Arial", 20, "bold"),
        "color": "#d9534f"
    },
    "process_text": {  # <--- NEW: For "Processing...", "Measuring...", etc.
        "font": ("Arial", 18, "italic"),
        "color": "#333333"
    }
}

# ----------------------------
# MAIN APP CLASS
# ----------------------------

class AnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title(UI_CONFIG["window"]["title"])
        self.root.geometry(UI_CONFIG["window"]["size"])
        self.root.configure(bg=UI_CONFIG["window"]["bg"])
        self.root.resizable(False, False)

        self.is_running = False
        self.thread = None

        # AUDIO BOX
        self.audio_frame, self.audio_label = self.create_box(
            "LoveBirdID", "Waveform Display",
            UI_CONFIG["audio_box"]
        )

        # BUTTONS
        self.restart_button = tk.Button(
            root, text="Restart",
            command=self.restart,
            font=UI_CONFIG["restart_btn"]["font"],
            bg=UI_CONFIG["restart_btn"]["bg"],
            fg=UI_CONFIG["restart_btn"]["fg"]
        )
        self.restart_button.place(
            x=UI_CONFIG["restart_btn"]["x"],
            y=UI_CONFIG["restart_btn"]["y"],
            width=UI_CONFIG["restart_btn"]["width"],
            height=UI_CONFIG["restart_btn"]["height"]
        )

        self.start_button = tk.Button(
            root, text="Start\nAnalysis",
            command=self.start_analyzing,
            font=UI_CONFIG["start_btn"]["font"],
            bg=UI_CONFIG["start_btn"]["bg"],
            fg=UI_CONFIG["start_btn"]["fg"]
        )
        self.start_button.place(
            x=UI_CONFIG["start_btn"]["x"],
            y=UI_CONFIG["start_btn"]["y"],
            width=UI_CONFIG["start_btn"]["width"],
            height=UI_CONFIG["start_btn"]["height"]
        )

        # WEIGHT BOX
        self.weight_frame, self.weight_label = self.create_box(
            "Weight (g)", "0.00",
            UI_CONFIG["weight_box"]
        )

        # GENDER BOX (special case: two labels inside)
        self.gender_frame = tk.Frame(
            root,
            bg=UI_CONFIG["gender_box"]["bg"],
            highlightbackground=UI_CONFIG["gender_box"]["border"],
            highlightthickness=2
        )
        self.gender_frame.place(
            x=UI_CONFIG["gender_box"]["x"],
            y=UI_CONFIG["gender_box"]["y"],
            width=UI_CONFIG["gender_box"]["width"],
            height=UI_CONFIG["gender_box"]["height"]
        )

        tk.Label(
            self.gender_frame, text="Gender",
            font=UI_CONFIG["gender_box"]["title_font"],
            bg=UI_CONFIG["gender_box"]["bg"],
            fg=UI_CONFIG["gender_box"]["text_color"]
        ).pack(anchor="n", pady=5)

        self.gender_symbol = tk.Label(
            self.gender_frame, text="?",
            font=UI_CONFIG["gender_box"]["symbol_font"],
            bg=UI_CONFIG["gender_box"]["bg"],
            fg="#ffffff"  # Neutral color initially
        )
        self.gender_symbol.pack()

        self.gender_text = tk.Label(
            self.gender_frame, text="Unknown",
            font=UI_CONFIG["gender_box"]["text_font"],
            bg=UI_CONFIG["gender_box"]["bg"],
            fg=UI_CONFIG["gender_box"]["text_color"]
        )
        self.gender_text.pack()

    def create_box(self, title, initial_text, config):
        frame = tk.Frame(
            self.root,
            bg=config["bg"],
            highlightbackground=config["border"],
            highlightthickness=2
        )
        frame.place(x=config["x"], y=config["y"],
                    width=config["width"], height=config["height"])

        tk.Label(
            frame, text=title,
            font=config["title_font"],
            bg=config["bg"], fg=config["text_color"]
        ).pack(anchor="n", pady=5)

        content_label = tk.Label(
            frame, text=initial_text,
            font=config["text_font"],
            bg=config["bg"], fg=config["text_color"]
        )
        content_label.pack(expand=True)

        return frame, content_label

    def start_analyzing(self):
        if self.is_running:
            return
        self.is_running = True

        self.audio_label.config(
            text="Audio Recording...",
            font=UI_CONFIG["process_text"]["font"],
            fg=UI_CONFIG["process_text"]["color"]
        )
        self.weight_label.config(
            text="Measuring...",
            font=UI_CONFIG["process_text"]["font"],
            fg=UI_CONFIG["process_text"]["color"]
        )
        self.gender_text.config(
            text="Processing...",
            font=UI_CONFIG["process_text"]["font"],
            fg=UI_CONFIG["process_text"]["color"]
        )
        self.gender_symbol.config(text="?")

        self.thread = threading.Thread(target=self.run_analysis)
        self.thread.start()

    def run_analysis(self):
        for i in range(10, 0, -1):
            if not self.is_running:
                return
            self.audio_label.config(
                text=f"Audio Recording...\n{i}s left",
                fg=UI_CONFIG["countdown"]["color"],
                font=UI_CONFIG["countdown"]["font"]
            )
            time.sleep(1)

        if not self.is_running:
            return

        self.audio_label.config(
            text="(Waveform Displayed Here)",
            fg=UI_CONFIG["audio_box"]["text_color"],
            font=UI_CONFIG["audio_box"]["text_font"]
        )

        weight = round(random.uniform(4.5, 6.5), 2)
        self.weight_label.config(
            text=str(weight),
            font=UI_CONFIG["weight_box"]["text_font"],
            fg=UI_CONFIG["weight_box"]["text_color"]
        )

        time.sleep(2)
        if not self.is_running:
            return

        gender_choice = random.choice([("♂", "Male"), ("♀", "Female")])
        symbol, text = gender_choice

        if symbol == "♂":
            symbol_color = UI_CONFIG["gender_box"]["male_symbol_color"]
            
        else:
            symbol_color = UI_CONFIG["gender_box"]["female_symbol_color"]
        
        self.gender_symbol.config(
            text=symbol,
            font=UI_CONFIG["gender_box"]["symbol_font"],
            fg=symbol_color
        )
        self.gender_text.config(
            text=text,
            font=UI_CONFIG["gender_box"]["text_font"],
            fg=UI_CONFIG["gender_box"]["text_color"]
        )

        self.is_running = False

    def restart(self):
        self.is_running = False
        self.audio_label.config(
            text="Waveform Display",
            font=UI_CONFIG["audio_box"]["text_font"],
            fg=UI_CONFIG["audio_box"]["text_color"]
        )
        self.weight_label.config(
            text="0.00",
            font=UI_CONFIG["weight_box"]["text_font"],
            fg=UI_CONFIG["weight_box"]["text_color"]
        )
        self.gender_symbol.config(text="?", fg=UI_CONFIG["gender_box"]["symbol_color"])
        self.gender_text.config(
            text="Unknown",
            font=UI_CONFIG["gender_box"]["text_font"],
            fg=UI_CONFIG["gender_box"]["text_color"]
        )


# ----------------------------
# RUN APP
# ----------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalyzerApp(root)
    root.mainloop()
