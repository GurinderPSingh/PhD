import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
from PIL import Image, ImageTk
import zlib
import base64
import webbrowser
import subprocess
import tempfile
import os

# Use "mmdc.cmd" if it is in your system PATH; otherwise, update with the correct absolute path.
mmdc_path = "mmdc.cmd"  # or, e.g., r"C:\Users\gurin\AppData\Roaming\npm\mmdc.cmd"

######################################
# FUNCTION: Open in Mermaid Live Editor
######################################
def open_in_mermaid_live_editor(mermaid_code: str):
    try:
        code = mermaid_code.strip()
        if not code:
            messagebox.showwarning("Warning", "Please enter some Mermaid code.")
            return

        # Compress and encode the code (as used by mermaid.live)
        compressed = zlib.compress(code.encode('utf-8'), level=9)
        b64 = base64.urlsafe_b64encode(compressed).decode('utf-8')
        url = f"https://mermaid.live/edit#pako:{b64}"
        webbrowser.open(url)
    except Exception as e:
        messagebox.showerror("Error", f"Could not generate link:\n{e}")

######################################
# FUNCTION: Generate Preview Diagram
######################################
def generate_preview(mermaid_code: str) -> str:
    # Write the Mermaid code to a temporary file with UTF-8 encoding
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".mmd", encoding="utf-8") as tmp:
        tmp.write(mermaid_code)
        tmp_file = tmp.name

    # Output preview file (PNG) in the system temp directory
    preview_file = os.path.join(tempfile.gettempdir(), "diagram_preview.png")
    try:
        result = subprocess.run(
            [mmdc_path, "-i", tmp_file, "-o", preview_file],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            messagebox.showerror("Error", f"Error generating preview diagram:\n{result.stderr}")
            return None
        return preview_file
    finally:
        os.remove(tmp_file)

######################################
# FUNCTION: Save High Quality Diagram
######################################
def save_high_quality(mermaid_code: str):
    save_path = filedialog.asksaveasfilename(
        defaultextension=".png", filetypes=[("PNG files", "*.png")],
        title="Save High Quality Diagram"
    )
    if not save_path:
        return

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".mmd", encoding="utf-8") as tmp:
        tmp.write(mermaid_code)
        tmp_file = tmp.name

    try:
        # Use a scale factor (e.g., scale 2 for higher quality)
        result = subprocess.run(
            [mmdc_path, "-i", tmp_file, "-o", save_path, "--scale", "2"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            messagebox.showerror("Error", f"Error generating high quality diagram:\n{result.stderr}")
        else:
            messagebox.showinfo("Success", f"High quality diagram saved to:\n{save_path}")
    finally:
        os.remove(tmp_file)

######################################
# GUI Action Functions
######################################
def generate_and_show_preview():
    code = text_area.get("1.0", tk.END).strip()
    if not code:
        messagebox.showwarning("Warning", "Please enter some Mermaid code.")
        return
    preview_file = generate_preview(code)
    if preview_file is not None:
        try:
            img = Image.open(preview_file)
            tk_img = ImageTk.PhotoImage(img)
            output_canvas.delete("all")
            output_canvas.create_image(0, 0, anchor="nw", image=tk_img)
            output_canvas.image = tk_img  # Keep reference so it isn’t garbage-collected
            # Set the scrollable region to match the image size
            output_canvas.config(scrollregion=(0, 0, img.width, img.height))
        except Exception as e:
            messagebox.showerror("Error", f"Error loading preview image:\n{e}")

def open_live_editor():
    code = text_area.get("1.0", tk.END)
    open_in_mermaid_live_editor(code)

def save_hq():
    code = text_area.get("1.0", tk.END).strip()
    if not code:
        messagebox.showwarning("Warning", "Please enter some Mermaid code.")
        return
    save_high_quality(code)

######################################
# Functions to Adjust PanedWindow Sash
######################################
def expand_code_editor():
    x, y = paned_window.sash_coord(0)
    paned_window.sash_place(0, x + 20, y)

def expand_diagram_viewer():
    x, y = paned_window.sash_coord(0)
    paned_window.sash_place(0, x - 20, y)

######################################
# SET UP THE GUI
######################################
root = tk.Tk()
root.title("Mermaid Diagram Tool")

# PanedWindow with adjustable sash
paned_window = tk.PanedWindow(root, orient=tk.HORIZONTAL, sashwidth=10, sashrelief="raised")
paned_window.pack(fill=tk.BOTH, expand=True)

# Left pane: Code Editor
left_frame = tk.Frame(paned_window)
paned_window.add(left_frame, minsize=300)
label_code = tk.Label(left_frame, text="Mermaid Code:")
label_code.pack(pady=5)
text_area = scrolledtext.ScrolledText(left_frame, width=50, height=25)
text_area.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

# Right pane: Diagram Preview using a scrollable Canvas
right_frame = tk.Frame(paned_window)
paned_window.add(right_frame, minsize=300)
label_preview = tk.Label(right_frame, text="Diagram Preview:")
label_preview.pack(pady=5)
output_canvas = tk.Canvas(right_frame, background="white")
output_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Vertical scrollbar
scrollbar_v = tk.Scrollbar(right_frame, orient="vertical", command=output_canvas.yview)
scrollbar_v.pack(side=tk.RIGHT, fill="y")
# Horizontal scrollbar
scrollbar_h = tk.Scrollbar(right_frame, orient="horizontal", command=output_canvas.xview)
scrollbar_h.pack(side=tk.BOTTOM, fill="x")
output_canvas.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)

# Bind mouse wheel events for vertical scrolling
def on_mousewheel(event):
    output_canvas.yview_scroll(-1 * int(event.delta / 120), "units")

output_canvas.bind("<MouseWheel>", on_mousewheel)
output_canvas.bind("<Button-4>", lambda event: output_canvas.yview_scroll(-1, "units"))
output_canvas.bind("<Button-5>", lambda event: output_canvas.yview_scroll(1, "units"))

# Bottom button frame with controls to adjust pane sizes
button_frame = tk.Frame(root)
button_frame.pack(pady=5)
btn_live_editor = tk.Button(button_frame, text="Test in Live Editor", command=open_live_editor)
btn_live_editor.pack(side=tk.LEFT, padx=5)
btn_generate = tk.Button(button_frame, text="Generate Preview", command=generate_and_show_preview)
btn_generate.pack(side=tk.LEFT, padx=5)
btn_save = tk.Button(button_frame, text="Save High Quality PNG", command=save_hq)
btn_save.pack(side=tk.LEFT, padx=5)
btn_expand_code = tk.Button(button_frame, text="Expand Code Editor", command=expand_code_editor)
btn_expand_code.pack(side=tk.LEFT, padx=5)
btn_expand_viewer = tk.Button(button_frame, text="Expand Diagram Viewer", command=expand_diagram_viewer)
btn_expand_viewer.pack(side=tk.LEFT, padx=5)

root.mainloop()
