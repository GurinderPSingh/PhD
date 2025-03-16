import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
from PIL import Image, ImageTk
import zlib
import base64
import webbrowser
import subprocess
import tempfile
import os

# Set the mmdc command.
# If mmdc is in your system PATH, you can use "mmdc.cmd".
# Otherwise, update this path to the full location of your mmdc.cmd.
mmdc_path = "mmdc.cmd"  # or r"C:\Users\YourUsername\AppData\Roaming\npm\mmdc.cmd"


###############################################
# Helper Functions for Graphics (Icon & Header)
###############################################
def set_window_icon(root, icon_path="icon.png"):
    """Sets the window icon if the icon file exists."""
    if os.path.exists(icon_path):
        try:
            icon = ImageTk.PhotoImage(file=icon_path)
            root.iconphoto(False, icon)
        except Exception as e:
            print("Could not load window icon:", e)


def load_header_image(frame, header_path="header.png"):
    """Loads a header/banner image if available and adds it to the given frame."""
    if os.path.exists(header_path):
        try:
            header_img = Image.open(header_path)
            header_tk = ImageTk.PhotoImage(header_img)
            header_label = tk.Label(frame, image=header_tk)
            header_label.image = header_tk  # keep reference
            header_label.pack(pady=5)
        except Exception as e:
            print("Could not load header image:", e)


###############################################
# Core Functionalities
###############################################
def open_in_mermaid_live_editor(mermaid_code: str):
    """Encodes the Mermaid code and opens it in the Mermaid Live Editor."""
    try:
        code = mermaid_code.strip()
        if not code:
            messagebox.showwarning("Warning", "Please enter some Mermaid code.")
            return
        # Compress and encode (as used by mermaid.live)
        compressed = zlib.compress(code.encode('utf-8'), level=9)
        b64 = base64.urlsafe_b64encode(compressed).decode('utf-8')
        url = f"https://mermaid.live/edit#pako:{b64}"
        webbrowser.open(url)
    except Exception as e:
        messagebox.showerror("Error", f"Could not generate link:\n{e}")


def generate_preview(mermaid_code: str) -> str:
    """
    Renders a Mermaid diagram as a PNG preview by writing the code to a temporary file
    and calling the Mermaid CLI.

    Returns the path to the generated preview PNG.
    """
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".mmd") as tmp:
        tmp.write(mermaid_code)
        tmp_file = tmp.name
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


def save_high_quality(mermaid_code: str):
    """
    Generates a high-quality PNG using a scale factor and opens a Save As dialog so the user can save it.
    """
    save_path = filedialog.asksaveasfilename(
        defaultextension=".png", filetypes=[("PNG files", "*.png")],
        title="Save High Quality Diagram"
    )
    if not save_path:
        return
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".mmd") as tmp:
        tmp.write(mermaid_code)
        tmp_file = tmp.name
    try:
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


###############################################
# GUI Action Functions
###############################################
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
            output_canvas.image = tk_img  # Keep reference
            # Set the scrollable region to the image size
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


def expand_code_editor():
    """Expands the code editor pane by moving the sash."""
    x, y = paned_window.sash_coord(0)
    paned_window.sash_place(0, x + 20, y)


def expand_diagram_viewer():
    """Expands the diagram viewer pane by moving the sash."""
    x, y = paned_window.sash_coord(0)
    paned_window.sash_place(0, x - 20, y)


def on_mousewheel(event):
    # For Windows (event.delta is multiple of 120)
    output_canvas.yview_scroll(-1 * int(event.delta / 120), "units")


###############################################
# SET UP THE GUI
###############################################
root = tk.Tk()
root.title("Mermaid Diagram Tool")

# Set custom window icon (if icon.png is provided)
set_window_icon(root, "icon.png")

# Main frame for header and content
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Optional header frame with a banner image (if header.png is provided)
header_frame = tk.Frame(main_frame)
header_frame.pack(fill=tk.X)
load_header_image(header_frame, "header.png")

# PanedWindow for adjustable code editor and preview viewer
paned_window = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL, sashwidth=10, sashrelief="raised")
paned_window.pack(fill=tk.BOTH, expand=True)

# Left pane: Code Editor
left_frame = tk.Frame(paned_window)
paned_window.add(left_frame, minsize=300)
label_code = tk.Label(left_frame, text="Mermaid Code:", font=("Helvetica", 12, "bold"))
label_code.pack(pady=5)
text_area = scrolledtext.ScrolledText(left_frame, width=50, height=25, font=("Consolas", 11))
text_area.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

# Right pane: Diagram Preview (with a scrollable Canvas)
right_frame = tk.Frame(paned_window)
paned_window.add(right_frame, minsize=300)
label_preview = tk.Label(right_frame, text="Diagram Preview:", font=("Helvetica", 12, "bold"))
label_preview.pack(pady=5)
output_canvas = tk.Canvas(right_frame, background="white")
output_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar_v = tk.Scrollbar(right_frame, orient="vertical", command=output_canvas.yview)
scrollbar_v.pack(side=tk.RIGHT, fill="y")
scrollbar_h = tk.Scrollbar(right_frame, orient="horizontal", command=output_canvas.xview)
scrollbar_h.pack(side=tk.BOTTOM, fill="x")
output_canvas.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
output_canvas.bind("<MouseWheel>", on_mousewheel)
output_canvas.bind("<Button-4>", lambda event: output_canvas.yview_scroll(-1, "units"))
output_canvas.bind("<Button-5>", lambda event: output_canvas.yview_scroll(1, "units"))

# Bottom button frame with additional pane-adjustment controls
button_frame = tk.Frame(root)
button_frame.pack(pady=5)
btn_live_editor = tk.Button(button_frame, text="Test in Live Editor", command=open_live_editor, font=("Helvetica", 10))
btn_live_editor.pack(side=tk.LEFT, padx=5)
btn_generate = tk.Button(button_frame, text="Generate Preview", command=generate_and_show_preview,
                         font=("Helvetica", 10))
btn_generate.pack(side=tk.LEFT, padx=5)
btn_save = tk.Button(button_frame, text="Save High Quality PNG", command=save_hq, font=("Helvetica", 10))
btn_save.pack(side=tk.LEFT, padx=5)
btn_expand_code = tk.Button(button_frame, text="Expand Code Editor", command=expand_code_editor, font=("Helvetica", 10))
btn_expand_code.pack(side=tk.LEFT, padx=5)
btn_expand_viewer = tk.Button(button_frame, text="Expand Diagram Viewer", command=expand_diagram_viewer,
                              font=("Helvetica", 10))
btn_expand_viewer.pack(side=tk.LEFT, padx=5)

root.mainloop()
