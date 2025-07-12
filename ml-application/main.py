import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageGrab
import threading
import time
import os
from dotenv import load_dotenv
import openai
import base64
import io
import mss
import mss.tools

class ScreenCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Context Capture & Analysis")
        self.root.geometry("800x600")
        
        # Load environment variables
        load_dotenv()
        self.openai_key = os.getenv('OPENAI_KEY')
        
        if not self.openai_key:
            messagebox.showerror("Error", "OPENAI_KEY not found in .env file!")
            return
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.openai_key)
        
        # Capture state
        self.captured_image = None
        self.capture_window = None
        self.is_capturing = False
        self.selection_start = None
        self.selection_end = None
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Controls
        self.setup_control_panel(main_frame)
        
        # Right panel - Display and Chat
        self.setup_display_panel(main_frame)
        
    def setup_control_panel(self, parent):
        """Setup the left control panel"""
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Capture button
        self.capture_btn = ttk.Button(control_frame, text="🎯 Start Screen Capture", 
                                     command=self.start_capture, style="Accent.TButton")
        self.capture_btn.grid(row=0, column=0, pady=(0, 10), sticky=(tk.W, tk.E))
        
        # Instructions
        instructions = ttk.Label(control_frame, 
                               text="1. Click 'Start Screen Capture'\n"
                                    "2. Draw a rectangle over the area\n"
                                    "3. Press Enter to capture\n"
                                    "4. Enter your prompt below",
                               wraplength=200, justify=tk.LEFT)
        instructions.grid(row=1, column=0, pady=(0, 20), sticky=(tk.W, tk.E))
        
        # Prompt input
        ttk.Label(control_frame, text="Your Prompt:").grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        self.prompt_entry = ttk.Entry(control_frame, width=30)
        self.prompt_entry.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Example prompts
        example_frame = ttk.LabelFrame(control_frame, text="Example Prompts", padding="5")
        example_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        examples = [
            "What do you see in this image?",
            "Describe the content in detail",
            "What is the main topic or subject?",
            "Analyze the layout and structure",
            "What actions could I take based on this?"
        ]
        
        for i, example in enumerate(examples):
            btn = ttk.Button(example_frame, text=example, 
                           command=lambda e=example: self.prompt_entry.delete(0, tk.END) or self.prompt_entry.insert(0, e))
            btn.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=1)
        
        # Send button
        self.send_btn = ttk.Button(control_frame, text="🚀 Send to AI", 
                                 command=self.send_to_ai, state='disabled')
        self.send_btn.grid(row=5, column=0, pady=(10, 0), sticky=(tk.W, tk.E))
        
        # Status
        self.status_label = ttk.Label(control_frame, text="Ready to capture", 
                                     foreground="green")
        self.status_label.grid(row=6, column=0, pady=(10, 0), sticky=(tk.W, tk.E))
        
    def setup_display_panel(self, parent):
        """Setup the right display panel"""
        display_frame = ttk.Frame(parent)
        display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(1, weight=1)
        
        # Captured image display
        image_frame = ttk.LabelFrame(display_frame, text="Captured Image", padding="10")
        image_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        image_frame.columnconfigure(0, weight=1)
        
        # Create canvas for image display
        self.image_canvas = tk.Canvas(image_frame, width=400, height=300, bg='white')
        self.image_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Placeholder text
        self.image_canvas.create_text(200, 150, text="No image captured yet", 
                                    fill="gray", font=("Arial", 12))
        
        # Chat display
        chat_frame = ttk.LabelFrame(display_frame, text="AI Response", padding="10")
        chat_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        self.chat_text = scrolledtext.ScrolledText(chat_frame, height=15, width=50)
        self.chat_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def start_capture(self):
        """Start the screen capture process"""
        self.is_capturing = True
        self.capture_btn.config(state='disabled')
        self.status_label.config(text="Starting capture...", foreground="orange")
        
        # Use a different approach - capture first, then let user select area
        self.capture_full_screen()
        
    def capture_full_screen(self):
        """Capture all monitors and let user select area"""
        try:
            # Capture all monitors
            with mss.mss() as sct:
                # Get all monitors
                monitors = sct.monitors[1:]  # Skip the "all monitors" entry
                
                if len(monitors) == 1:
                    # Single monitor - capture directly
                    screenshot = sct.grab(monitors[0])
                    self.full_screen_image = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
                    self.monitor_info = monitors[0]
                else:
                    # Multiple monitors - create a combined image
                    self.create_combined_monitor_image(sct, monitors)
                
            # Create selection window
            self.create_selection_window()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture screen: {str(e)}")
            self.cancel_capture()
    
    def create_combined_monitor_image(self, sct, monitors):
        """Create a combined image from all monitors"""
        # Calculate total dimensions
        min_x = min(monitor['left'] for monitor in monitors)
        min_y = min(monitor['top'] for monitor in monitors)
        max_x = max(monitor['left'] + monitor['width'] for monitor in monitors)
        max_y = max(monitor['top'] + monitor['height'] for monitor in monitors)
        
        total_width = max_x - min_x
        total_height = max_y - min_y
        
        # Create a combined image
        combined_image = Image.new('RGB', (total_width, total_height), 'white')
        
        # Capture each monitor and paste it in the correct position
        for i, monitor in enumerate(monitors):
            screenshot = sct.grab(monitor)
            monitor_image = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
            
            # Calculate position relative to the combined image
            x = monitor['left'] - min_x
            y = monitor['top'] - min_y
            
            # Paste the monitor image
            combined_image.paste(monitor_image, (x, y))
        
        self.full_screen_image = combined_image
        self.monitor_info = {
            'left': min_x,
            'top': min_y,
            'width': total_width,
            'height': total_height
        }
        
        # Store monitor information for coordinate mapping
        self.monitors = monitors
        self.combined_offset = (min_x, min_y)
    
    def create_selection_window(self):
        """Create a window for selecting area from the captured screen"""
        # Create new window
        self.selection_window = tk.Toplevel()
        self.selection_window.title("Select Area - Multi-Monitor Capture")
        self.selection_window.attributes('-topmost', True)
        
        # Get screen dimensions
        screen_width = self.selection_window.winfo_screenwidth()
        screen_height = self.selection_window.winfo_screenheight()
        
        # Resize full screen image to fit window
        display_width = min(screen_width - 100, 1400)
        display_height = min(screen_height - 100, 900)
        
        # Calculate aspect ratio
        img_width, img_height = self.full_screen_image.size
        ratio = min(display_width/img_width, display_height/img_height)
        
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        # Resize image for display
        self.display_image = self.full_screen_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.display_image)
        
        # Set window size
        self.selection_window.geometry(f"{new_width + 40}x{new_height + 100}")
        
        # Create canvas
        self.selection_canvas = tk.Canvas(self.selection_window, 
                                        width=new_width, height=new_height,
                                        bg='white')
        self.selection_canvas.pack(pady=10)
        
        # Display the screenshot
        self.selection_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        # Show monitor information
        if hasattr(self, 'monitors') and len(self.monitors) > 1:
            monitor_info = f"Captured {len(self.monitors)} monitors. Total size: {img_width}x{img_height}"
        else:
            monitor_info = f"Captured single monitor. Size: {img_width}x{img_height}"
        
        info_label = ttk.Label(self.selection_window, text=monitor_info, font=("Arial", 10))
        info_label.pack(pady=2)
        
        # Instructions
        instructions = ttk.Label(self.selection_window, 
                               text="Click and drag to select area, then press Enter to capture")
        instructions.pack(pady=5)
        
        # Bind mouse events
        self.selection_canvas.bind('<Button-1>', self.on_selection_mouse_down)
        self.selection_canvas.bind('<B1-Motion>', self.on_selection_mouse_drag)
        self.selection_canvas.bind('<ButtonRelease-1>', self.on_selection_mouse_up)
        self.selection_canvas.bind('<Key>', self.on_selection_key_press)
        self.selection_canvas.focus_set()
        
        # Bind escape key to cancel
        self.selection_window.bind('<Escape>', lambda e: self.cancel_capture())
        
        self.status_label.config(text="Select area on the screenshot", foreground="blue")
        
    def on_selection_mouse_down(self, event):
        """Handle mouse button press on selection window"""
        self.selection_start = (event.x, event.y)
        self.selection_rect = None
        
    def on_selection_mouse_drag(self, event):
        """Handle mouse drag on selection window"""
        if self.selection_start:
            # Remove previous rectangle
            if self.selection_rect:
                self.selection_canvas.delete(self.selection_rect)
            
            # Draw new rectangle
            x1, y1 = self.selection_start
            x2, y2 = event.x, event.y
            self.selection_rect = self.selection_canvas.create_rectangle(
                x1, y1, x2, y2, outline='red', width=3, fill='', stipple='gray50'
            )
            
    def on_selection_mouse_up(self, event):
        """Handle mouse button release on selection window"""
        if self.selection_start:
            self.selection_end = (event.x, event.y)
            # Ensure coordinates are in correct order
            x1, y1 = self.selection_start
            x2, y2 = self.selection_end
            self.selection_start = (min(x1, x2), min(y1, y2))
            self.selection_end = (max(x1, x2), max(y1, y2))
            
    def on_selection_key_press(self, event):
        """Handle key press on selection window"""
        if event.keysym == 'Return' and self.selection_start and self.selection_end:
            self.crop_selected_area()
        elif event.keysym == 'Escape':
            self.cancel_capture()
            
    def crop_selected_area(self):
        """Crop the selected area from the full screen image"""
        try:
            # Close selection window
            self.selection_window.destroy()
            self.selection_window = None
            
            # Calculate the scale factor between display and original image
            display_width, display_height = self.display_image.size
            original_width, original_height = self.full_screen_image.size
            
            scale_x = original_width / display_width
            scale_y = original_height / display_height
            
            # Convert display coordinates to original image coordinates
            x1, y1 = self.selection_start
            x2, y2 = self.selection_end
            
            orig_x1 = int(x1 * scale_x)
            orig_y1 = int(y1 * scale_y)
            orig_x2 = int(x2 * scale_x)
            orig_y2 = int(y2 * scale_y)
            
            # Ensure minimum size
            if orig_x2 - orig_x1 < 10 or orig_y2 - orig_y1 < 10:
                messagebox.showwarning("Warning", "Selection too small! Please select a larger area.")
                self.cancel_capture()
                return
            
            # Crop the image
            self.captured_image = self.full_screen_image.crop((orig_x1, orig_y1, orig_x2, orig_y2))
            
            # Display the captured image
            self.display_captured_image()
            
            # Update UI
            self.send_btn.config(state='normal')
            self.capture_btn.config(state='normal')  # Re-enable capture button
            self.status_label.config(text="Image captured successfully! Ready for another capture.", foreground="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to crop image: {str(e)}")
            self.cancel_capture()
            
    def display_captured_image(self):
        """Display the captured image in the canvas"""
        if self.captured_image:
            # Clear canvas
            self.image_canvas.delete("all")
            
            # Resize image to fit canvas
            canvas_width = 400
            canvas_height = 300
            
            # Calculate aspect ratio
            img_width, img_height = self.captured_image.size
            ratio = min(canvas_width/img_width, canvas_height/img_height)
            
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            # Resize image
            resized_image = self.captured_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(resized_image)
            
            # Display image
            self.image_canvas.create_image(canvas_width//2, canvas_height//2, image=photo, anchor=tk.CENTER)
            self.image_canvas.image = photo  # Keep a reference
            
    def cancel_capture(self):
        """Cancel the capture process"""
        if hasattr(self, 'selection_window') and self.selection_window:
            self.selection_window.destroy()
            self.selection_window = None
        
        if hasattr(self, 'capture_window') and self.capture_window:
            self.capture_window.destroy()
            self.capture_window = None
        
        self.is_capturing = False
        self.capture_btn.config(state='normal')  # Always re-enable capture button
        self.status_label.config(text="Capture cancelled. Ready for new capture.", foreground="red")
        
    def send_to_ai(self):
        """Send the captured image and prompt to OpenAI"""
        if not self.captured_image:
            messagebox.showwarning("Warning", "No image captured!")
            return
            
        prompt = self.prompt_entry.get().strip()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a prompt!")
            return
            
        # Disable send button and show status
        self.send_btn.config(state='disabled')
        self.status_label.config(text="Sending to AI...", foreground="orange")
        
        # Run AI request in separate thread
        thread = threading.Thread(target=self.process_ai_request, args=(prompt,))
        thread.daemon = True
        thread.start()
        
    def process_ai_request(self, prompt):
        """Process the AI request in a separate thread"""
        try:
            # Convert image to base64
            img_buffer = io.BytesIO()
            self.captured_image.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            # Prepare the message
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ]
            
            # Send to OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=1000
            )
            
            # Get response
            ai_response = response.choices[0].message.content
            
            # Update GUI on main thread
            self.root.after(0, lambda: self.display_ai_response(ai_response))
            
        except Exception as e:
            error_msg = f"Error communicating with AI: {str(e)}"
            self.root.after(0, lambda: self.display_ai_response(f"ERROR: {error_msg}"))
            
    def display_ai_response(self, response):
        """Display the AI response in the chat area"""
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        
        # Format the response
        formatted_response = f"[{timestamp}] AI Response:\n{response}\n\n"
        
        # Add to chat
        self.chat_text.insert(tk.END, formatted_response)
        self.chat_text.see(tk.END)
        
        # Re-enable send button and update status
        self.send_btn.config(state='normal')
        self.status_label.config(text="Response received. Ready for new capture.", foreground="green")

def main():
    root = tk.Tk()
    app = ScreenCaptureApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 