import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import time
import os
from PIL import Image, ImageTk
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DDPMPipeline, DDIMPipeline
from transformers import pipeline
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class GenAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GenAI Image Generation Tool")
        self.root.geometry("1200x800")
        
        # Generation state
        self.is_generating = False
        self.current_model = None
        self.generated_images = []
        
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
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel - Controls
        self.setup_control_panel(main_frame)
        
        # Right panel - Display and Gallery
        self.setup_display_panel(main_frame)
        
    def setup_control_panel(self, parent):
        """Setup the left control panel"""
        control_frame = ttk.LabelFrame(parent, text="Generation Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Model selection
        ttk.Label(control_frame, text="Model:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.model_var = tk.StringVar(value="Stable Diffusion")
        model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, 
                                  values=["Stable Diffusion", "DALL-E Mini", "Custom Model"], 
                                  state="readonly", width=25)
        model_combo.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Prompt input
        ttk.Label(control_frame, text="Prompt:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        self.prompt_entry = scrolledtext.ScrolledText(control_frame, height=4, width=30)
        self.prompt_entry.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Example prompts
        example_frame = ttk.LabelFrame(control_frame, text="Example Prompts", padding="5")
        example_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        examples = [
            "A beautiful sunset over mountains",
            "A futuristic city with flying cars",
            "A cute cat wearing a space helmet",
            "Abstract art with vibrant colors",
            "A steampunk robot in a Victorian setting"
        ]
        
        for i, example in enumerate(examples):
            btn = ttk.Button(example_frame, text=example, 
                           command=lambda e=example: self.set_prompt(e))
            btn.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=1)
        
        # Generation parameters
        params_frame = ttk.LabelFrame(control_frame, text="Parameters", padding="5")
        params_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Number of images
        ttk.Label(params_frame, text="Number of images:").grid(row=0, column=0, sticky=tk.W)
        self.num_images_var = tk.IntVar(value=1)
        num_spinbox = ttk.Spinbox(params_frame, from_=1, to=4, textvariable=self.num_images_var, width=10)
        num_spinbox.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        # Image size
        ttk.Label(params_frame, text="Image size:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.size_var = tk.StringVar(value="512x512")
        size_combo = ttk.Combobox(params_frame, textvariable=self.size_var,
                                 values=["256x256", "512x512", "768x768", "1024x1024"],
                                 state="readonly", width=10)
        size_combo.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=(5, 0))
        
        # Steps
        ttk.Label(params_frame, text="Steps:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.steps_var = tk.IntVar(value=20)
        steps_spinbox = ttk.Spinbox(params_frame, from_=10, to=100, textvariable=self.steps_var, width=10)
        steps_spinbox.grid(row=2, column=1, sticky=tk.W, padx=(5, 0), pady=(5, 0))
        
        # Guidance scale
        ttk.Label(params_frame, text="Guidance scale:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        self.guidance_var = tk.DoubleVar(value=7.5)
        guidance_scale = ttk.Scale(params_frame, from_=1.0, to=20.0, variable=self.guidance_var, 
                                  orient=tk.HORIZONTAL, length=150)
        guidance_scale.grid(row=3, column=1, sticky=tk.W, padx=(5, 0), pady=(5, 0))
        
        # Generate button
        self.generate_btn = ttk.Button(control_frame, text="🎨 Generate Images", 
                                     command=self.start_generation, style="Accent.TButton")
        self.generate_btn.grid(row=6, column=0, pady=(10, 5), sticky=(tk.W, tk.E))
        
        # Stop button
        self.stop_btn = ttk.Button(control_frame, text="⏹️ Stop Generation", 
                                 command=self.stop_generation, state='disabled')
        self.stop_btn.grid(row=7, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Status
        self.status_label = ttk.Label(control_frame, text="Ready to generate", 
                                     foreground="green")
        self.status_label.grid(row=8, column=0, pady=(10, 0), sticky=(tk.W, tk.E))
        
    def setup_display_panel(self, parent):
        """Setup the right display panel"""
        display_frame = ttk.Frame(parent)
        display_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(1, weight=1)
        
        # Gallery frame
        gallery_frame = ttk.LabelFrame(display_frame, text="Generated Images", padding="10")
        gallery_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        gallery_frame.columnconfigure(0, weight=1)
        
        # Create canvas for image gallery
        self.gallery_canvas = tk.Canvas(gallery_frame, width=600, height=400, bg='white')
        self.gallery_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Scrollbar for gallery
        gallery_scrollbar = ttk.Scrollbar(gallery_frame, orient="vertical", command=self.gallery_canvas.yview)
        gallery_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.gallery_canvas.configure(yscrollcommand=gallery_scrollbar.set)
        
        # Gallery controls
        controls_frame = ttk.Frame(gallery_frame)
        controls_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(controls_frame, text="💾 Save All", command=self.save_all_images).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(controls_frame, text="🗑️ Clear Gallery", command=self.clear_gallery).grid(row=0, column=1, padx=5)
        ttk.Button(controls_frame, text="📊 Show Stats", command=self.show_stats).grid(row=0, column=2, padx=5)
        
        # Progress and logs
        log_frame = ttk.LabelFrame(display_frame, text="Generation Log", padding="10")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, width=60)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Show placeholder
        self.show_placeholder()
        
    def set_prompt(self, prompt):
        """Set the prompt text"""
        self.prompt_entry.delete(1.0, tk.END)
        self.prompt_entry.insert(1.0, prompt)
        
    def show_placeholder(self):
        """Show placeholder in gallery"""
        self.gallery_canvas.delete("all")
        self.gallery_canvas.create_text(300, 200, text="No images generated yet", 
                                      fill="gray", font=("Arial", 16))
        
    def start_generation(self):
        """Start the image generation process"""
        prompt = self.prompt_entry.get(1.0, tk.END).strip()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter a prompt!")
            return
            
        # Disable generate button and enable stop button
        self.generate_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.is_generating = True
        
        # Start generation in separate thread
        thread = threading.Thread(target=self.generate_images, args=(prompt,))
        thread.daemon = True
        thread.start()
        
    def generate_images(self, prompt):
        """Generate images using the selected model"""
        try:
            self.update_status("Initializing model...", "orange")
            self.log_message("Starting image generation...")
            
            # Get parameters
            num_images = self.num_images_var.get()
            size = self.size_var.get()
            steps = self.steps_var.get()
            guidance_scale = self.guidance_var.get()
            
            # Parse size
            width, height = map(int, size.split('x'))
            
            # Load model based on selection
            model_name = self.model_var.get()
            if model_name == "Stable Diffusion":
                self.load_stable_diffusion_model()
            elif model_name == "DALL-E Mini":
                self.load_dalle_mini_model()
            else:
                self.update_status("Custom models not implemented yet", "red")
                return
                
            if not self.current_model:
                self.update_status("Failed to load model", "red")
                return
                
            self.update_status("Generating images...", "blue")
            self.log_message(f"Generating {num_images} image(s) with prompt: '{prompt}'")
            self.log_message(f"Parameters: {size}, {steps} steps, guidance={guidance_scale}")
            
            # Generate images
            images = []
            for i in range(num_images):
                if not self.is_generating:
                    break
                    
                self.log_message(f"Generating image {i+1}/{num_images}...")
                
                if model_name == "Stable Diffusion":
                    image = self.generate_stable_diffusion(prompt, width, height, steps, guidance_scale)
                elif model_name == "DALL-E Mini":
                    image = self.generate_dalle_mini(prompt)
                    
                if image:
                    images.append(image)
                    self.log_message(f"Image {i+1} generated successfully")
                else:
                    self.log_message(f"Failed to generate image {i+1}")
                    
            if images:
                self.generated_images.extend(images)
                self.display_images(images)
                self.update_status(f"Generated {len(images)} image(s) successfully!", "green")
                self.log_message(f"Generation completed! Total images: {len(self.generated_images)}")
            else:
                self.update_status("No images were generated", "red")
                
        except Exception as e:
            error_msg = f"Error during generation: {str(e)}"
            self.log_message(error_msg)
            self.update_status("Generation failed", "red")
            messagebox.showerror("Error", error_msg)
        finally:
            self.is_generating = False
            self.generate_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            
    def load_stable_diffusion_model(self):
        """Load Stable Diffusion model"""
        try:
            self.log_message("Loading Stable Diffusion model...")
            self.current_model = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if torch.cuda.is_available():
                self.current_model = self.current_model.to("cuda")
            self.log_message("Stable Diffusion model loaded successfully")
        except Exception as e:
            self.log_message(f"Failed to load Stable Diffusion model: {str(e)}")
            self.current_model = None
            
    def load_dalle_mini_model(self):
        """Load DALL-E Mini model"""
        try:
            self.log_message("Loading DALL-E Mini model...")
            self.current_model = pipeline("text-to-image", model="dalle-mini/dalle-mini")
            self.log_message("DALL-E Mini model loaded successfully")
        except Exception as e:
            self.log_message(f"Failed to load DALL-E Mini model: {str(e)}")
            self.current_model = None
            
    def generate_stable_diffusion(self, prompt, width, height, steps, guidance_scale):
        """Generate image using Stable Diffusion"""
        try:
            with torch.no_grad():
                result = self.current_model(
                    prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=1
                )
                return result.images[0]
        except Exception as e:
            self.log_message(f"Stable Diffusion generation error: {str(e)}")
            return None
            
    def generate_dalle_mini(self, prompt):
        """Generate image using DALL-E Mini"""
        try:
            result = self.current_model(prompt)
            return result.images[0]
        except Exception as e:
            self.log_message(f"DALL-E Mini generation error: {str(e)}")
            return None
            
    def display_images(self, images):
        """Display generated images in the gallery"""
        # Clear existing images
        self.gallery_canvas.delete("all")
        
        if not images:
            self.show_placeholder()
            return
            
        # Calculate layout
        cols = 2
        rows = (len(images) + cols - 1) // cols
        img_width = 280
        img_height = 280
        padding = 20
        
        for i, image in enumerate(images):
            row = i // cols
            col = i % cols
            
            x = col * (img_width + padding) + padding
            y = row * (img_height + padding) + padding
            
            # Resize image for display
            display_image = image.resize((img_width, img_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(display_image)
            
            # Create image on canvas
            img_id = self.gallery_canvas.create_image(x + img_width//2, y + img_height//2, image=photo)
            self.gallery_canvas.image = photo  # Keep reference
            
            # Add border
            self.gallery_canvas.create_rectangle(x, y, x + img_width, y + img_height, 
                                               outline='gray', width=2)
            
            # Add image number
            self.gallery_canvas.create_text(x + 10, y + 10, text=f"#{len(self.generated_images) - len(images) + i + 1}", 
                                          fill='white', font=("Arial", 12, "bold"))
        
        # Update scroll region
        total_height = rows * (img_height + padding) + padding
        self.gallery_canvas.configure(scrollregion=(0, 0, 600, total_height))
        
    def stop_generation(self):
        """Stop the generation process"""
        self.is_generating = False
        self.log_message("Generation stopped by user")
        self.update_status("Generation stopped", "orange")
        
    def save_all_images(self):
        """Save all generated images"""
        if not self.generated_images:
            messagebox.showwarning("Warning", "No images to save!")
            return
            
        # Ask for directory
        directory = filedialog.askdirectory(title="Select directory to save images")
        if not directory:
            return
            
        try:
            for i, image in enumerate(self.generated_images):
                filename = f"generated_image_{i+1:03d}.png"
                filepath = os.path.join(directory, filename)
                image.save(filepath)
                
            self.log_message(f"Saved {len(self.generated_images)} images to {directory}")
            messagebox.showinfo("Success", f"Saved {len(self.generated_images)} images successfully!")
        except Exception as e:
            error_msg = f"Failed to save images: {str(e)}"
            self.log_message(error_msg)
            messagebox.showerror("Error", error_msg)
            
    def clear_gallery(self):
        """Clear all generated images"""
        if self.generated_images:
            result = messagebox.askyesno("Confirm", "Are you sure you want to clear all images?")
            if result:
                self.generated_images.clear()
                self.show_placeholder()
                self.log_message("Gallery cleared")
                
    def show_stats(self):
        """Show generation statistics"""
        if not self.generated_images:
            messagebox.showinfo("Statistics", "No images generated yet")
            return
            
        stats = f"Total images generated: {len(self.generated_images)}\n"
        stats += f"Current model: {self.model_var.get()}\n"
        stats += f"Average image size: {self.generated_images[0].size if self.generated_images else 'N/A'}\n"
        
        messagebox.showinfo("Generation Statistics", stats)
        
    def update_status(self, message, color="black"):
        """Update status label"""
        self.status_label.config(text=message, foreground=color)
        
    def log_message(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, formatted_message)
        self.log_text.see(tk.END)
        self.root.update_idletasks()

def main():
    root = tk.Tk()
    app = GenAIApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
