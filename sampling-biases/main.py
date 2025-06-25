import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from data_handler import MNISTDataHandler
from trainer import ModelTrainer

class MNISTLearningTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Sampling Biases Learning Tool")
        self.root.geometry("1400x900")
        
        # Initialize data handler and trainer
        self.data_handler = None
        self.trainer = None
        self.sample_images = {}
        self.selected_classes = set()
        self.class_sliders = {}
        self.confusion_matrix = None
        
        # Initialize GUI
        self.setup_gui()
        self.load_data()
        
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
        
        # Left panel - Class selection
        self.setup_class_selection_panel(main_frame)
        
        # Right panel - Tabbed interface
        self.setup_tabbed_panel(main_frame)
        
    def setup_class_selection_panel(self, parent):
        """Setup the left panel with class selection"""
        left_frame = ttk.LabelFrame(parent, text="MNIST Digit Classes", padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Instructions
        instructions = ttk.Label(left_frame, 
                               text="Click on digit classes to include them in your dataset.\n"
                                    "Selected classes will appear in the control panel.",
                               wraplength=300)
        instructions.grid(row=0, column=0, columnspan=2, pady=(0, 10))
        
        # Class selection grid
        self.class_frame = ttk.Frame(left_frame)
        self.class_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.class_buttons = {}
        
    def setup_tabbed_panel(self, parent):
        """Setup the right panel with tabbed interface"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Setup tabs
        self.setup_control_tab()
        self.setup_results_tab()
        
    def setup_control_tab(self):
        """Setup the control tab with dataset configuration and training controls"""
        control_frame = ttk.Frame(self.notebook)
        self.notebook.add(control_frame, text="Controls")
        control_frame.columnconfigure(0, weight=1)
        control_frame.rowconfigure(0, weight=1)
        
        # Create a canvas with scrollbar for the entire control tab
        canvas = tk.Canvas(control_frame)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Grid the canvas and scrollbar
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        control_frame.columnconfigure(0, weight=1)
        control_frame.rowconfigure(0, weight=1)
        
        # Bind mouse wheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Selected classes frame with collapsible functionality
        self.setup_collapsible_dataset_frame(scrollable_frame)
        
        # Control buttons frame
        control_buttons_frame = ttk.LabelFrame(scrollable_frame, text="Training Controls", padding="10")
        control_buttons_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        control_buttons_frame.columnconfigure(0, weight=1)
        
        # Buttons
        button_frame = ttk.Frame(control_buttons_frame)
        button_frame.grid(row=0, column=0, pady=(0, 10))
        
        self.create_dataset_btn = ttk.Button(button_frame, text="Create Dataset", 
                                           command=self.create_dataset, state='disabled')
        self.create_dataset_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.train_btn = ttk.Button(button_frame, text="Start Training", 
                                  command=self.start_training, state='disabled')
        self.train_btn.grid(row=0, column=1, padx=5)
        
        self.test_btn = ttk.Button(button_frame, text="Test Model", 
                                 command=self.test_model, state='disabled')
        self.test_btn.grid(row=0, column=2, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop Training", 
                                 command=self.stop_training, state='disabled')
        self.stop_btn.grid(row=0, column=3, padx=(5, 0))
        
        # Progress and results
        self.progress_text = scrolledtext.ScrolledText(control_buttons_frame, height=15, width=60)
        self.progress_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        control_buttons_frame.rowconfigure(1, weight=1)
        
    def setup_collapsible_dataset_frame(self, parent):
        """Setup collapsible dataset configuration frame"""
        # Main frame
        self.dataset_frame = ttk.LabelFrame(parent, text="Selected Classes & Dataset Distribution", padding="10")
        self.dataset_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.dataset_frame.columnconfigure(0, weight=1)
        
        # Header frame with toggle button
        header_frame = ttk.Frame(self.dataset_frame)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        header_frame.columnconfigure(1, weight=1)
        
        # Toggle button
        self.toggle_btn = ttk.Button(header_frame, text="▼ Collapse", 
                                   command=self.toggle_dataset_frame, width=15)
        self.toggle_btn.grid(row=0, column=0, sticky=tk.W)
        
        # Add legend for slider colors
        legend_frame = ttk.Frame(header_frame)
        legend_frame.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        ttk.Label(legend_frame, text="Data Split Legend:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(legend_frame, text="🟦 Training", foreground="blue").grid(row=0, column=1, padx=(10, 5))
        ttk.Label(legend_frame, text="🟨 Validation", foreground="orange").grid(row=0, column=2, padx=(10, 5))
        ttk.Label(legend_frame, text="🟥 Test (auto)", foreground="red").grid(row=0, column=3, padx=(10, 5))
        
        # Content frame (initially visible)
        self.dataset_content_frame = ttk.Frame(self.dataset_frame)
        self.dataset_content_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.dataset_content_frame.columnconfigure(0, weight=1)
        
        # Store reference to the content frame for sliders
        self.selected_frame = self.dataset_content_frame
        
        # Track collapse state
        self.dataset_frame_collapsed = False
        
    def toggle_dataset_frame(self):
        """Toggle the visibility of the dataset configuration frame"""
        if self.dataset_frame_collapsed:
            # Expand
            self.dataset_content_frame.grid()
            self.toggle_btn.config(text="▼ Collapse")
            self.dataset_frame_collapsed = False
        else:
            # Collapse
            self.dataset_content_frame.grid_remove()
            self.toggle_btn.config(text="▶ Expand")
            self.dataset_frame_collapsed = True
        
    def setup_results_tab(self):
        """Setup the results tab with confusion matrix visualization"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(results_frame, text="Confusion Matrix Analysis", 
                               font=("Arial", 14, "bold"))
        title_label.grid(row=0, column=0, pady=(10, 5))
        
        # Instructions
        instructions = ttk.Label(results_frame, 
                               text="Run a test to see the confusion matrix heatmap and detailed analysis.",
                               font=("Arial", 10))
        instructions.grid(row=1, column=0, pady=(0, 10))
        
        # Create matplotlib figure for confusion matrix
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, results_frame)
        self.canvas.get_tk_widget().grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Results text area
        self.results_text = scrolledtext.ScrolledText(results_frame, height=10, width=80)
        self.results_text.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        results_frame.rowconfigure(3, weight=1)
        
        # Show placeholder
        self.show_placeholder_plot()
        
    def show_placeholder_plot(self):
        """Show placeholder plot when no results are available"""
        self.ax.clear()
        self.ax.text(0.5, 0.5, 'Run a test to see the confusion matrix', 
                    ha='center', va='center', transform=self.ax.transAxes,
                    fontsize=14, color='gray')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.canvas.draw()
        
    def plot_confusion_matrix(self, confusion_matrix):
        """Plot confusion matrix as heatmap"""
        self.ax.clear()
        
        # Create heatmap
        im = self.ax.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')
        
        # Add colorbar
        cbar = self.fig.colorbar(im, ax=self.ax)
        cbar.set_label('Number of Predictions', rotation=270, labelpad=15)
        
        # Set labels
        self.ax.set_xlabel('Predicted Label')
        self.ax.set_ylabel('True Label')
        self.ax.set_title('Confusion Matrix Heatmap')
        
        # Set tick labels
        self.ax.set_xticks(range(10))
        self.ax.set_yticks(range(10))
        self.ax.set_xticklabels(range(10))
        self.ax.set_yticklabels(range(10))
        
        # Add text annotations
        for i in range(10):
            for j in range(10):
                value = confusion_matrix[i, j]
                if value > 0:
                    # Use white text for dark backgrounds, black for light
                    color = 'white' if value > confusion_matrix.max() / 2 else 'black'
                    self.ax.text(j, i, str(value), ha='center', va='center', 
                               color=color, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.setp(self.ax.get_xticklabels(), rotation=0)
        
        self.canvas.draw()
        
    def load_data(self):
        """Load MNIST data in a separate thread"""
        def load_thread():
            self.update_progress("Loading MNIST dataset...")
            self.data_handler = MNISTDataHandler()
            self.sample_images = self.data_handler.get_sample_images(3)
            
            # Update GUI on main thread
            self.root.after(0, self.create_class_buttons)
            self.root.after(0, lambda: self.update_progress("Dataset loaded! Click digit classes to select them."))
        
        thread = threading.Thread(target=load_thread)
        thread.daemon = True
        thread.start()
        
    def create_class_buttons(self):
        """Create buttons for each digit class with sample images"""
        for class_label in range(10):
            frame = ttk.Frame(self.class_frame)
            frame.grid(row=class_label // 2, column=class_label % 2, padx=5, pady=5, sticky=(tk.W, tk.E))
            
            # Create a composite image showing samples
            images = self.sample_images[class_label]
            composite = np.concatenate(images, axis=1)
            
            # Convert to PIL and resize
            pil_image = Image.fromarray((composite * 255).astype(np.uint8))
            pil_image = pil_image.resize((120, 40), Image.NEAREST)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Button
            btn = tk.Button(frame, image=photo, text=f"Digit {class_label}",
                          compound=tk.TOP, command=lambda c=class_label: self.toggle_class(c),
                          relief=tk.RAISED, bd=2)
            btn.image = photo  # Keep a reference
            btn.grid(row=0, column=0)
            
            self.class_buttons[class_label] = btn
            
    def toggle_class(self, class_label):
        """Toggle selection of a digit class"""
        if class_label in self.selected_classes:
            self.selected_classes.remove(class_label)
            self.class_buttons[class_label].config(relief=tk.RAISED, bg='SystemButtonFace')
            self.remove_class_sliders(class_label)
        else:
            self.selected_classes.add(class_label)
            self.class_buttons[class_label].config(relief=tk.SUNKEN, bg='lightblue')
            self.add_class_sliders(class_label)
        
        # Update create dataset button
        self.create_dataset_btn.config(state='normal' if self.selected_classes else 'disabled')
        
    def add_class_sliders(self, class_label):
        """Add sliders for controlling train/val split for a class"""
        frame = ttk.LabelFrame(self.selected_frame, text=f"Digit {class_label}", padding="5")
        frame.grid(row=len(self.class_sliders) + 1, column=0, sticky=(tk.W, tk.E), pady=2)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)
        
        max_samples = len(self.data_handler.train_indices_by_class[class_label])
        
        # Training slider
        train_frame = ttk.Frame(frame)
        train_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        train_frame.columnconfigure(1, weight=1)
        
        ttk.Label(train_frame, text="Training:", foreground="blue").grid(row=0, column=0, padx=(0, 10), sticky=tk.W)
        train_slider = ttk.Scale(train_frame, from_=10, to=max_samples-10, orient=tk.HORIZONTAL)
        train_slider.set(min(400, max_samples-50))
        train_slider.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        train_label = ttk.Label(train_frame, text=str(int(train_slider.get())), foreground="blue")
        train_label.grid(row=0, column=2, sticky=tk.W)
        
        # Validation slider
        val_frame = ttk.Frame(frame)
        val_frame.grid(row=0, column=2, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        val_frame.columnconfigure(1, weight=1)
        
        ttk.Label(val_frame, text="Validation:", foreground="orange").grid(row=0, column=0, padx=(0, 10), sticky=tk.W)
        val_slider = ttk.Scale(val_frame, from_=10, to=max_samples-10, orient=tk.HORIZONTAL)
        val_slider.set(min(100, max_samples-400))
        val_slider.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        val_label = ttk.Label(val_frame, text=str(int(val_slider.get())), foreground="orange")
        val_label.grid(row=0, column=2, sticky=tk.W)
        
        # Test samples (calculated automatically)
        test_frame = ttk.Frame(frame)
        test_frame.grid(row=1, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(5, 0))
        
        test_label = ttk.Label(test_frame, text="Test: 0 samples", foreground="red")
        test_label.grid(row=0, column=0, sticky=tk.W)
        
        # Total samples label
        total_label = ttk.Label(test_frame, text=f"Total: {max_samples} samples available")
        total_label.grid(row=0, column=1, sticky=tk.E)
        
        # Constraint validation
        def update_splits():
            train_val = int(train_slider.get())
            val_val = int(val_slider.get())
            test_val = max_samples - train_val - val_val
            
            # Update labels
            train_label.config(text=str(train_val))
            val_label.config(text=str(val_val))
            test_label.config(text=f"Test: {test_val} samples")
            
            # Color coding for constraint validation
            if test_val < 0:
                test_label.config(foreground="red", text=f"Test: {test_val} samples (INVALID)")
                total_label.config(foreground="red")
            elif test_val < 10:
                test_label.config(foreground="orange", text=f"Test: {test_val} samples (SMALL)")
                total_label.config(foreground="black")
            else:
                test_label.config(foreground="red", text=f"Test: {test_val} samples")
                total_label.config(foreground="black")
        
        # Update validation slider range based on training slider
        def update_val_range():
            train_val = int(train_slider.get())
            max_val = max_samples - train_val - 10  # Leave at least 10 for test
            val_slider.config(to=max_val)
            if int(val_slider.get()) > max_val:
                val_slider.set(max_val)
            update_splits()
        
        # Update training slider range based on validation slider
        def update_train_range():
            val_val = int(val_slider.get())
            max_train = max_samples - val_val - 10  # Leave at least 10 for test
            train_slider.config(to=max_train)
            if int(train_slider.get()) > max_train:
                train_slider.set(max_train)
            update_splits()
        
        # Connect sliders
        def on_train_change(val):
            update_val_range()
        
        def on_val_change(val):
            update_train_range()
        
        train_slider.config(command=on_train_change)
        val_slider.config(command=on_val_change)
        
        # Initialize
        update_splits()
        
        self.class_sliders[class_label] = {
            'frame': frame,
            'train_slider': train_slider,
            'val_slider': val_slider,
            'train_label': train_label,
            'val_label': val_label,
            'test_label': test_label,
            'total_label': total_label,
            'max_samples': max_samples
        }
        
    def remove_class_sliders(self, class_label):
        """Remove sliders for a class"""
        if class_label in self.class_sliders:
            self.class_sliders[class_label]['frame'].destroy()
            del self.class_sliders[class_label]
            
            # Reposition remaining sliders
            for i, (label, widgets) in enumerate(self.class_sliders.items()):
                widgets['frame'].grid(row=i + 1, column=0, sticky=(tk.W, tk.E), pady=2)
    
    def create_dataset(self):
        """Create custom dataset based on user selections"""
        if not self.selected_classes:
            messagebox.showwarning("Warning", "Please select at least one digit class.")
            return
        
        # Validate that splits are valid
        for class_label in self.selected_classes:
            widgets = self.class_sliders[class_label]
            train_val = int(widgets['train_slider'].get())
            val_val = int(widgets['val_slider'].get())
            test_val = widgets['max_samples'] - train_val - val_val
            
            if test_val < 0:
                messagebox.showerror("Error", f"Digit {class_label}: Invalid split (test samples would be negative)")
                return
            elif test_val < 10:
                result = messagebox.askyesno("Warning", 
                    f"Digit {class_label}: Test set would only have {test_val} samples. Continue anyway?")
                if not result:
                    return
            
        selected_classes = list(self.selected_classes)
        train_samples = [int(self.class_sliders[c]['train_slider'].get()) for c in selected_classes]
        val_samples = [int(self.class_sliders[c]['val_slider'].get()) for c in selected_classes]
        test_samples = [self.class_sliders[c]['max_samples'] - train_samples[i] - val_samples[i] 
                       for i, c in enumerate(selected_classes)]
        
        def create_thread():
            self.update_progress("Creating custom dataset...")
            
            train_dataset, val_dataset, test_dataset = self.data_handler.create_custom_dataset_with_splits(
                selected_classes, train_samples, val_samples, test_samples
            )
            
            self.train_loader, self.val_loader, self.test_loader = self.data_handler.get_data_loaders(
                train_dataset, val_dataset, test_dataset
            )
            
            # Calculate dataset statistics
            train_size = len(train_dataset)
            val_size = len(val_dataset)
            test_size = len(test_dataset)
            
            stats = f"Dataset created successfully!\n"
            stats += f"Training samples: {train_size}\n"
            stats += f"Validation samples: {val_size}\n"
            stats += f"Test samples: {test_size}\n"
            stats += f"Selected classes: {sorted(selected_classes)}\n\n"
            
            # Show per-class breakdown
            stats += "Per-class breakdown:\n"
            for i, class_label in enumerate(selected_classes):
                stats += f"Digit {class_label}: Train={train_samples[i]}, Val={val_samples[i]}, Test={test_samples[i]}\n"
            
            self.root.after(0, lambda: self.update_progress(stats))
            self.root.after(0, lambda: self.train_btn.config(state='normal'))
            
        thread = threading.Thread(target=create_thread)
        thread.daemon = True
        thread.start()
        
    def start_training(self):
        """Start model training"""
        if not hasattr(self, 'train_loader'):
            messagebox.showwarning("Warning", "Please create a dataset first.")
            return
            
        self.trainer = ModelTrainer(progress_callback=self.update_progress)
        self.train_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.test_btn.config(state='disabled')
        
        # Start training
        training_thread = self.trainer.train_model(self.train_loader, self.val_loader, epochs=5)
        
        # Monitor training completion
        def monitor_training():
            training_thread.join()
            self.root.after(0, self.training_completed)
            
        monitor_thread = threading.Thread(target=monitor_training)
        monitor_thread.daemon = True
        monitor_thread.start()
        
    def training_completed(self):
        """Handle training completion"""
        self.train_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.test_btn.config(state='normal')
        
    def stop_training(self):
        """Stop model training"""
        if self.trainer:
            self.trainer.stop_training()
            self.update_progress("Training stopped by user.")
            self.training_completed()
            
    def test_model(self):
        """Test the trained model"""
        if not self.trainer:
            messagebox.showwarning("Warning", "Please train a model first.")
            return
            
        def test_thread():
            self.update_progress("Testing model...")
            
            test_loss, overall_accuracy, class_accuracies, class_totals, confusion_matrix = self.trainer.test(self.test_loader)
            
            # Store confusion matrix for plotting
            self.confusion_matrix = confusion_matrix
            
            results = f"\n=== TEST RESULTS ===\n"
            results += f"Overall Test Accuracy: {overall_accuracy:.2f}%\n"
            results += f"Test Loss: {test_loss:.4f}\n\n"
            results += "Per-class Results:\n"
            
            for class_label in sorted(class_accuracies.keys()):
                if class_totals[class_label] > 0:
                    results += f"Digit {class_label}: {class_accuracies[class_label]:.1f}% "
                    results += f"({class_totals[class_label]} samples)\n"
            
            results += "\nThis shows how sampling biases affect model performance!\n"
            results += "Notice how classes with fewer samples might have lower accuracy.\n"
            
            # Add confusion matrix analysis
            results += "\n" + "="*60 + "\n"
            results += "CONFUSION MATRIX ANALYSIS:\n"
            results += "="*60 + "\n"
            
            # Find most confused pairs
            max_confusion = 0
            most_confused_pair = (0, 0)
            for i in range(10):
                for j in range(10):
                    if i != j and confusion_matrix[i][j] > max_confusion:
                        max_confusion = confusion_matrix[i][j]
                        most_confused_pair = (i, j)
            
            if max_confusion > 0:
                results += f"Most confused: {most_confused_pair[0]} → {most_confused_pair[1]} ({max_confusion} times)\n"
            
            # Calculate precision and recall for each class
            results += "\nPer-class Metrics:\n"
            for i in range(10):
                if class_totals[i] > 0:
                    # True positives (diagonal)
                    tp = confusion_matrix[i][i]
                    # False positives (sum of column minus diagonal)
                    fp = sum(confusion_matrix[j][i] for j in range(10)) - tp
                    # False negatives (sum of row minus diagonal)
                    fn = sum(confusion_matrix[i][j] for j in range(10)) - tp
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    results += f"Digit {i}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}\n"
            
            # Update GUI on main thread
            self.root.after(0, lambda: self.update_progress(results))
            self.root.after(0, lambda: self.update_results_tab(results))
            self.root.after(0, lambda: self.plot_confusion_matrix(confusion_matrix))
            self.root.after(0, lambda: self.notebook.select(1))  # Switch to results tab
            
        thread = threading.Thread(target=test_thread)
        thread.daemon = True
        thread.start()
        
    def update_results_tab(self, results):
        """Update the results tab with detailed analysis"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, results)
        
    def update_progress(self, message):
        """Update progress text"""
        self.progress_text.insert(tk.END, message + "\n")
        self.progress_text.see(tk.END)
        self.root.update_idletasks()

def main():
    root = tk.Tk()
    app = MNISTLearningTool(root)
    root.mainloop()

if __name__ == "__main__":
    main() 