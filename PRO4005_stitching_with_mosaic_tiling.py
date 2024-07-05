import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import glob
import re
from stitch2d import Mosaic
import json
import sys
from tooltip import ToolTip


class Atlas:
    def __init__(self, window):
        self.window = window
        self.window.title("Atlas 2.0")
        self.window.configure(bg="#202020")  # Set background color     

        # Create main frames for Capturing and Stitching sections
        self.left_frame = tk.Frame(window, bg="#202020")
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.right_frame = tk.Frame(window, bg="#202020")
        self.right_frame.grid(row=0, column=2, padx=(30, 30), pady=10, sticky="nsew")

        self.separator = tk.Frame(window, bg="gray", width=2)
        self.separator.grid(row=0, column=1, sticky="ns")

        # Add headers
        self.left_header = tk.Label(self.left_frame, text="Capturing", bg="#202020", fg="white", font=("Arial", 14, "bold"))
        self.left_header.pack(pady=(0, 10))

        self.right_header = tk.Label(self.right_frame, text="Stitching", bg="#202020", fg="white", font=("Arial", 14, "bold"))
        self.right_header.pack(pady=(0, 10))

        # Add widgets to the left frame (Capturing section)
        self.camera_label = tk.Label(self.left_frame, text="Select Camera Device:", bg="#202020", fg="white", font=("Arial", 12))
        self.camera_label.pack(pady=(10, 0))

        self.camera_var = tk.StringVar(value="0")  # Set default camera device to 0
        self.camera_devices = self.get_camera_devices()
        self.camera_dropdown = tk.OptionMenu(self.left_frame, self.camera_var, *self.camera_devices)
        self.camera_dropdown.config(font=("Arial", 12))
        self.camera_dropdown["menu"].config(font=("Arial", 12))
        self.camera_dropdown.pack()

        # Create resolution dropdown with default value
        self.resolution_label = tk.Label(self.left_frame, text="Select Resolution:", bg="#202020", fg="white", font=("Arial", 12))
        self.resolution_label.pack(pady=(10, 0))

        # For the Hamamatsu C2400-d3 camera, these are the possible resolutions:
        self.resolution_options = [("160x120"), 
                                   ("320x240"),
                                   ("640x480"),
                                   ("720x480"),
                                   ("720x576")]
        
        self.resolution_var = tk.StringVar(value="720x480")  # Default resolution
        self.resolution_dropdown = tk.OptionMenu(self.left_frame, self.resolution_var, *self.resolution_options)
        self.resolution_dropdown.config(font=("Arial", 12))
        self.resolution_dropdown["menu"].config(font=("Arial", 12))
        self.resolution_dropdown.pack()

        # Create image format info label
        self.format_label = tk.Label(self.left_frame, text="Image Format: .tiff", bg="#202020", fg="white", font=("Arial", 12))
        self.format_label.pack(pady=(10, 0))

        self.format_var = tk.StringVar(value="tiff")  # Default format is tiff
        
        # Create start button
        self.start_button = tk.Button(self.left_frame, text="Start Camera", command=self.start_video_feed, font=("Arial", 12))
        self.start_button.pack(pady=(10, 0), padx=(20,20))

        # Create capture photo button
        self.capture_button = tk.Button(self.left_frame, text="Capture Photo (p)", command=self.capture_photo, font=("Arial", 12))
        self.capture_button.pack(pady=(10, 0), padx=(20,20))

        # Create capture new series button
        self.new_series_button = tk.Button(self.left_frame, text="Capture New Image Series", command=self.start_new_series, font=("Arial", 12))
        self.new_series_button.pack(pady=(10, 10), padx=(40, 40))

        # Add widgets to the right frame (Stitching section)

        # Create Gamma label
        self.gamma_label = tk.Label(self.right_frame, text="Gamma:", bg="#202020", fg="white", font=("Arial", 12))
        self.gamma_label.pack(pady=(10, 5))

        # Create Gamma entry box with default value
        self.gamma_entry_var = tk.DoubleVar(value=0.5)  # Default gamma value
        self.gamma_entry = tk.Entry(self.right_frame, textvariable=self.gamma_entry_var, font=("Arial", 12), width=4, justify='center')
        self.gamma_entry.pack(pady=(0, 10))

        # Create help icon
        self.help_icon = tk.Label(self.right_frame, text="?", bg="#202020", fg="white", font=("Arial", 12, "bold"))
        self.help_icon.pack(pady=(0, 10))

        # Add tooltip to the help icon
        self.tooltip = ToolTip(self.help_icon, text="Gamma correction adjusts the brightness and contrast of the image. Change it if the stitching fails for better feature detection.")

        # Add checkbox for Apply Normalization
        self.apply_normalization_var = tk.BooleanVar()  # Variable to store the state of the checkbox
        self.apply_normalization_checkbox = tk.Checkbutton(self.right_frame, text="Apply Normalization", variable=self.apply_normalization_var, bg="#202020", fg="white", font=("Arial", 12), selectcolor="#202020")
        self.apply_normalization_checkbox.pack(pady=(0, 10))

        # Create stitch acquired images button
        self.stitch_button = tk.Button(self.right_frame, text="Stitch Acquired Images", command=self.stitch_images, font=("Arial", 12))
        self.stitch_button.pack(pady=(10, 0), padx=(20,20))

        # Create stitch existing images button
        self.stitch_existing_button = tk.Button(self.right_frame, text="Stitch Existing Images", command=self.stitch_existing_images, font=("Arial", 12))
        self.stitch_existing_button.pack(pady=(10, 0), padx=(20,20))

        # Initialize video capture and variables
        self.video_capture = None
        self.is_video_running = False
        self.is_recording = False
        self.video_window = None
        self.video_frame = None
        self.video_writer = None

        # Initialize photo naming variables
        self.base_filename = None
        self.photo_count = 0

        # Initialize captured image display window
        self.captured_image_window = None
        self.captured_image_label = None
        self.captured_image_filename_label = None

        # Initialize list to store captured images
        self.captured_images = []

        # Bind the on_close method to the window's close event
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

        # Initialize save folder
        self.save_folder = None

    def on_close(self):
        # Release the video capture if it is open
        if self.video_capture is not None and self.video_capture.isOpened():
            self.video_capture.release()

        # Destroy the main window to close the application
        self.window.destroy()
        sys.exit()    

    def get_camera_devices(self):
        # Use OpenCV to get the available camera devices
        camera_devices = []
        for i in range(10):  # Assume a maximum of 10 camera devices
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                camera_devices.append(str(i))
                cap.release()
        return camera_devices

    def start_video_feed(self):
        camera_device = int(self.camera_var.get())
        resolution_str = self.resolution_var.get()
        width, height = resolution_str.split("x")
        width = int(width.strip())
        height = int(height.strip())

        self.video_capture = cv2.VideoCapture(camera_device, cv2.CAP_DSHOW)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Check the actual resolution set by the camera
        actual_width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
      
        self.is_video_running = True
        self.start_button.configure(state=tk.DISABLED)

        self.create_video_feed_window()

        self.update_video_feed()

    def create_video_feed_window(self):
        self.video_window = tk.Toplevel(self.window)
        self.video_window.title("Video Feed")
        self.video_window.configure(bg="#202020")

        # Create video frame
        self.video_frame = tk.Label(self.video_window)
        self.video_frame.pack(padx=10, pady=10)

        # Display requested and actual resolution
        requested_resolution = self.resolution_var.get()
        actual_width = self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        resolution_text = f"Requested Resolution: {requested_resolution}\nActual Resolution: {int(actual_width)}x{int(actual_height)}"
        resolution_label = tk.Label(self.video_window, text=resolution_text, bg="#202020", fg="white", font=("Arial", 12))
        resolution_label.pack(pady=(0, 10))

        # Bind the close event to stop the video feed
        self.video_window.protocol("WM_DELETE_WINDOW", self.stop_video_feed)

        # Bind key events for quitting, saving frame, and starting/stopping recording
        self.video_window.bind('q', self.quit_program)
        self.video_window.bind('p', self.save_frame)


    def update_video_feed(self):
        if self.is_video_running:
            ret, frame = self.video_capture.read()

            if ret:
                if self.is_recording and self.video_writer is not None:
                    self.video_writer.write(frame)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
                self.video_frame.configure(image=img)
                self.video_frame.imgtk = img

        if self.video_window is not None:
            self.video_window.after(10, self.update_video_feed)

    def stop_video_feed(self):
        self.is_video_running = False
        self.start_button.configure(state=tk.NORMAL)
        if self.video_capture is not None:
            self.video_capture.release()
        if self.video_writer is not None:
            self.video_writer.release()
        if self.video_window is not None:
            self.video_window.destroy()

    def quit_program(self, event=None):
        self.stop_video_feed()
        self.window.quit()

    def save_frame(self, event=None):
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if ret:
                format_choice = self.format_var.get()

                if self.base_filename is None:
                    # Prompt user to set base filename for the first photo
                    self.base_filename = filedialog.asksaveasfilename(defaultextension=f".{format_choice}",
                                                                    filetypes=[(f"{format_choice.upper()} files", f"*.{format_choice}")])
                    if not self.base_filename:
                        return
                    self.photo_count = 1
                    self.save_folder = os.path.dirname(os.path.abspath(self.base_filename))

                else:
                    self.photo_count += 1

                save_path = f"{os.path.splitext(self.base_filename)[0]}_{self.photo_count}.{format_choice}"


                if format_choice == "tiff":
                    # Correct color space conversion for TIFF
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    Image.fromarray(frame_rgb).save(save_path)
                elif format_choice == "jpg":
                    # Save as JPEG with correct BGR color order
                    cv2.imwrite(save_path, frame)

                # Show the captured image in a new window
                self.show_captured_image(save_path, frame)
                
                # Append the captured image to the list
                self.captured_images.append(frame)

    def capture_photo(self):
        self.capture_and_save_average_frame()

    def start_new_series(self):
        self.base_filename = None
        self.photo_count = 0
        self.capture_photo()

    def capture_and_save_average_frame(self):
        if self.video_capture is None:
            return

        # Capture 10 frames at 30 fps
        frames = []
        for _ in range(10):
            ret, frame = self.video_capture.read()
            if not ret:
                return
            frames.append(frame)
            cv2.waitKey(33)  # Wait approximately 33ms to achieve 30fps

        # Compute the average of the frames
        avg_frame = np.mean(frames, axis=0).astype(np.uint8)

        format_choice = self.format_var.get()

        if self.base_filename is None:
            # Prompt user to set base filename for the first photo
            self.base_filename = filedialog.asksaveasfilename(defaultextension=f".{format_choice}",
                                                            filetypes=[(f"{format_choice.upper()} files", f"*.{format_choice}")])
            if not self.base_filename:
                return
            self.photo_count = 1
        else:
            self.photo_count += 1

        save_path = f"{os.path.splitext(self.base_filename)[0]}_{self.photo_count}.{format_choice}"
       
        
        # Correct color space conversion for TIFF (if necessary)
        avg_frame_rgb = cv2.cvtColor(avg_frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(avg_frame_rgb).save(save_path)
        
        # Show the captured image in a new window
        self.show_captured_image(save_path, avg_frame)

        # Append the captured image to the list
        self.captured_images.append(avg_frame)

        # Ensure the video window stays in focus
        self.video_window.lift()
        self.video_window.focus_force()

    def show_captured_image(self, filename, image):
        if self.captured_image_window is None or not self.captured_image_window.winfo_exists():
            self.captured_image_window = tk.Toplevel(self.window)
            self.captured_image_window.title("Captured Image")
            self.captured_image_window.configure(bg="#202020")

            # Display filename label
            self.captured_image_filename_label = tk.Label(self.captured_image_window, text=filename, bg="#202020", fg="white", font=("Arial", 12))
            self.captured_image_filename_label.pack(pady=(10, 0))

            # Display image label
            self.captured_image_label = tk.Label(self.captured_image_window)
            self.captured_image_label.pack(padx=10, pady=10)
        else:
            self.captured_image_filename_label.configure(text=filename)

        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        self.captured_image_label.configure(image=img)
        self.captured_image_label.imgtk = img

        # Show or bring the captured image window to the front
        self.captured_image_window.lift()
        self.captured_image_window.focus_force()

        # Force the video window to come into focus
        if self.video_window:
            self.video_window.lift()
            self.video_window.focus_force()

    
    ## Stitching images

    def stitch_images(self):
        if len(self.captured_images) == 0:
            messagebox.showinfo("Stitch Images", "No images taken yet!")
            return
        
        ## Cropping images
   
        def load_images(folder_path):
            # Get list of image files
            image_files = glob.glob(os.path.join(folder_path, '*.tiff'))
            
            # Function to extract the numeric part of the filename
            def extract_number(filename):
                match = re.search(r'_(\d+)\.tiff$', filename)
                return int(match.group(1)) if match else -1
            
            # Sort files by the extracted number
            sorted_files = sorted(image_files, key=extract_number)
            
            # Load images in the sorted order
            images = [cv2.imread(image_file) for image_file in sorted_files]
            return images, sorted_files

        def crop_images(image_list):
            cropped_images = []
            
            for im in image_list:
                height, width, channels = im.shape
                crop_img = im[30:height-30, 30:width-30]  # Crop the image
                cropped_images.append(crop_img)  # Add cropped image to the new list
            
            return cropped_images
        
        def normalize_images(input_folder, output_folder):
            # Create output folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Get list of all files in the input folder
            image_files = os.listdir(input_folder)

            for image_file in image_files:
                # Read grayscale image
                image_path = os.path.join(input_folder, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if image is not None:
                    # Normalize image
                    normalized_image = cv2.normalize(
                        image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

                    # Save normalized image to output folder with "normalized_" prefix
                    normalized_filename = f"normalized_{image_file}"
                    output_path = os.path.join(output_folder, normalized_filename)
                    cv2.imwrite(output_path, normalized_image)

                    
                else:
                    print(f"Could not read image: {image_file}")
                
            print(f"{len(image_files)} images normalized and saved to {output_folder}")


        # Initialize tkinter and hide the root window
        root = tk.Tk()
        root.withdraw()

        # Open a dialog for folder selection
        folder_path = self.save_folder

        if folder_path:
            print("Cropping images")
            sys.stdout.flush()

            # Define output folder inside the selected folder
            output_folder_crop = os.path.join(folder_path, 'cropped_images')
            
            # Create output folder if it doesn't exist
            os.makedirs(output_folder_crop, exist_ok=True)
            
            # Load images
            images, sorted_files = load_images(folder_path)
            
            # Crop images
            cropped_images = crop_images(images)
            
            # Save cropped images
            for cropped_img, orig_file in zip(cropped_images, sorted_files):
                # Extract the number from the original filename
                number = re.search(r'_(\d+)\.tiff$', orig_file).group(1)
                output_path_crop = os.path.join(output_folder_crop, f"cropped_image_{number}.tiff")
                cv2.imwrite(output_path_crop, cropped_img)
            
            print(f"{len(cropped_images)} images cropped and saved to {output_folder_crop}")
            sys.stdout.flush()  # Flush stdout to ensure immediate display

            # Define output folder for normalized images (defined here to avoid UnboundLocalError)
            output_folder_normalized = os.path.join(folder_path, 'normalized_images')

            # Check if normalization is needed
            if self.apply_normalization_var.get():
                print("Normalizing images")
                sys.stdout.flush()

                # Define output folder for normalized images
                output_folder_normalized = os.path.join(folder_path, 'normalized_images')

                # Normalize images
                normalize_images(output_folder_crop, output_folder_normalized)

        else:
            print("No folder selected.")
            sys.stdout.flush()  # Flush stdout to ensure immediate display


        ## Enhancing images
        print("Enhancing images")
        sys.stdout.flush()

        def enhance_image_quality(image):

            # Enhance contrast using gamma correction on the grayscale image
            enhanced_image = adjust_contrast(image, gamma=self.gamma_entry_var.get())
            
            # Sharpen the image
            enhanced_image = sharpen_image(enhanced_image)
            
            return enhanced_image

        def adjust_contrast(image, gamma=self.gamma_entry_var.get()):
            # Apply gamma correction to enhance contrast (reversible)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            adjusted = cv2.LUT(image, table)
            return adjusted

        def sharpen_image(image):
            # Create a sharpening kernel
            kernel = np.array([[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])
            
            # Sharpen the image using filter2D convolution
            sharpened_image = cv2.filter2D(image, -1, kernel)
            
            return sharpened_image
        def create_alpha_mask(shape, edge_width):
            """Create an alpha mask with smooth edges for blending."""
            h, w = shape[:2]
            mask = np.zeros((h, w), dtype=np.float32)
            
            # Horizontal gradient
            for i in range(edge_width):
                alpha = i / edge_width
                mask[:, i] = alpha
                mask[:, w - i - 1] = alpha
            
            # Vertical gradient
            for j in range(edge_width):
                alpha = j / edge_width
                mask[j, :] = np.maximum(mask[j, :], alpha)
                mask[h - j - 1, :] = np.maximum(mask[h - j - 1, :], alpha)
            
            mask[edge_width:-edge_width, edge_width:-edge_width] = 1
            
            return mask

        def blend_images(base_img, new_img, mask, x, y):
            """Blend new_img into base_img at position (x, y) using mask."""
            roi = base_img[y:y + new_img.shape[0], x:x + new_img.shape[1]]
            
            # Convert mask to 3 channels
            alpha = mask[:, :, np.newaxis]
            
            # Blend images
            blended = roi * (1 - alpha) + new_img * alpha
            base_img[y:y + new_img.shape[0], x:x + new_img.shape[1]] = blended

        # Define folder paths
        if os.path.isdir(output_folder_normalized):
            folder_path_enh = output_folder_normalized
        else:
            folder_path_enh = output_folder_crop
        output_folder_enh = os.path.join(folder_path, 'enhanced_images')

        # Create output folder if it doesn't exist
        os.makedirs(output_folder_enh, exist_ok=True)

        # Get list of image files in the folder
        image_files = glob.glob(os.path.join(folder_path_enh, '*.tiff'))

        # Iterate through each image file
        for image_file in image_files:
            # Load image
            original_image = cv2.imread(image_file)
            
            if original_image is None:
                print(f"Error: Could not read image from {image_file}")
                continue
            
            # Enhance the image quality
            enhanced_image = enhance_image_quality(original_image)
            
            # Save the enhanced image
            filename = os.path.basename(image_file)
            output_path_enh = os.path.join(output_folder_enh, f"enhanced_{filename}")
            
            cv2.imwrite(output_path_enh, enhanced_image)
            
        print(f"{len(image_files)} images enhanced and saved to {output_folder_enh}")
        sys.stdout.flush()  # Flush stdout to ensure immediate display

        ## Creating mosaic

        print("Creating mosaic, this might take few minutes")
        sys.stdout.flush()

        # Function to extract numeric part from filename
        def extract_number(filename):
            return int(filename.split('_')[-1].split('.')[0])

        # Define the folder containing the images
        source_folder = output_folder_enh

        # Get list of image files in the source folder
        image_files = sorted([f for f in os.listdir(source_folder) if f.endswith('.tiff')])  # Adjust file extension as needed

        # Sort image files based on the extracted number
        image_files_sorted = sorted(image_files, key=lambda x: extract_number(x))

        # Create a Mosaic object
        mosaic = Mosaic(source_folder)

        # Align and smooth seams
        mosaic.align()
        mosaic.smooth_seams()

        # Show the stitched image (if applicable in your library)
        #mosaic.show()

        # Define the folder to save the JSON parameters file
        output_folder_mosaic = output_folder_enh

        # Save the parameters to a JSON file
        params_file = "params_mosaic.json"
        params_path = os.path.join(output_folder_mosaic, params_file)
        mosaic.save_params(params_path)

        # Verify the saved parameters file
        with open(params_path, 'r') as f:
            params_data = json.load(f)
            if 'filenames' in params_data:
                print(f"Parameters include {len(params_data['coords'])} image(s)")

        print(f"Parameters saved to: {params_path}")
        sys.stdout.flush()  # Flush stdout to ensure immediate display

        ## Stitch images

        print("Stitching images")
        sys.stdout.flush()

        # Load the JSON file
        json_path = os.path.join(output_folder_mosaic, 'params_mosaic.json')
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract metadata
        tile_shape = data['metadata']['tile_shape']
        tile_width, tile_height = tile_shape[1], tile_shape[0]

        # Determine the size of the canvas
        max_x = max([coord[1] for coord in data['coords'].values()]) + tile_width
        max_y = max([coord[0] for coord in data['coords'].values()]) + tile_height

        # Create a blank canvas (black background)
        canvas = np.zeros((int(max_y), int(max_x), 3), dtype=np.uint8)

        # Set edge width for blending
        edge_width = 30  # Adjust as needed

        # Place each image on the canvas with blending
        for key, coord in data['coords'].items():
            filename = data['filenames'][key]
            img_path = os.path.join(output_folder_mosaic, filename)
            img = cv2.imread(img_path)
            x, y = int(coord[1]), int(coord[0])  # Flipped coordinates

            # Create an alpha mask for the current image
            alpha_mask = create_alpha_mask(img.shape, edge_width)

            # Blend the current image onto the canvas
            blend_images(canvas, img, alpha_mask, x, y)

        # Resize the final stitched image for display
        resize_factor = 0.6  # Resize to 60% of the original size
        resized_canvas = cv2.resize(canvas, None, fx=resize_factor, fy=resize_factor)

        # Display the final stitched image in a resized window
        # cv2.namedWindow('Stitched Image', cv2.WINDOW_NORMAL)  # Create a resizable window
        # cv2.resizeWindow('Stitched Image', int(resized_canvas.shape[1]), int(resized_canvas.shape[0]))  # Resize the window
        # cv2.imshow('Stitched Image', resized_canvas)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Define folder to save stitched image in
        output_folder_stitch = os.path.join(folder_path, 'Stitching result')

        # Create folder if it doesn't exist
        os.makedirs(output_folder_stitch, exist_ok=True)

        #  Save the stitched image
        stitched_image_name = f'stitched_image.tiff'
        output_path = os.path.join(output_folder_stitch, stitched_image_name)
        cv2.imwrite(output_path, canvas)

        print(f"Stitched image saved to {output_folder_stitch}")
        sys.stdout.flush()  # Flush stdout to ensure immediate display      
                   

    def stitch_existing_images(self):
        
        ## Cropping images

        def load_images(folder_path):
            # Get list of image files
            image_files = glob.glob(os.path.join(folder_path, '*.tiff'))
            
            # Function to extract the numeric part of the filename
            def extract_number(filename):
                match = re.search(r'_(\d+)\.tiff$', filename)
                return int(match.group(1)) if match else -1
            
            # Sort files by the extracted number
            sorted_files = sorted(image_files, key=extract_number)
            
            # Load images in the sorted order
            images = [cv2.imread(image_file) for image_file in sorted_files]
            return images, sorted_files

        def crop_images(image_list):
            
            cropped_images = []
            
            for im in image_list:
                height, width, channels = im.shape
                crop_img = im[30:height-30, 30:width-30]  # Crop the image
                cropped_images.append(crop_img)  # Add cropped image to the new list
            
            return cropped_images
        
        def normalize_images(input_folder, output_folder):
            # Create output folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Get list of all files in the input folder
            image_files = os.listdir(input_folder)

            for image_file in image_files:
                # Read grayscale image
                image_path = os.path.join(input_folder, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                if image is not None:
                    # Normalize image
                    normalized_image = cv2.normalize(
                        image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

                    # Save normalized image to output folder with "normalized_" prefix
                    normalized_filename = f"normalized_{image_file}"
                    output_path = os.path.join(output_folder, normalized_filename)
                    cv2.imwrite(output_path, normalized_image)
                   
                else:
                    print(f"Could not read image: {image_file}")
        
            print(f"{len(image_files)} images normalized and saved to {output_folder}")

        # Initialize tkinter and hide the root window
        root = tk.Tk()
        root.withdraw()

        # Open a dialog for folder selection
        folder_path = filedialog.askdirectory(title="Select Folder Containing Images")

        if folder_path:

            print("Cropping images")
            sys.stdout.flush()

            # Define output folder inside the selected folder
            output_folder_crop = os.path.join(folder_path, 'cropped_images')
            
            # Create output folder if it doesn't exist
            os.makedirs(output_folder_crop, exist_ok=True)
            
            # Load images
            images, sorted_files = load_images(folder_path)
            
            # Crop images
            cropped_images = crop_images(images)
            
            # Save cropped images
            for cropped_img, orig_file in zip(cropped_images, sorted_files):
                # Extract the number from the original filename
                number = re.search(r'_(\d+)\.tiff$', orig_file).group(1)
                output_path_crop = os.path.join(output_folder_crop, f"cropped_image_{number}.tiff")
                cv2.imwrite(output_path_crop, cropped_img)
            
            print(f"{len(cropped_images)} images cropped and saved to {output_folder_crop}")
            sys.stdout.flush()  # Flush stdout to ensure immediate display

            # Define output folder for normalized images (defined here to avoid UnboundLocalError)
            output_folder_normalized = os.path.join(folder_path, 'normalized_images')

            # Check if normalization is needed
            if self.apply_normalization_var.get():
                print("Normalizing images")
                sys.stdout.flush()

                # Define output folder for normalized images
                output_folder_normalized = os.path.join(folder_path, 'normalized_images')

                # Normalize images
                normalize_images(output_folder_crop, output_folder_normalized)

        else:
            print("No folder selected.")
            sys.stdout.flush()  # Flush stdout to ensure immediate display


        ## Enhancing images

        print("Enhancing images")
        sys.stdout.flush()

        def enhance_image_quality(image):

            # Enhance contrast using gamma correction on the grayscale image
            enhanced_image = adjust_contrast(image, gamma=self.gamma_entry_var.get())
            
            # Sharpen the image
            enhanced_image = sharpen_image(enhanced_image)
            
            return enhanced_image

        def adjust_contrast(image, gamma=self.gamma_entry_var.get()):
            # Apply gamma correction to enhance contrast (reversible)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            adjusted = cv2.LUT(image, table)
            return adjusted

        def sharpen_image(image):
            # Create a sharpening kernel
            kernel = np.array([[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])
            
            # Sharpen the image using filter2D convolution
            sharpened_image = cv2.filter2D(image, -1, kernel)
            
            return sharpened_image
        def create_alpha_mask(shape, edge_width):
            """Create an alpha mask with smooth edges for blending."""
            h, w = shape[:2]
            mask = np.zeros((h, w), dtype=np.float32)
            
            # Horizontal gradient
            for i in range(edge_width):
                alpha = i / edge_width
                mask[:, i] = alpha
                mask[:, w - i - 1] = alpha
            
            # Vertical gradient
            for j in range(edge_width):
                alpha = j / edge_width
                mask[j, :] = np.maximum(mask[j, :], alpha)
                mask[h - j - 1, :] = np.maximum(mask[h - j - 1, :], alpha)
            
            mask[edge_width:-edge_width, edge_width:-edge_width] = 1
            
            return mask

        def blend_images(base_img, new_img, mask, x, y):
            """Blend new_img into base_img at position (x, y) using mask."""
            roi = base_img[y:y + new_img.shape[0], x:x + new_img.shape[1]]
            
            # Convert mask to 3 channels
            alpha = mask[:, :, np.newaxis]
            
            # Blend images
            blended = roi * (1 - alpha) + new_img * alpha
            base_img[y:y + new_img.shape[0], x:x + new_img.shape[1]] = blended

        # Define folder paths
        if os.path.isdir(output_folder_normalized):
            folder_path_enh = output_folder_normalized
        else:
            folder_path_enh = output_folder_crop
        output_folder_enh = os.path.join(folder_path, 'enhanced_images')

        # Create output folder if it doesn't exist
        os.makedirs(output_folder_enh, exist_ok=True)

        # Get list of image files in the folder
        image_files = glob.glob(os.path.join(folder_path_enh, '*.tiff'))

        # Iterate through each image file
        for image_file in image_files:
            # Load image
            original_image = cv2.imread(image_file)
            
            if original_image is None:
                print(f"Error: Could not read image from {image_file}")
                continue
            
            # Enhance the image quality
            enhanced_image = enhance_image_quality(original_image)
            
            # Display and save the enhanced image
            filename = os.path.basename(image_file)
            output_path_enh = os.path.join(output_folder_enh, f"enhanced_{filename}")
            
            cv2.imwrite(output_path_enh, enhanced_image)
            
        print(f"{len(image_files)} images enhanced and saved to {output_folder_enh}")
        sys.stdout.flush()  # Flush stdout to ensure immediate display

        ## Creating mosaic

        print("Creating mosaic, this might take a few minutes")
        sys.stdout.flush()

        # Function to extract numeric part from filename
        def extract_number(filename):
            return int(filename.split('_')[-1].split('.')[0])

        # Define the folder containing the images
        source_folder = output_folder_enh

        # Get list of image files in the source folder
        image_files = sorted([f for f in os.listdir(source_folder) if f.endswith('.tiff')])  # Adjust file extension as needed

        # Sort image files based on the extracted number
        image_files_sorted = sorted(image_files, key=lambda x: extract_number(x))

        # Create a Mosaic object
        mosaic = Mosaic(source_folder)

        # Align and smooth seams
        mosaic.align()
        mosaic.smooth_seams()

        # Show the stitched image (if applicable in your library)
        # mosaic.show()

        # Define the folder to save the JSON parameters file
        output_folder_mosaic = output_folder_enh

        # Save the parameters to a JSON file
        params_file = "params_mosaic.json"
        params_path = os.path.join(output_folder_mosaic, params_file)
        mosaic.save_params(params_path)

        # Verify the saved parameters file
        with open(params_path, 'r') as f:
            params_data = json.load(f)
            if 'filenames' in params_data:
                print(f"Parameters include {len(params_data['coords'])} image(s)")

        print(f"Parameters saved to: {params_path}")
        sys.stdout.flush()  # Flush stdout to ensure immediate display

        ## Stitch images

        print("Stitching images")
        sys.stdout.flush()

        # Load the JSON file
        json_path = os.path.join(output_folder_mosaic, 'params_mosaic.json')
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract metadata
        tile_shape = data['metadata']['tile_shape']
        tile_width, tile_height = tile_shape[1], tile_shape[0]

        # Determine the size of the canvas
        max_x = max([coord[1] for coord in data['coords'].values()]) + tile_width
        max_y = max([coord[0] for coord in data['coords'].values()]) + tile_height

        # Create a blank canvas (black background)
        canvas = np.zeros((int(max_y), int(max_x), 3), dtype=np.uint8)

        # Set edge width for blending
        edge_width = 30  # Adjust as needed

        # Place each image on the canvas with blending
        for key, coord in data['coords'].items():
            filename = data['filenames'][key]
            img_path = os.path.join(output_folder_mosaic, filename)
            img = cv2.imread(img_path)
            x, y = int(coord[1]), int(coord[0])  # Flipped coordinates

            # Create an alpha mask for the current image
            alpha_mask = create_alpha_mask(img.shape, edge_width)

            # Blend the current image onto the canvas
            blend_images(canvas, img, alpha_mask, x, y)

        # Resize the final stitched image for display
        resize_factor = 0.6  # Resize to 60% of the original size
        resized_canvas = cv2.resize(canvas, None, fx=resize_factor, fy=resize_factor)

        # Display the final stitched image in a resized window
        # cv2.namedWindow('Stitched Image', cv2.WINDOW_NORMAL)  # Create a resizable window
        # cv2.resizeWindow('Stitched Image', int(resized_canvas.shape[1]), int(resized_canvas.shape[0]))  # Resize the window
        # cv2.imshow('Stitched Image', resized_canvas)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Define folder to save stitched image in
        output_folder_stitch = os.path.join(folder_path, 'Stitching result')

        # Create folder if it doesn't exist
        os.makedirs(output_folder_stitch, exist_ok=True)

        #  Save the stitched image
        stitched_image_name = f'stitched_image.tiff'
        output_path = os.path.join(output_folder_stitch, stitched_image_name)
        cv2.imwrite(output_path, canvas)

        print(f"Stitched image saved to {output_folder_stitch}")
        sys.stdout.flush()  # Flush stdout to ensure immediate display


# Create the main window
window = tk.Tk()

# Configure dark mode appearance
window.configure(bg="#202020")
window.tk_setPalette(background="#202020", foreground="white")

# Create the Atlas app
app = Atlas(window)

# Start the main event loop
window.mainloop()
