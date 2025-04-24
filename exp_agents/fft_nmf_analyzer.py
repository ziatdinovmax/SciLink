import numpy as np
from scipy import fftpack
from scipy import ndimage
from sklearn.decomposition import NMF
from skimage.util import view_as_windows
from skimage import io, color
import os

from .utils import load_image


class SlidingFFTNMF:
    
    def __init__(self, window_size_x=64, window_size_y=64, 
                 window_step_x=16, window_step_y=16,
                 interpolation_factor=2, zoom_factor=2, 
                 hamming_filter=True, components=4):
        '''Sliding Window FFT with NMF unmixing.
        This class calculates the FFT window transform
        and unmixes the output using NMF'''

        self.window_step_x = window_step_x
        self.window_step_y = window_step_y
        self.window_size_x = window_size_x
        self.window_size_y = window_size_y
        self.interpol_factor = interpolation_factor
        self.zoom_factor = zoom_factor
        self.hamming_filter = hamming_filter
        self.components = components
        
        # Initialize hamming window
        bw2d = np.outer(np.hamming(self.window_size_x), np.ones(self.window_size_y))
        self.hamming_window = np.sqrt(bw2d * bw2d.T)
        
    def make_windows(self, image):
        """Generate windows from an image using efficient striding operations"""
        
        # Handle color images by converting to grayscale
        if len(image.shape) > 2:
            # Convert RGB to grayscale 
            if image.shape[2] >= 3:
                image = color.rgb2gray(image[:,:,:3])  # Handle RGBA images
            else:
                image = np.mean(image, axis=2)  # Simple average for other formats
        
        # Ensure image is float type and normalize to 0-1
        image = image.astype(float)
        if np.max(image) > 0:  # Avoid division by zero
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        # Check if image is big enough for windowing
        if image.shape[0] < self.window_size_x or image.shape[1] < self.window_size_y:
            raise ValueError(f"Image dimensions {image.shape} are smaller than window size ({self.window_size_x}, {self.window_size_y})")
        
        # Pad image if necessary to ensure we can extract at least one window
        pad_x = max(0, self.window_size_x - image.shape[0])
        pad_y = max(0, self.window_size_y - image.shape[1])
        if pad_x > 0 or pad_y > 0:
            image = np.pad(image, ((0, pad_x), (0, pad_y)), mode='constant')
            print(f"Image padded to size {image.shape}")
        
        # Define window parameters
        window_size = (self.window_size_x, self.window_size_y)
        window_step = (self.window_step_x, self.window_step_y)
        
        # Use view_as_windows to efficiently create sliding windows
        windows = view_as_windows(image, window_size, step=window_step)
        
        # Store window shape information for later visualization
        self.windows_shape = (windows.shape[0], windows.shape[1])
        print(f"Created {self.windows_shape[0]}×{self.windows_shape[1]} = {windows.shape[0] * windows.shape[1]} windows")
        
        # Create position vectors for visualization
        x_positions = np.arange(0, windows.shape[1] * window_step[1], window_step[1])
        y_positions = np.arange(0, windows.shape[0] * window_step[0], window_step[0])
        xx, yy = np.meshgrid(x_positions, y_positions)
        self.pos_vec = np.column_stack((yy.flatten(), xx.flatten()))
        
        # Reshape to the expected output format
        return windows.reshape(-1, window_size[0], window_size[1])

    def process_fft(self, windows):
        """Perform FFT on each window with optional hamming filter and zooming"""
        
        num_windows = windows.shape[0]
        fft_results = []
        
        for i in range(num_windows):
            img_window = windows[i].copy()  # Make a copy to avoid modifying original
            
            # Apply Hamming filter if requested
            if self.hamming_filter:
                img_window = img_window * self.hamming_window
                
            # Compute 2D FFT and shift for visualization
            fft_result = fftpack.fftshift(fftpack.fft2(img_window))
            
            # Take the magnitude of the complex FFT result (ensures non-negative values)
            fft_mag = np.abs(fft_result)
            
            # Apply log transform to enhance visibility of lower amplitude frequencies
            fft_mag = np.log1p(fft_mag)  # log(1+x) avoids log(0) issues
            
            # Zoom in on center region
            center_x, center_y = self.window_size_x // 2, self.window_size_y // 2
            zoom_size = max(1, self.window_size_x // (2 * self.zoom_factor))  # Ensure minimum size of 1
            
            # Extract center region, with boundary checking
            x_min = max(0, center_x - zoom_size)
            x_max = min(fft_mag.shape[0], center_x + zoom_size)
            y_min = max(0, center_y - zoom_size)
            y_max = min(fft_mag.shape[1], center_y + zoom_size)
            
            zoomed = fft_mag[x_min:x_max, y_min:y_max]
            
            # Apply interpolation if the interpol factor is greater than 1
            if self.interpol_factor > 1:
                try:
                    final_fft = ndimage.zoom(zoomed, self.interpol_factor, order=1)
                except:
                    print(f"Warning: Interpolation failed for window {i}, using original")
                    final_fft = zoomed
            else:
                final_fft = zoomed
            
            fft_results.append(final_fft)
        
        # Ensure all results have the same shape by padding if necessary
        shapes = [result.shape for result in fft_results]
        max_shape = tuple(max(s[i] for s in shapes) for i in range(2))
        
        for i, result in enumerate(fft_results):
            if result.shape != max_shape:
                padded = np.zeros(max_shape)
                padded[:result.shape[0], :result.shape[1]] = result
                fft_results[i] = padded
        
        self.fft_size = max_shape
        result_array = np.array(fft_results)
        
        # Final check for NaN or Inf values
        result_array = np.nan_to_num(result_array)
        
        return result_array
    
    def run_nmf(self, fft_results):
        """Run NMF on FFT results to extract components"""
        
        # Reshape for NMF
        fft_flat = fft_results.reshape(fft_results.shape[0], -1)
        
        # Ensure all values are non-negative
        fft_flat = np.maximum(0, fft_flat)  # Hard clip any negatives to zero
        
        # Check if we have valid data
        if np.all(fft_flat == 0) or np.isnan(fft_flat).any() or np.isinf(fft_flat).any():
            raise ValueError("Invalid data for NMF: contains zeros, NaNs or Infs")
        
        # Check if we have enough windows
        if fft_flat.shape[0] < self.components:
            print(f"Warning: Number of windows ({fft_flat.shape[0]}) is less than components ({self.components})")
            self.components = min(fft_flat.shape[0], 3)  # Reduce components to avoid error
            print(f"Reducing components to {self.components}")
            
        nmf = NMF(
            n_components=self.components, 
            init='random', 
            random_state=42, 
            max_iter=1000,
            tol=1e-4,
            solver='cd'  # Coordinate descent is typically more robust
        )
        abundances = nmf.fit_transform(fft_flat)
        components = nmf.components_
       
        # Reshape components and abundances for visualization
        try:
            components = components.reshape(self.components, self.fft_size[0], self.fft_size[1])
            abundances = abundances.reshape(self.windows_shape[0], self.windows_shape[1], self.components)
        except Exception as e:
            print(f"Error reshaping results: {e}")
            # Try to reshape in a more flexible way
            components_flat = components.copy()
            components = np.zeros((self.components, self.fft_size[0], self.fft_size[1]))
            for i in range(self.components):
                flat_size = min(components_flat[i].size, self.fft_size[0] * self.fft_size[1])
                components[i].flat[:flat_size] = components_flat[i][:flat_size]
            
            abundances = np.zeros((self.windows_shape[0], self.windows_shape[1], self.components))
            for i in range(min(abundances.shape[2], self.components)):
                abundances[:,:,i] = abundances.reshape(-1, self.components)[:,i].reshape(self.windows_shape)
        
        return components, abundances
    

    def analyze_image(self, image_path, output_path=None):
        """Full analysis pipeline for an image"""
        
        # Store image path for later use
        self.image_path = image_path
        
        # Read the image
        print(f"Reading image: {image_path}")
        image = load_image(image_path)
            
        if output_path is None:
            base_dir = os.path.dirname(image_path)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(base_dir, f"{base_name}_analysis")
        
        print("Creating windows...")
        windows = self.make_windows(image)
        
        print("Computing FFTs...")
        fft_results = self.process_fft(windows)
        
        print("Running NMF analysis...")
        components, abundances = self.run_nmf(fft_results)

        print("Saving NumPy arrays...")
        np.save(f"{output_path}_components.npy", components)
        np.save(f"{output_path}_abundances.npy", abundances.transpose(-1, 0, 1))

        abundances = abundances.transpose(-1, 0, 1) # (n_components, h, w)
        
        return components, abundances