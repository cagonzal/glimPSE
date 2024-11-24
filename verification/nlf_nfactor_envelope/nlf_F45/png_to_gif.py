import os
from PIL import Image

def create_gif_from_pngs(input_folder, output_file, duration=500):
    # Get a list of PNG files in the input folder
    png_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    
    # Sort the files to ensure correct order
    png_files.sort()
    
    # Open all PNG files
    images = []
    for png_file in png_files:
        file_path = os.path.join(input_folder, png_file)
        img = Image.open(file_path)
        images.append(img)
    
    # Save the images as a GIF
    if images:
        images[0].save(
            output_file,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF created successfully: {output_file}")
    else:
        print("No PNG files found in the specified directory.")

# Example usage
input_folder = 'lst_pngs'
output_file = 'spectra.gif'
create_gif_from_pngs(input_folder, output_file)
