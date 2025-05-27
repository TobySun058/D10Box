import fitz  # PyMuPDF
from PIL import Image
import os

# Function to convert PDF pages to high-quality JPG
def pdf_to_jpg(pdf_path, output_folder, zoom_x=4.0, zoom_y=4.0):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through each page of the PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # Load each page

        # Apply zoom matrix to increase resolution
        matrix = fitz.Matrix(zoom_x, zoom_y)
        pix = page.get_pixmap(matrix=matrix)

        # Convert pixmap to a Pillow Image
        mode = "RGB" if pix.alpha == 0 else "RGBA"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

        # Define output image path
        output_image_path = os.path.join(output_folder, f"page_{page_num + 1}.jpg")
        
        # Save the image as a high-quality JPG
        img.convert("RGB").save(output_image_path, "JPEG", quality=100)
        print(f"Saved: {output_image_path}")
    
    print(f"All pages have been converted and saved to {output_folder}.")

# Example usage
pdf_path = './drawings/Creston_Dement_Public_Library.pdf'  # Your PDF file path
output_folder = './drawings/Creston_Dement_Public_Library'  # Folder to save images

pdf_to_jpg(pdf_path, output_folder, zoom_x=2.0, zoom_y=2.0)  # 2.0 = 144 DPI, 3.0 = 216 DPI, etc.
