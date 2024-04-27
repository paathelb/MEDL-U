import cv2
import imageio

# path = '/home/hpaat/FF/csgm-mri-langevin/saved_fig_hmc/'
path = '/home/hpaat/my_exp/MTrans-evidential/small_experiments/saved_scatter_plot/'

# List of image filenames
# image_filenames = [
#     "/home/hpaat/FF/csgm-mri-langevin/new_data/images/fwd_rev_y_slice_0_p1.png",
#     "/home/hpaat/FF/csgm-mri-langevin/new_data/images/x_slice_0.png",
#     "/home/hpaat/FF/csgm-mri-langevin/new_data/images/y_imag_slice_0.png",
#     "/home/hpaat/FF/csgm-mri-langevin/new_data/images/y_real_slice_0.png",
#     # Add more image filenames here
# ]

# image_titles = [
#     "fwd_rev_y_slice_0",
#     "x_slice_0",
#     "y_imag_slice_0",
#     "y_real_slice_0",
#     # Add more titles here
# ]

image_filenames = []
image_titles = []
for number in range(9,289,10):
    image_filenames.append(path + 'img_' + str(number) + '.png')
    image_titles.append("Epoch_" + str(number))

# Output GIF filename
output_filename = path + "show.gif"

# Duration (in seconds) for each frame
frame_duration = 0.50

# Create a list to store the frames
frames = []

# Iterate over the image filenames and titles
for image_filename, title in zip(image_filenames, image_titles):
    # Read the image using OpenCV
    image = cv2.imread(image_filename)

    # Add the title to the image using OpenCV
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position = (200, 200)
    font_scale = 0.50
    font_color = (0, 0, 0)  # White
    font_thickness = 2
    cv2.putText(image, title, text_position, font, font_scale, font_color, font_thickness)

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Append the image to the frames list
    frames.append(image_rgb)

# Save the frames as a GIF using imageio
imageio.mimsave(output_filename, frames, duration=frame_duration)

print("GIF created successfully!")
