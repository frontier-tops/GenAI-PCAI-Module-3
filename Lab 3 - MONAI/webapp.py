import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import AsDiscrete
from monai.config import print_config
from monai.transforms import LoadImage, Orientation
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose
)

# Create directories to save generated and segmented images/videos
os.makedirs("generated_images", exist_ok=True)
os.makedirs("segmented_images", exist_ok=True)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .title {
        font-size: 2.5em;
        color: #333;
        text-align: center;
        margin-bottom: 20px;
    }
    .image-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        margin: 10px 0;
    }
    .caption {
        text-align: center;
        color: #666;
        font-size: 1.1em;
    }
    </style>
""", unsafe_allow_html=True)

def find_label_center_loc(x):
    """
    Find the center location of non-zero elements in a binary mask.

    Args:
        x (torch.Tensor): Binary mask tensor. Expected shape: [H, W, D] or [C, H, W, D].

    Returns:
        list: Center locations for each dimension. Each element is either
              the middle index of non-zero locations or None if no non-zero elements exist.
    """
    label_loc = torch.where(x != 0)
    center_loc = []
    for loc in label_loc:
        unique_loc = torch.unique(loc)
        if len(unique_loc) == 0:
            center_loc.append(None)
        else:
            center_loc.append(unique_loc[len(unique_loc) // 2])

    return center_loc


def normalize_label_to_uint8(colorize, label, n_label):
    """
    Normalize and colorize a label tensor to a uint8 image.

    Args:
        colorize (torch.Tensor): Weight tensor for colorization. Expected shape: [3, n_label, 1, 1].
        label (torch.Tensor): Input label tensor. Expected shape: [1, H, W].
        n_label (int): Number of unique labels.

    Returns:
        numpy.ndarray: Normalized and colorized image as uint8 numpy array. Shape: [H, W, 3].
    """
    with torch.no_grad():
        post_label = AsDiscrete(to_onehot=n_label)
        label = post_label(label).permute(1, 0, 2, 3)
        label = F.conv2d(label, weight=colorize)
        label = torch.clip(label, 0, 1).squeeze().permute(1, 2, 0).cpu().numpy()

    draw_img = (label * 255).astype(np.uint8)

    return draw_img


def visualize_one_slice_in_3d(image, axis: int = 2, center=None, mask_bool=True, n_label=105, colorize=None):
    """
    Extract and visualize a 2D slice from a 3D image or label tensor.

    Args:
        image (torch.Tensor): Input 3D image or label tensor. Expected shape: [1, H, W, D].
        axis (int, optional): Axis along which to extract the slice (0, 1, or 2). Defaults to 2.
        center (int, optional): Index of the slice to extract. If None, the middle slice is used.
        mask_bool (bool, optional): If True, treat the input as a label mask and normalize it. Defaults to True.
        n_label (int, optional): Number of labels in the mask. Used only if mask_bool is True. Defaults to 105.
        colorize (torch.Tensor, optional): Colorization weights for label normalization.
                                           Expected shape: [3, n_label, 1, 1] if provided.

    Returns:
        numpy.ndarray: 2D slice of the input. If mask_bool is True, returns a normalized uint8 array
                       with shape [3, H, W]. Otherwise, returns a float32 array with shape [3, H, W].

    Raises:
        ValueError: If the specified axis is not 0, 1, or 2.
    """
    # draw image
    if center is None:
        center = image.shape[2:][axis] // 2
    if axis == 0:
        draw_img = image[..., center, :, :]
    elif axis == 1:
        draw_img = image[..., :, center, :]
    elif axis == 2:
        draw_img = image[..., :, :, center]
    else:
        raise ValueError("axis should be in [0,1,2]")
    if mask_bool:
        draw_img = normalize_label_to_uint8(colorize, draw_img, n_label)
    else:
        draw_img = draw_img.squeeze().cpu().numpy().astype(np.float32)
        draw_img = np.stack((draw_img,) * 3, axis=-1)
    return draw_img


def show_image(image, title="mask"):
    """
    Plot and display an input image.

    Args:
        image (numpy.ndarray): Image to be displayed. Expected shape: [H, W] for grayscale or [H, W, 3] for RGB.
        title (str, optional): Title for the plot. Defaults to "mask".
    """
    plt.figure("check", (24, 12))
    plt.subplot(1, 2, 1)
    plt.title(title)
    plt.imshow(image)
    plt.show()


def to_shape(a, shape):
    """
    Pad an image to a desired shape.

    This function pads a 3D numpy array (image) with zeros to reach the specified shape.
    The padding is added equally on both sides of each dimension, with any odd padding
    added to the end.

    Args:
        a (numpy.ndarray): Input 3D array to be padded. Expected shape: [X, Y, Z].
        shape (tuple): Desired output shape as (x_, y_, z_).

    Returns:
        numpy.ndarray: Padded array with the desired shape [x_, y_, z_].

    Note:
        If the input shape is larger than the desired shape in any dimension,
        no padding is removed; the original size is maintained for that dimension.
        Padding is done using numpy's pad function with 'constant' mode (zero-padding).
    """
    x_, y_, z_ = shape
    x, y, z = a.shape
    x_pad = x_ - x
    y_pad = y_ - y
    z_pad = z_ - z
    return np.pad(
        a,
        (
            (x_pad // 2, x_pad // 2 + x_pad % 2),
            (y_pad // 2, y_pad // 2 + y_pad % 2),
            (z_pad // 2, z_pad // 2 + z_pad % 2),
        ),
        mode="constant",
    )


def get_xyz_plot(image, center_loc_axis, mask_bool=True, n_label=105, colorize=None, target_class_index=0):
    """
    Generate a concatenated XYZ plot of 2D slices from a 3D image.

    This function creates visualizations of three orthogonal slices (XY, XZ, YZ) from a 3D image
    and concatenates them into a single 2D image.

    Args:
        image (torch.Tensor): Input 3D image tensor. Expected shape: [1, H, W, D].
        center_loc_axis (list): List of three integers specifying the center locations for each axis.
        mask_bool (bool, optional): Whether to apply masking. Defaults to True.
        n_label (int, optional): Number of labels for visualization. Defaults to 105.
        colorize (torch.Tensor, optional): Colorization weights. Expected shape: [3, n_label, 1, 1] if provided.
        target_class_index (int, optional): Index of the target class. Defaults to 0.

    Returns:
        numpy.ndarray: Concatenated 2D image of the three orthogonal slices. Shape: [max(H,W,D), 3*max(H,W,D), 3].

    Note:
        The output image is padded to ensure all slices have the same dimensions.
    """
    target_shape = list(image.shape[1:])  # [1,H,W,D]
    img_list = []

    for axis in range(3):
        center = center_loc_axis[axis]

        img = visualize_one_slice_in_3d(
            torch.flip(image.unsqueeze(0), [-3, -2, -1]),
            axis,
            center=center,
            mask_bool=mask_bool,
            n_label=n_label,
            colorize=colorize,
        )
        img = img.transpose([2, 1, 0])

        img = to_shape(img, (3, max(target_shape), max(target_shape)))
        img_list.append(img)
        img = np.concatenate(img_list, axis=2).transpose([1, 2, 0])
    return img

def generate_mri_sample(config_path: str, output_dir: str):
    import glob
    import os
    # Capture existing output files
    existing_files = set(glob.glob(os.path.join(output_dir, "**", "*.nii.gz"), recursive=True))

    import monai
    monai.bundle.run(config_file=config_path)

    # Identify new files
    all_files = set(glob.glob(os.path.join(output_dir, "**", "*.nii.gz"), recursive=True))
    new_files = all_files - existing_files

    if new_files:
        return max(new_files, key=os.path.getmtime)  # Return the most recent MRI file
    else:
        return None  # No new MRI file detected
    
def generate_mri_sample(config_path: str, output_dir: str):
    import glob
    import os
    existing_files = set(glob.glob(os.path.join(output_dir, "**", "*.nii.gz"), recursive=True))

    import monai
    monai.bundle.run(config_file=config_path)

    all_files = set(glob.glob(os.path.join(output_dir, "**", "*.nii.gz"), recursive=True))
    new_files = all_files - existing_files

    if new_files:
        return max(new_files, key=os.path.getmtime)
    else:
        return None

def generate_3d_mri_volume():
    filepath = generate_mri_sample(config_path="configs/inference.json", output_dir="output/")

    # load image/mask pairs
    loader = LoadImage(image_only=True, ensure_channel_first=True)
    orientation = Orientation(axcodes="RAS")
    image_volume = orientation(loader(filepath))
    print(image_volume.shape)
    st.success("MRI volume generated successfully!")
    return image_volume

def segment_image(image_volume):
    # Segment the 3D volume (example: thresholding on each slice)
    VAL_AMP = True

    # standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
    device = torch.device("cuda:0")
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)

    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # define inference method
    def inference(input):
        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(240, 240, 160),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )

        if VAL_AMP:
            with torch.cuda.amp.autocast():
                return _compute(input)
        else:
            return _compute(input)
        
    model.load_state_dict(torch.load(os.path.join("./models", "best_metric_model.pth")))
    model.eval()

    with torch.no_grad():
        val_input = image_volume.unsqueeze(1).repeat(1, 4, 1, 1, 1).to(device)
        val_output = inference(val_input)
        print(val_input.shape, val_output.shape)
        val_output = post_trans(val_output[0])
        segmented_volume = torch.sum(val_output, dim=0, keepdim=True)
    st.success("MRI volume segmented successfully!")
    return segmented_volume

def create_animation_mp4(volume, output_path, caption):
    # Create animation from 3D volume and save as MP4
    frames = []
    # Ensure the volume is a NumPy array (convert if it's a PyTorch tensor)
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()  # Convert tensor to NumPy array
    volume = volume[0]
    fig = plt.figure(figsize=(6, 6))
    for i in range(volume.shape[0]):
        frames.append([plt.imshow(volume[i], aspect=1.0, cmap="gray", animated=True)])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    try:
        ani.save(output_path, writer='ffmpeg', fps=20)
        st.success(f"Animation saved Successfully!")
    except Exception as e:
        st.error(f"Failed to save animation: {str(e)}")
    plt.close(fig)  # Close the figure to free memory

    # Display the video if it exists
    if os.path.exists(output_path):
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.video(output_path)
        st.markdown(f'<p class="caption">{caption} ({volume.shape[0]} Slices)</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning(f"Video file {output_path} not found.")

def main():
    st.markdown('<h1 class="title">3D MRI Generator + segmentation</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        with st.container():
            st.subheader("Generate 3D MRI Volume")
            if st.button("Generate MRI", key="generate"):
                with st.spinner("Generating 3D MRI volume..."):
                    # Generate 3D MRI volume
                    image_volume = generate_3d_mri_volume()
                    st.session_state["image_volume"] = image_volume
                    # st.success("MRI volume generated successfully!")

            # Display animation for generated volume
            if "image_volume" in st.session_state:
                image_volume = st.session_state["image_volume"]
                output_path = "generated_images/mri_animation.mp4"
                create_animation_mp4(image_volume, output_path, "Generated MRI Animation")

    with col2:
        with st.container():
            st.subheader("Segment 3D MRI Volume")
            if st.button("Segment MRI", key="segment"):
                if "image_volume" in st.session_state:
                    with st.spinner("Segmenting 3D MRI volume..."):
                        image_volume = st.session_state["image_volume"]
                        segmented_volume = segment_image(image_volume)
                        st.session_state["segmented_volume"] = segmented_volume
                        # st.success("MRI volume segmented successfully!")
                else:
                    st.warning("Please generate an MRI volume first!")

            # Display animation for segmented volume
            if "segmented_volume" in st.session_state:
                segmented_volume = st.session_state["segmented_volume"]
                output_path = "segmented_images/segmented_mri_animation.mp4"
                create_animation_mp4(segmented_volume, output_path, "Segmented MRI Animation")

    # Footer
    st.markdown("""
        <hr style="border: 1px solid #ddd;">
        <p style="text-align: center; color: #888;">Powered by Streamlit & MONAI</p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()