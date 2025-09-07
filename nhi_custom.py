import sys, os, cv2, torch, argparse, numpy as np
sys.path.append('core')
from config.parser import parse_args
from raft import RAFT
from utils.flow_viz import flow_to_image
from utils.utils import load_ckpt
def create_color_bar(height, width, color_map):
    """
    Create a color bar image using a specified color map.

    :param height: The height of the color bar.
    :param width: The width of the color bar.
    :param color_map: The OpenCV colormap to use.
    :return: A color bar image.
    """
    # Generate a linear gradient
    gradient = np.linspace(0, 255, width, dtype=np.uint8)
    gradient = np.repeat(gradient[np.newaxis, :], height, axis=0)

    # Apply the colormap
    color_bar = cv2.applyColorMap(gradient, color_map)

    return color_bar

def add_color_bar_to_image(image, color_bar, orientation='vertical'):
    """
    Add a color bar to an image.

    :param image: The original image.
    :param color_bar: The color bar to add.
    :param orientation: 'vertical' or 'horizontal'.
    :return: Combined image with the color bar.
    """
    if orientation == 'vertical':
        return cv2.vconcat([image, color_bar])
    else:
        return cv2.hconcat([image, color_bar])

def get_heatmap(info, args):
    raw_b = info[:, 2:]
    log_b = torch.zeros_like(raw_b)
    weight = info[:, :2].softmax(dim=1)
    log_b[:, 0] = torch.clamp(raw_b[:,0], min=0, max=args.var_max)
    log_b[:,1] = torch.clamp(raw_b[:, 1], min =args.var_min, max=0)
    return (log_b*weight).sum(dim=1, keepdim =True)


def vis_heatmap(name, image, heatmap):
    # theta = 0.01
    # print(heatmap.max(), heatmap.min(), heatmap.mean())
    heatmap = heatmap[:, :, 0]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    # heatmap = heatmap > 0.01
    heatmap = (heatmap * 255).astype(np.uint8)
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = image * 0.3 + colored_heatmap * 0.7
    # Create a color bar
    height, width = image.shape[:2]
    color_bar = create_color_bar(50, width, cv2.COLORMAP_JET)  # Adjust the height and colormap as needed
    # Add the color bar to the image
    overlay = overlay.astype(np.uint8)
    combined_image = add_color_bar_to_image(overlay, color_bar, 'vertical')
    cv2.imwrite(name, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

def calc_flow(args, model, img1, img2):
    output = model(img1, img2, iters=args.iters, test_mode= True)
    return output['flow'][-1], output['info'][-1]
def process_image_pair(model, args, img1_path, img2_path, device):
    """Process a single pair of images"""
    img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)

    #Convert to tensors
    img1_tensor = torch.tensor(img1, dtype=torch.float32).permute(2,0,1)[None].to(device)
    img2_tensor = torch.tensor(img2, dtype = torch.float32).permute(2,0,1)[None].to(device)

    #Calculate flow and uncertainty
    flow, info = calc_flow(args, model, img1_tensor, img2_tensor)
    heatmap = get_heatmap(info, args)

    #generate visualizations
    flow_vis = flow_to_image(flow[0].permute(1,2,0).cpu().numpy(), convert_to_bgr = True)

    # cv2.imwrite(img2_path.replace(".jpg", "_flow.jpg"), flow_vis)
    # HEATMAP OVERLAY
    hm = heatmap[0,0].cpu().numpy()
    hm = ((hm - hm.min()/ (hm.max() - hm.min()))* 255).astype(np.uint8)
    hm_colored = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    overlay = (img1 *0.3 + cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)*0.7).astype(np.uint8)
    heatmap_vis = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    return flow_vis, heatmap_vis
    
    # vis_heatmap(img2_path.replace(".jpg", "_heatmap.jpg"), img2_tensor.permute(1, 2, 0).cpu().numpy(), heatmap[0].permute(1, 2, 0).cpu().numpy())

@torch.no_grad()
def process_image_sequence(model, args, image_dir, pattern, output_dir, device):
    """Process sequence of images"""
    import glob
    os.makedirs(output_dir, exist_ok = True)

    images = sorted(glob.glob(f"{image_dir}/{pattern}"))
    for i in range(len(images)-1):
        flow_vis, heatmap_vis = process_image_pair(model, args, images[i], images[i+1],device)

        cv2.imwrite(f'{output_dir}/flow_{i}.jpg', flow_vis)
        cv2.imwrite(f'{output_dir}/heatmap_{i}.jpg', heatmap_vis)
        if i %30 == 0: 
            print(f"Processed {i+1}/{len(images) -1} pairs")
    print("=====================\nDone")

@torch.no_grad()
def process_video(model, args, video_path, output_dir, device):
    """Process video files"""
    os.makedirs(output_dir, exist_ok = True)
    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    frame_idx = 0
    while True:
        ret, curr_frame = cap.read()
        if not ret: break

        flow_vis, heatmap_vis= process_image_pair(model, args, None, None, device)

        prev_rgb=cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
        curr_rgb=cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
        prev_tensor =torch.tensor(prev_rgb, dtype=torch.float32).permute(2,0,1)[None].to(device)
        curr_tensor=torch.tensor(curr_rgb, dtype=torch.float32).permute(2,0,1)[None].to(device)

        flow, info = calc_flow(args, model, prev_frame, curr_frame)
        heatmap = get_heatmap(info,args)
        
        flow_vis = flow_to_image(flow[0].permute(1,2,0).cpu().numpy(), convert_to_bgr=True)
        hm = heatmap[0,0].cpu().numpy()
        hm = ((hm-hm.min())/(hm.max()-hm.min())*255).astype(np.uint8)
        hm_colored= cv2.applyColorMap(hm,cv2.COLORMAP_JET)

        overlay = (prev_rgb*0.3+ cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)*0.7).astype(np.uint8)
        heatmap_vis =cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(f'{output_dir}/flow_{frame_idx}.jpg', flow_vis)
        cv2.imwrite(f'{output_dir}/heatmap_{frame_idx},jpg', heatmap_vis)

        prev_frame = curr_frame
        frame_idx +=1
        if frame_idx % 30 == 0: print(f"Processed {frame_idx} frames")
    cap.release()
    print(f"=====================\nDone")
def images_to_video(image_dir, output_path, pattern, fps=30):
    """Convert sequence of images to video"""
    import glob
    from natsort import natsorted
    images = natsorted(glob.glob(f"{image_dir}/{pattern}"))
    if not images: 
        print(f"No images found with pattern: {image_dir}/{pattern}")
        return
    
    # Read first image to get dimensions
    frame = cv2.imread(images[0])
    h, w, _ = frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for img_path in images:
        frame = cv2.imread(img_path)
        writer.write(frame)
    
    writer.release()
    print(f"Created video: {output_path} from {len(images)} images")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--path', default=None)
    parser.add_argument('--url', default=None)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--video', help='input video file')
    parser.add_argument('--images', help='input image directory')
    parser.add_argument('--pattern', default='*.jpg', help='image pattern (e.g., img_*.jpg)')
    parser.add_argument('--output', default='./video_output')
    parser.add_argument('--fps', type=int, default=30, help='output video fps')
    parser.add_argument('--make-videos', action='store_true', help='create videos from images')

    args =parse_args(parser)

    if args.make_videos:
        images_to_video(args.output, f"{args.output}/flow_video.mp4", "flow_*.jpg", args.fps)
        images_to_video(args.output, f"{args.output}/heatmap_video.mp4", "heatmap_*.jpg", args.fps)
        return
    
    #load model 
    model = RAFT(args)
    if args.path: load_ckpt(model, args.path)
    else: model = RAFT.from_pretrained(args.url, args= args)

    device= torch.device(args.device)
    model.to(device).eval()

    # Process input
    if args.images:
        process_image_sequence(model, args, args.images, args.pattern, args.output, device)
    elif args.video:
        process_video(model, args, args.video, args.output, device)
    else:
        print("Provide either --video or --images")
        return
    if args.make_videos:
        images_to_video(args.output, f"{args.output}/flow_video.mp4", "flow_*.jpg", args.fps)
        images_to_video(args.output, f"{args.output}/heatmap_video.mp4", "heatmap_*.jpg", args.fps)

if __name__ == '__main__':
    main()
#python nhi_custom.py --cfg config/eval/spring-M.json --path models/Tartan-C-T-TSKH-spring540x960-M.pth --images "C:/Users/ongng/OneDrive/Desktop/rover_data_20250804_112504/" --pattern "photo*.jpg" --output ./results

#python nhi_custom.py --cfg config/eval/spring-M.json --path models/Tartan-C-T-TSKH-spring540x960-M.pth --video dummy --make-videos --output ./results --fps 24