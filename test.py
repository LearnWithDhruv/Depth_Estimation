import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import time
import json
import os

CONFIG_FILE = "depth_calibration.json"
DEFAULT_CALIBRATION = {
    "depth_scale": 10.0,
    "focal_length": 500,
    "calibration_points": []
}

def load_or_create_calibration():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            print(f"Error loading calibration file. Creating new one.")
    
    # Create default calibration file
    with open(CONFIG_FILE, 'w') as f:
        json.dump(DEFAULT_CALIBRATION, f, indent=4)
    
    return DEFAULT_CALIBRATION

# Load MiDaS model - using a smaller and faster model
def load_midas_model():
    model_type = "MiDaS_small"  
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)

    transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return midas, transform, device

# Estimate depth using MiDaS
def estimate_depth(frame, midas, transform, device):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    
    input_batch = transform(img).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch.unsqueeze(0))
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
    
    return prediction

# Calculate calibrated distance
def calculate_distance(depth_map, point1, point2, calibration):
    # Get depth values at the two points
    depth1 = depth_map[point1[1], point1[0]]
    depth2 = depth_map[point2[1], point2[0]]
    
    # Apply calibration based on collected reference points
    z1 = calibrate_depth(depth1, calibration)
    z2 = calibrate_depth(depth2, calibration)
    
    # Calculate pixel distance
    pixel_distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    # Convert to real-world X-Y distance using depth and focal length
    focal_length = calibration["focal_length"]
    xy_distance = pixel_distance * ((z1 + z2) / 2) / focal_length
    
    # Calculate 3D Euclidean distance
    distance = np.sqrt(xy_distance**2 + (z1 - z2)**2)
    
    return distance, z1, z2

# Apply calibration to raw depth values
def calibrate_depth(raw_depth, calibration):
    # If we have calibration points, use them for interpolation
    if len(calibration["calibration_points"]) >= 2:
        # Sort calibration points by raw_depth
        cal_points = sorted(calibration["calibration_points"], key=lambda x: x["raw_depth"])
        
        # Find the two closest calibration points for interpolation
        if raw_depth <= cal_points[0]["raw_depth"]:
            # Below lowest calibration point, use linear extrapolation
            cal_ratio = cal_points[0]["real_distance"] / cal_points[0]["raw_depth"]
            return raw_depth * cal_ratio
        
        elif raw_depth >= cal_points[-1]["raw_depth"]:
            # Above highest calibration point, use linear extrapolation
            cal_ratio = cal_points[-1]["real_distance"] / cal_points[-1]["raw_depth"]
            return raw_depth * cal_ratio
        
        else:
            # Interpolate between two points
            for i in range(len(cal_points) - 1):
                if cal_points[i]["raw_depth"] <= raw_depth <= cal_points[i+1]["raw_depth"]:
                    # Linear interpolation
                    low_point = cal_points[i]
                    high_point = cal_points[i+1]
                    
                    # t is a value between 0 and 1 indicating position between low and high
                    t = (raw_depth - low_point["raw_depth"]) / (high_point["raw_depth"] - low_point["raw_depth"])
                    
                    # Interpolate the real distance
                    return low_point["real_distance"] + t * (high_point["real_distance"] - low_point["real_distance"])
    
    return raw_depth * calibration["depth_scale"]

# Add a calibration point
def add_calibration_point(calibration, raw_depth, real_distance):
    calibration["calibration_points"].append({
        "raw_depth": raw_depth,
        "real_distance": real_distance
    })
    
    # Save updated calibration
    with open(CONFIG_FILE, 'w') as f:
        json.dump(calibration, f, indent=4)
    
    return calibration

# Mouse callback to select points
points = []
def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) >= 2:
            points.pop(0) 
        points.append((x, y))
        print(f"Selected point: ({x}, {y})")

def main():
    calibration = load_or_create_calibration()
    print(f"Loaded calibration: depth_scale={calibration['depth_scale']}, focal_length={calibration['focal_length']}")
    print(f"Calibration points: {len(calibration['calibration_points'])}")
    
    midas, transform, device = load_midas_model()
    print(f"Using device: {device}")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    cv2.namedWindow("RGB Feed")
    cv2.setMouseCallback("RGB Feed", select_points)
    
    prev_frame_time = 0
    new_frame_time = 0
    
    process_every_n_frames = 5
    frame_count = 0
    last_depth_map = None
    
    calibration_mode = False
    current_real_distance = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        
        if frame_count % process_every_n_frames == 0:
            last_depth_map = estimate_depth(frame, midas, transform, device)
            
            depth_map_vis = cv2.normalize(last_depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_map_vis = cv2.applyColorMap(depth_map_vis, cv2.COLORMAP_MAGMA)
            
            for i, point in enumerate(calibration["calibration_points"]):
                text_pos = (20, 30 + i * 20)
                cv2.putText(depth_map_vis, f"Cal #{i+1}: Raw={point['raw_depth']:.2f}, Real={point['real_distance']:.2f}m", 
                            text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Depth Map", depth_map_vis)
        
        frame_count += 1
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        mode_text = "CALIBRATION MODE" if calibration_mode else "MEASUREMENT MODE"
        cv2.putText(frame, mode_text, (frame.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 0, 255) if calibration_mode else (0, 255, 0), 2)
        
        cv2.putText(frame, "C: Toggle calibration mode", (10, frame.shape[0] - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "+/-: Adjust real distance in calibration", (10, frame.shape[0] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if calibration_mode:
            cv2.putText(frame, f"Set real distance: {current_real_distance:.2f} m", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if len(points) == 2 and last_depth_map is not None:
            distance, depth1, depth2 = calculate_distance(last_depth_map, points[0], points[1], calibration)
            
            cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
            cv2.circle(frame, points[0], 5, (0, 0, 255), -1)
            cv2.circle(frame, points[1], 5, (0, 0, 255), -1)
            
            if calibration_mode:
                raw_depth_avg = (last_depth_map[points[0][1], points[0][0]] + last_depth_map[points[1][1], points[1][0]]) / 2
                cv2.putText(frame, f"Raw depth avg: {raw_depth_avg:.4f}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Press Enter to add calibration point", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"Distance: {distance:.2f} meters", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Depth 1: {depth1:.2f}m, Depth 2: {depth2:.2f}m", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("RGB Feed", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  
            break
        elif key == ord('c'): 
            calibration_mode = not calibration_mode
            print(f"Calibration mode: {calibration_mode}")
        elif key == ord('+') or key == ord('='):  
            current_real_distance += 0.05
            print(f"Set real distance: {current_real_distance:.2f} m")
        elif key == ord('-') or key == ord('_'):  
            current_real_distance = max(0.05, current_real_distance - 0.05)
            print(f"Set real distance: {current_real_distance:.2f} m")
        elif key == 13 and calibration_mode and len(points) == 2:  
            raw_depth_avg = (last_depth_map[points[0][1], points[0][0]] + last_depth_map[points[1][1], points[1][0]]) / 2
            
            calibration = add_calibration_point(calibration, raw_depth_avg, current_real_distance)
            print(f"Added calibration point: raw_depth={raw_depth_avg:.4f}, real_distance={current_real_distance:.2f}m")
            print(f"Total calibration points: {len(calibration['calibration_points'])}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()