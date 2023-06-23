# This script calculates a YOLOv5's average inference time on the SKU500 dataset.
# It takes the exact same arguments as yolov5/val.py

# Import val.py
import sys
sys.path.append(r'.\yolov5')
import val

num_val_runs = 25 # The number of validation runs to take an average over

if __name__ == '__main__':
    total_time = 0
    for run in range(num_val_runs):
        # Display progress
        print(f"Progress: {run + 1}/{num_val_runs} ")

        # Simply pass the arguments throught to val.py
        _, _, time_data = val.run(**vars(val.parse_opt()))

        # Extract time data
        total_time += time_data[1]
    
    # Find the average inference time
    average_time = total_time / num_val_runs
    print(f"Average inference time: {average_time:.2f}")
    print(f"Average FPS: {1000 / average_time:.2f}")